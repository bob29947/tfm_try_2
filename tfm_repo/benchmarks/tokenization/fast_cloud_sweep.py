#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Focused four-L4 tuning sweep for the direct S3 tokenization path.

The command is intended to run on the head node of the existing four-node AWS
cluster.  It keeps one Ray cluster alive, but creates a fresh set of four GPU
actors for every configuration/phase.  A small first-touch workload warms each
actor before its measured train-only trial.  Trial objects are always deleted;
only the optional, fully validated final arm is retained.

The default workflow has two selection stages:

* screen each configuration using at most 32 train row groups per key range;
* rerun the fastest two configurations on the complete train split.

This matters because the complete train partitions contain roughly 46--47 row
groups, so their final batch geometry is different from the bounded screen.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


TFM_ROOT = Path(__file__).resolve().parents[2]
if str(TFM_ROOT) not in sys.path:
    sys.path.insert(0, str(TFM_ROOT))

from benchmarks.tokenization.cloud_benchmark import (
    DEFAULT_INPUT_URI,
    DEFAULT_OUTPUT_ROOT,
    EXPECTED_ROWS,
    EXPECTED_SEQUENCES,
    S3Store,
    atomic_write_json,
    collect_cluster_metadata,
    execute_benchmark,
    parse_args as parse_benchmark_args,
    parse_s3_uri,
    release_fast_actors,
    s3_join,
    s3_prefixes_overlap,
    split_input_uris,
    utc_now,
    validate_cluster_shape,
    validate_fast_output_files,
    validate_parquet_footer_contract,
    validate_run_id,
)


LOGGER = logging.getLogger("tfm.fast_cloud_sweep")
MIB = 1 << 20
DEFAULT_SCREEN_ROW_GROUPS = 32
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class TrialConfig:
    """One actor-local I/O/batching configuration."""

    name: str
    row_groups_per_batch: int = 16
    s3_connections: int = 8
    kvikio_task_size_bytes: int = 4 * MIB
    write_threads: int = 4
    output_shard_size_bytes: int = 128 * MIB
    overlap_split_writes: bool = False

    def validate(self) -> "TrialConfig":
        if not _NAME_RE.fullmatch(self.name):
            raise ValueError(f"Unsafe trial name: {self.name!r}")
        for field in (
            "row_groups_per_batch",
            "s3_connections",
            "kvikio_task_size_bytes",
            "write_threads",
            "output_shard_size_bytes",
        ):
            if int(getattr(self, field)) < 1:
                raise ValueError(f"{field} must be positive for {self.name!r}")
        # Train-only trials cannot exercise cross-split write overlap.  Keeping
        # this false prevents a trial label from claiming a behavior it did not
        # measure; the final full arm has a separate opt-in flag.
        if self.overlap_split_writes:
            raise ValueError("train-only trial configs cannot overlap split writes")
        return self


def default_trial_configs() -> tuple[TrialConfig, ...]:
    """Return a deliberately small read-focused matrix.

    Row-group batching is tested before increasing remote concurrency.  The
    The remaining variants cover the evidenced KvikIO thread/task-size range,
    a one-call 48-row-group train batch, and an eight-writer candidate without
    turning the sweep into a full Cartesian search.
    """

    return (
        TrialConfig("baseline-rg16-c8-t4", 16, 8, 4 * MIB, 4, 128 * MIB),
        TrialConfig("rg32-c8-t4", 32, 8, 4 * MIB, 4, 128 * MIB),
        TrialConfig("rg32-c16-t4", 32, 16, 4 * MIB, 4, 128 * MIB),
        TrialConfig("rg32-c32-t4", 32, 32, 4 * MIB, 4, 128 * MIB),
        TrialConfig("rg32-c64-t4", 32, 64, 4 * MIB, 4, 128 * MIB),
        TrialConfig("rg32-c32-t0p5", 32, 32, MIB // 2, 4, 128 * MIB),
        TrialConfig("rg32-c32-t1", 32, 32, 1 * MIB, 4, 128 * MIB),
        TrialConfig("rg32-c32-t2", 32, 32, 2 * MIB, 4, 128 * MIB),
        TrialConfig("rg32-c32-t16", 32, 32, 16 * MIB, 4, 128 * MIB),
        TrialConfig("rg32-c64-t1", 32, 64, 1 * MIB, 4, 128 * MIB),
        TrialConfig("rg48-c32-t1", 48, 32, 1 * MIB, 4, 128 * MIB),
        TrialConfig("rg32-c32-t1-w8", 32, 32, 1 * MIB, 8, 128 * MIB),
    )


_TRIAL_KEYS = {
    "name": "name",
    "row_groups": "row_groups_per_batch",
    "row_groups_per_batch": "row_groups_per_batch",
    "connections": "s3_connections",
    "s3_connections": "s3_connections",
    "task_mib": "kvikio_task_size_bytes",
    "write_threads": "write_threads",
    "shard_mib": "output_shard_size_bytes",
}


def parse_trial_config(value: str) -> TrialConfig:
    """Parse ``name=...,row_groups=...,connections=...,task_mib=...``."""

    values: dict[str, Any] = {}
    for component in value.split(","):
        key, separator, raw = component.strip().partition("=")
        if not separator or not key or not raw:
            raise argparse.ArgumentTypeError(
                "trial configs must be comma-separated key=value fields"
            )
        if key not in _TRIAL_KEYS:
            raise argparse.ArgumentTypeError(f"unknown trial config field {key!r}")
        field = _TRIAL_KEYS[key]
        if field in values:
            raise argparse.ArgumentTypeError(f"duplicate trial config field {key!r}")
        if field == "name":
            values[field] = raw
            continue
        try:
            number = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{key} must be an integer") from exc
        if key in {"task_mib", "shard_mib"}:
            number *= MIB
        values[field] = number
    if "name" not in values:
        raise argparse.ArgumentTypeError("trial config requires name=...")
    try:
        return TrialConfig(**values).validate()
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def validate_trial_configs(configs: Sequence[TrialConfig]) -> tuple[TrialConfig, ...]:
    if not configs:
        raise ValueError("at least one trial config is required")
    validated = tuple(config.validate() for config in configs)
    names = [config.name for config in validated]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate trial config names: {duplicates}")
    return validated


def trial_prefix(output_root: str, run_id: str, phase: str, name: str) -> str:
    validate_run_id(run_id)
    if phase not in {"screen", "confirmation"}:
        raise ValueError(f"unknown tuning phase: {phase!r}")
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"unsafe trial name: {name!r}")
    return s3_join(output_root, run_id, "tuning", phase, name)


def choose_candidates(
    trials: Sequence[Mapping[str, Any]],
    *,
    count: int,
    forced_name: str | None = None,
) -> list[str]:
    """Select the lowest measured runner totals among safe trials."""

    successful = [
        trial
        for trial in trials
        if trial.get("status") == "succeeded" and trial.get("safe") is True
    ]
    by_name = {str(trial["config"]["name"]): trial for trial in successful}
    if forced_name is not None:
        if forced_name not in by_name:
            raise RuntimeError(
                f"forced config {forced_name!r} did not complete a safe screen trial"
            )
        return [forced_name]
    if count < 0:
        raise ValueError("confirmation candidate count cannot be negative")
    successful.sort(
        key=lambda trial: (
            float(trial["timings"]["runner_total_s"]),
            str(trial["config"]["name"]),
        )
    )
    return [str(trial["config"]["name"]) for trial in successful[:count]]


def choose_winner(
    screen_trials: Sequence[Mapping[str, Any]],
    confirmation_trials: Sequence[Mapping[str, Any]],
    *,
    forced_name: str | None = None,
) -> Mapping[str, Any]:
    """Prefer full-train evidence; fall back to the bounded screen if disabled."""

    evidence = list(confirmation_trials) or list(screen_trials)
    candidates = [
        trial
        for trial in evidence
        if trial.get("status") == "succeeded" and trial.get("safe") is True
    ]
    if forced_name is not None:
        candidates = [
            trial for trial in candidates if trial["config"]["name"] == forced_name
        ]
    if not candidates:
        raise RuntimeError("no safe tuning trial is available for winner selection")
    return min(
        candidates,
        key=lambda trial: (
            float(trial["timings"]["runner_total_s"]),
            str(trial["config"]["name"]),
        ),
    )


def _fast_api():
    from src.tokenization.parquet_runner import run_s3_tokenization

    return run_s3_tokenization


def _create_trial_actors(config: TrialConfig, aws_region: str) -> list[Any]:
    """Return handles immediately so every failure path can kill all actors."""

    from src.tokenization.fast_actor import FastParquetSplitTokenizer
    from src.tokenization.parquet_runner import create_gpu_parquet_actors

    processor_kwargs = {
        "output_dtype": "uint16",
        "output_format": "binary-tensor",
        "row_groups_per_batch": config.row_groups_per_batch,
        "arrow_cpu_threads": 16,
        "write_threads": config.write_threads,
        "output_shard_size_bytes": config.output_shard_size_bytes,
        "s3_mode": True,
        "aws_region": aws_region,
        "s3_connections": config.s3_connections,
        "kvikio_task_size_bytes": config.kvikio_task_size_bytes,
        "overlap_split_writes": config.overlap_split_writes,
        "require_kvikio": True,
    }
    # Do not wait here. The warmup call's ready check is still timed as warmup,
    # while the caller already owns handles if one actor fails initialization.
    return list(
        create_gpu_parquet_actors(
            4,
            FastParquetSplitTokenizer,
            processor_kwargs=processor_kwargs,
            num_gpus_per_actor=1.0,
            num_cpus_per_actor=16,
            wait_until_ready=False,
        )
    )


def _invoke_fast(
    ray_module: Any,
    *,
    train_uri: str,
    output_uri: str,
    config: TrialConfig,
    actor_handles: Sequence[Any] | None,
    maximum_row_groups: int | None,
    aws_region: str,
) -> dict[str, Any]:
    api = _fast_api()
    kwargs: dict[str, Any] = {
        "actors": 4,
        "cpus_per_actor": 16,
        "gpus_per_actor": 1.0,
        "row_groups_per_batch": config.row_groups_per_batch,
        "write_threads": config.write_threads,
        "output_shard_size_bytes": config.output_shard_size_bytes,
        "actor_handles": actor_handles,
        "splits": ("train",),
        "processor_kwargs": {
            "output_format": "binary-tensor",
            "output_dtype": "uint16",
        },
        "max_row_groups_per_partition": maximum_row_groups,
        "aws_region": aws_region,
        "s3_connections": config.s3_connections,
        "kvikio_task_size_bytes": config.kvikio_task_size_bytes,
        "overlap_split_writes": config.overlap_split_writes,
    }
    missing = sorted(set(kwargs) - set(inspect.signature(api).parameters))
    if missing:
        raise RuntimeError(
            "Fast S3 API does not implement the tuning contract; missing "
            f"parameters: {missing}"
        )
    result = api(
        ray_module,
        {"train": train_uri},
        output_uri,
        **kwargs,
    )
    required = {
        "actors",
        "actor_stats",
        "backend_info",
        "config",
        "output_files",
        "raw_rows",
        "sequence_counts",
        "stage_timings",
        "writes_success_markers",
    }
    missing_result = sorted(required - set(result))
    if missing_result:
        raise RuntimeError(f"Fast S3 API result is missing: {missing_result}")
    if result["writes_success_markers"] is not False:
        raise RuntimeError("tuning trials must not write commit markers")
    if len(result["actors"]) != 4:
        raise RuntimeError(f"expected four fast actors, got {len(result['actors'])}")
    return result


def _flatten_actor_stats(core: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [stat for result in core.get("actor_stats", ()) for stat in result]


def summarize_trial_core(core: Mapping[str, Any], *, output_bytes: int) -> dict[str, Any]:
    """Extract comparable wall, critical-path, aggregate, and peak metrics."""

    timings = core["stage_timings"]
    stats = _flatten_actor_stats(core)
    metric_names = (
        "elapsed_s",
        "read_s",
        "tokenize_s",
        "sort_s",
        "sequence_s",
        "compute_s",
        "write_s",
        "write_wait_s",
        "write_overlap_s",
    )
    actor_critical = {
        name: max((float(stat.get(name, 0.0)) for stat in stats), default=0.0)
        for name in metric_names
    }
    actor_aggregate = {
        name: sum(float(stat.get(name, 0.0)) for stat in stats)
        for name in metric_names
    }
    peak_keys = (
        "peak_gpu_memory_bytes",
        "gpu_peak_memory_bytes",
        "peak_device_memory_bytes",
    )
    peak_values = [
        int(stat[key])
        for stat in stats
        for key in peak_keys
        if stat.get(key) is not None
    ]
    total_values = [
        int(stat["gpu_total_memory_bytes"])
        for stat in stats
        if stat.get("gpu_total_memory_bytes") is not None
    ]
    peak_gpu = max(peak_values) if peak_values else None
    gpu_total = min(total_values) if total_values else None
    raw_rows = int(core["raw_rows"]["train"])
    sequences = int(core["sequence_counts"]["train"])
    runner_total = float(timings["total_s"])
    return {
        "runner_total_s": runner_total,
        "plan_s": float(timings.get("plan_s", 0.0)),
        "process_wall_s": float(timings.get("process_s", 0.0)),
        "split_wall_s": float(timings.get("split_wall_s", {}).get("train", 0.0)),
        "actor_critical_path_s": actor_critical,
        "actor_aggregate_s": actor_aggregate,
        "reported_component_aggregate_s": {
            name: float(timings.get(name, 0.0))
            for name in ("read_s", "tokenize_s", "sort_s", "sequence_s", "write_s")
        },
        "peak_gpu_memory_bytes": peak_gpu,
        "gpu_total_memory_bytes": gpu_total,
        "peak_gpu_memory_fraction": (
            peak_gpu / gpu_total if peak_gpu is not None and gpu_total else None
        ),
        "raw_rows": raw_rows,
        "sequences": sequences,
        "rows_per_s": raw_rows / runner_total if runner_total else None,
        "sequences_per_s": sequences / runner_total if runner_total else None,
        "output_bytes": int(output_bytes),
    }


def _validate_backend(core: Mapping[str, Any], config: TrialConfig) -> None:
    backends = list(core.get("backend_info", ()))
    if len(backends) != 4:
        raise RuntimeError(f"expected four actor backend reports, got {len(backends)}")
    node_ids = {str(item.get("node_id")) for item in backends}
    if len(node_ids) != 4 or "None" in node_ids:
        raise RuntimeError(f"fast actors were not placed one per node: {backends!r}")
    for backend in backends:
        if backend.get("read_backend") != "cudf-kvikio":
            raise RuntimeError(f"KvikIO fail-closed backend is inactive: {backend!r}")
        realized_threads = backend.get(
            "kvikio_num_threads", backend.get("s3_connections")
        )
        if int(realized_threads) != config.s3_connections:
            raise RuntimeError(
                f"KvikIO thread mismatch: requested {config.s3_connections}, "
                f"realized {realized_threads!r}"
            )
        realized_task = backend.get("kvikio_task_size_bytes")
        if realized_task is not None and int(realized_task) != config.kvikio_task_size_bytes:
            raise RuntimeError(
                f"KvikIO task-size mismatch: requested {config.kvikio_task_size_bytes}, "
                f"realized {realized_task!r}"
            )


def cleanup_trial_prefix(store: Any, prefix: str) -> dict[str, Any]:
    """Delete visible objects and abort invisible multipart uploads."""

    errors: list[str] = []
    deleted = 0
    aborted = 0
    try:
        deleted = int(store.delete_prefix(prefix))
    except Exception as exc:
        errors.append(f"delete: {type(exc).__name__}: {exc}")
    try:
        aborted = int(store.abort_multipart_uploads(prefix))
    except Exception as exc:
        errors.append(f"multipart: {type(exc).__name__}: {exc}")
    try:
        remaining = list(store.list_files(prefix))
        if remaining:
            errors.append(f"{len(remaining)} visible objects remain")
    except Exception as exc:
        errors.append(f"verify: {type(exc).__name__}: {exc}")
    result = {
        "deleted_objects": deleted,
        "aborted_multipart_uploads": aborted,
        "errors": errors,
    }
    if errors:
        raise RuntimeError(f"trial cleanup failed for {prefix}: {errors}")
    return result


def run_train_trial(
    ray_module: Any,
    store: Any,
    *,
    train_uri: str,
    output_root: str,
    run_id: str,
    phase: str,
    config: TrialConfig,
    maximum_row_groups: int | None,
    aws_region: str,
    expected_counts: tuple[int, int] | None,
    maximum_peak_gpu_fraction: float,
) -> dict[str, Any]:
    """Warm and measure one configuration, always releasing actors/objects."""

    prefix = trial_prefix(output_root, run_id, phase, config.name)
    warmup_uri = s3_join(prefix, "warmup")
    measured_uri = s3_join(prefix, "measured")
    trial: dict[str, Any] = {
        "phase": phase,
        "config": asdict(config),
        "maximum_row_groups_per_partition": maximum_row_groups,
        "prefix": prefix,
        "status": "running",
        "safe": False,
        "started_at": utc_now(),
        "finished_at": None,
        "warmup": {},
        "timings": {},
        "validation": {},
        "cleanup": {},
        "failure": None,
    }
    actors: list[Any] = []
    failure: BaseException | None = None
    release_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    core: dict[str, Any] | None = None
    try:
        cleanup_trial_prefix(store, prefix)
        actors = _create_trial_actors(config, aws_region)
        warmup_started = time.perf_counter()
        warmup = _invoke_fast(
            ray_module,
            train_uri=train_uri,
            output_uri=warmup_uri,
            config=config,
            actor_handles=actors,
            maximum_row_groups=1,
            aws_region=aws_region,
        )
        actors = list(warmup["actors"])
        warmup_elapsed = time.perf_counter() - warmup_started
        _validate_backend(warmup, config)
        trial["warmup"] = {
            "elapsed_s": warmup_elapsed,
            "raw_rows": int(warmup["raw_rows"]["train"]),
            "sequences": int(warmup["sequence_counts"]["train"]),
            "backend_info": warmup["backend_info"],
        }

        measured_started = time.perf_counter()
        core = _invoke_fast(
            ray_module,
            train_uri=train_uri,
            output_uri=measured_uri,
            config=config,
            actor_handles=actors,
            maximum_row_groups=maximum_row_groups,
            aws_region=aws_region,
        )
        measured_elapsed = time.perf_counter() - measured_started
        _validate_backend(core, config)
        validate_fast_output_files(
            store,
            output_uri=measured_uri,
            output_files=core["output_files"],
            splits=("train",),
        )
        output_items = [
            item
            for item in store.list_files(s3_join(measured_uri, "train"))
            if item["uri"].endswith(".parquet")
        ]
        output_paths = [item["uri"] for item in output_items]
        footer = validate_parquet_footer_contract(store, output_paths, arm="fast")
        raw_rows = int(core["raw_rows"]["train"])
        sequences = int(core["sequence_counts"]["train"])
        if raw_rows <= 0 or sequences <= 0:
            raise RuntimeError(
                f"trial produced no useful data: rows={raw_rows}, sequences={sequences}"
            )
        if int(footer["rows"]) != sequences:
            raise RuntimeError(
                f"footer/output row mismatch: expected {sequences}, got {footer['rows']}"
            )
        if expected_counts is not None and (raw_rows, sequences) != expected_counts:
            raise RuntimeError(
                "trial count mismatch: "
                f"expected rows/sequences={expected_counts}, got {(raw_rows, sequences)}"
            )
        if maximum_row_groups is None:
            full_counts = (EXPECTED_ROWS["train"], EXPECTED_SEQUENCES["train"])
            if (raw_rows, sequences) != full_counts:
                raise RuntimeError(
                    f"full-train confirmation count mismatch: expected {full_counts}, "
                    f"got {(raw_rows, sequences)}"
                )
        output_bytes = sum(int(item["bytes"]) for item in output_items)
        trial["timings"] = summarize_trial_core(core, output_bytes=output_bytes)
        trial["timings"]["driver_measured_s"] = measured_elapsed
        peak_fraction = trial["timings"].get("peak_gpu_memory_fraction")
        if (
            peak_fraction is not None
            and float(peak_fraction) > maximum_peak_gpu_fraction
        ):
            raise RuntimeError(
                f"GPU memory high-water {peak_fraction:.1%} exceeds the safe "
                f"limit {maximum_peak_gpu_fraction:.1%}"
            )
        trial["validation"] = {
            "status": "passed",
            "footer": footer,
            "output_objects": len(output_items),
            "reported_config": core["config"],
            "backend_info": core["backend_info"],
        }
        trial["status"] = "succeeded"
        trial["safe"] = True
    except BaseException as exc:
        failure = exc
        trial["status"] = "failed"
        trial["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        LOGGER.exception("%s trial %s failed", phase, config.name)
    finally:
        if not actors and core is not None:
            actors = list(core.get("actors", ()))
        if actors:
            try:
                trial["actor_release"] = release_fast_actors(ray_module, actors)
            except BaseException as exc:
                release_error = exc
                trial["actor_release"] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
        try:
            trial["cleanup"] = cleanup_trial_prefix(store, prefix)
        except BaseException as exc:
            cleanup_error = exc
            trial["cleanup"] = {"error": f"{type(exc).__name__}: {exc}"}
        if release_error is not None or cleanup_error is not None:
            terminal = release_error or cleanup_error
            assert terminal is not None
            trial["status"] = "failed"
            trial["safe"] = False
            if trial["failure"] is None:
                trial["failure"] = {
                    "type": type(terminal).__name__,
                    "message": str(terminal),
                    "traceback": "post-trial release or cleanup failed",
                }
        trial["finished_at"] = utc_now()
    return trial


def final_benchmark_argv(
    args: argparse.Namespace,
    config: TrialConfig,
) -> list[str]:
    argv = [
        "--arm",
        "fast",
        "--input-uri",
        args.input_uri,
        "--output-root",
        args.output_root,
        "--run-id",
        args.run_id,
        "--results-dir",
        str(args.results_dir),
        "--ray-address",
        args.ray_address,
        "--aws-region",
        args.aws_region,
        "--expected-nodes",
        "4",
        "--expected-cpus",
        "64",
        "--expected-gpus",
        "4",
        "--skip-smoke",
        "--warmup",
        "--validate",
        "--retain-output",
        "--fast-row-groups-per-batch",
        str(config.row_groups_per_batch),
        "--fast-write-threads-per-actor",
        str(config.write_threads),
        "--fast-kvikio-remote-connections",
        str(config.s3_connections),
        "--fast-kvikio-task-size-bytes",
        str(config.kvikio_task_size_bytes),
        "--fast-output-shard-size-bytes",
        str(config.output_shard_size_bytes),
    ]
    if args.final_overlap_split_writes:
        argv.append("--fast-overlap-split-writes")
    if args.overwrite_final_output:
        argv.append("--overwrite-output")
    if args.cluster_bootstrap_seconds is not None:
        argv.extend(
            ["--cluster-bootstrap-seconds", str(args.cluster_bootstrap_seconds)]
        )
    return argv


def write_markdown_summary(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        f"# Fast S3 tuning sweep: {report['run_id']}",
        "",
        "| Phase | Config | Status | Runner total (s) | Process wall (s) | "
        "Critical read (s) | Critical write (s) | Peak GPU GiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in ("screen_trials", "confirmation_trials"):
        for trial in report.get(phase, ()):
            timings = trial.get("timings", {})
            critical = timings.get("actor_critical_path_s", {})
            peak = timings.get("peak_gpu_memory_bytes")
            peak_text = "n/a" if peak is None else f"{peak / (1 << 30):.2f}"
            lines.append(
                "| {phase} | {name} | {status} | {total} | {process} | "
                "{read} | {write} | {peak} |".format(
                    phase="screen" if phase == "screen_trials" else "full train",
                    name=trial["config"]["name"],
                    status=trial["status"],
                    total=f"{timings['runner_total_s']:.3f}" if timings else "n/a",
                    process=f"{timings['process_wall_s']:.3f}" if timings else "n/a",
                    read=f"{critical.get('read_s', 0.0):.3f}" if timings else "n/a",
                    write=f"{critical.get('write_s', 0.0):.3f}" if timings else "n/a",
                    peak=peak_text,
                )
            )
    selected = report.get("selected_config")
    lines.extend(
        [
            "",
            f"Selected: `{selected['name']}`" if selected else "Selected: none",
            "",
            "Selection uses complete-train confirmation when enabled; every tuning "
            "prefix is disposable and is checked empty after cleanup.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def execute_sweep(
    args: argparse.Namespace,
    *,
    ray_module: Any | None = None,
    store: Any | None = None,
) -> dict[str, Any]:
    configs = validate_trial_configs(args.trial_config or default_trial_configs())
    if args.select_config and args.select_config not in {c.name for c in configs}:
        raise ValueError(f"--select-config does not name a configured trial")
    tuning_root = s3_join(args.output_root, args.run_id, "tuning")
    if s3_prefixes_overlap(args.input_uri, tuning_root):
        raise ValueError("input and disposable tuning prefixes overlap")
    if store is None:
        store = S3Store()
    if ray_module is None:
        import ray as ray_module

    if not ray_module.is_initialized():
        ray_module.init(address=args.ray_address)
    cluster, environment = collect_cluster_metadata(ray_module)
    validate_cluster_shape(
        cluster, expected_nodes=4, expected_cpus=64, expected_gpus=4
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "fast-s3-cloud-tuning-sweep",
        "run_id": args.run_id,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "input_uri": args.input_uri,
        "tuning_prefix": tuning_root,
        "screen_row_groups_per_partition": args.screen_row_groups_per_partition,
        "confirmation_top": args.confirmation_top,
        "maximum_peak_gpu_fraction": args.max_peak_gpu_fraction,
        "cluster": cluster,
        "environment": environment,
        "configured_trials": [asdict(config) for config in configs],
        "screen_trials": [],
        "confirmation_trials": [],
        "selected_config": None,
        "selection_evidence": None,
        "final_run": {"status": "not-requested"},
        "failure": None,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.results_dir / "fast-sweep.json"
    markdown_path = args.results_dir / "fast-sweep.md"
    train_uri = split_input_uris(args.input_uri)["train"]
    failure: BaseException | None = None
    try:
        reference_counts: tuple[int, int] | None = None
        for config in configs:
            LOGGER.info("Screening fast config %s", config.name)
            trial = run_train_trial(
                ray_module,
                store,
                train_uri=train_uri,
                output_root=args.output_root,
                run_id=args.run_id,
                phase="screen",
                config=config,
                maximum_row_groups=args.screen_row_groups_per_partition,
                aws_region=args.aws_region,
                expected_counts=reference_counts,
                maximum_peak_gpu_fraction=args.max_peak_gpu_fraction,
            )
            report["screen_trials"].append(trial)
            if trial["safe"] and reference_counts is None:
                reference_counts = (
                    int(trial["timings"]["raw_rows"]),
                    int(trial["timings"]["sequences"]),
                )
            atomic_write_json(report_path, report)
            write_markdown_summary(report, markdown_path)

        if not any(trial["safe"] for trial in report["screen_trials"]):
            raise RuntimeError("every bounded tuning trial failed")
        candidate_names = choose_candidates(
            report["screen_trials"],
            count=args.confirmation_top,
            forced_name=args.select_config,
        )
        by_name = {config.name: config for config in configs}
        for name in candidate_names:
            LOGGER.info("Confirming fast config %s on full train", name)
            trial = run_train_trial(
                ray_module,
                store,
                train_uri=train_uri,
                output_root=args.output_root,
                run_id=args.run_id,
                phase="confirmation",
                config=by_name[name],
                maximum_row_groups=None,
                aws_region=args.aws_region,
                expected_counts=(EXPECTED_ROWS["train"], EXPECTED_SEQUENCES["train"]),
                maximum_peak_gpu_fraction=args.max_peak_gpu_fraction,
            )
            report["confirmation_trials"].append(trial)
            atomic_write_json(report_path, report)
            write_markdown_summary(report, markdown_path)

        winner = choose_winner(
            report["screen_trials"],
            report["confirmation_trials"],
            forced_name=args.select_config,
        )
        selected = by_name[str(winner["config"]["name"])]
        report["selected_config"] = asdict(selected)
        report["selection_evidence"] = {
            "phase": winner["phase"],
            "runner_total_s": winner["timings"]["runner_total_s"],
            "process_wall_s": winner["timings"]["process_wall_s"],
            "peak_gpu_memory_bytes": winner["timings"].get(
                "peak_gpu_memory_bytes"
            ),
        }
        atomic_write_json(report_path, report)
        write_markdown_summary(report, markdown_path)

        if args.run_final_full:
            final_config = replace(
                selected,
                overlap_split_writes=False,
            )
            final_args = parse_benchmark_args(final_benchmark_argv(args, final_config))
            LOGGER.info("Running final full fast arm with %s", selected.name)
            final_result = execute_benchmark(
                final_args, ray_module=ray_module, store=store
            )
            report["final_run"] = {
                "status": final_result["status"],
                "result_path": str(final_args.results_dir / "fast.json"),
                "output_uri": final_result["uris"]["output"],
                "pipeline_elapsed_s": final_result["pipeline"]["elapsed_s"],
                "validation_status": final_result["validation"]["status"],
                "overlap_split_writes": bool(args.final_overlap_split_writes),
            }
        report["status"] = "succeeded"
    except BaseException as exc:
        failure = exc
        report["status"] = "failed"
        report["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        LOGGER.exception("Fast cloud tuning sweep failed")
    finally:
        try:
            report["final_tuning_cleanup"] = cleanup_trial_prefix(
                store, tuning_root
            )
        except BaseException as exc:
            if failure is None:
                failure = exc
                report["status"] = "failed"
                report["failure"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": "final tuning-prefix cleanup failed",
                }
            else:
                report["final_tuning_cleanup"] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
        report["finished_at"] = utc_now()
        atomic_write_json(report_path, report)
        write_markdown_summary(report, markdown_path)
    if failure is not None:
        raise failure
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, type=validate_run_id)
    parser.add_argument("--input-uri", default=DEFAULT_INPUT_URI)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", "auto"))
    parser.add_argument(
        "--aws-region",
        default=os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
        ),
    )
    parser.add_argument(
        "--trial-config",
        action="append",
        type=parse_trial_config,
        help=(
            "Repeatable config: name=...,row_groups=32,connections=16,"
            "task_mib=4,write_threads=4,shard_mib=128. Defaults to a focused "
            "twelve-config matrix."
        ),
    )
    parser.add_argument(
        "--screen-row-groups-per-partition",
        type=int,
        default=DEFAULT_SCREEN_ROW_GROUPS,
    )
    parser.add_argument(
        "--confirmation-top",
        type=int,
        default=2,
        help="Rerun the fastest N safe configs on complete train; 0 disables.",
    )
    parser.add_argument(
        "--max-peak-gpu-fraction",
        type=float,
        default=0.92,
        help="Reject a trial whose actor GPU high-water exceeds this fraction.",
    )
    parser.add_argument(
        "--select-config",
        help="Force this configured name into full-train confirmation/selection.",
    )
    parser.add_argument("--run-final-full", action="store_true")
    parser.add_argument("--final-overlap-split-writes", action="store_true")
    parser.add_argument("--overwrite-final-output", action="store_true")
    parser.add_argument("--cluster-bootstrap-seconds", type=float, default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    parse_s3_uri(args.input_uri, require_key=True)
    parse_s3_uri(args.output_root, require_key=True)
    if args.screen_row_groups_per_partition < 1:
        parser.error("--screen-row-groups-per-partition must be positive")
    if args.confirmation_top < 0:
        parser.error("--confirmation-top cannot be negative")
    if not 0.0 < args.max_peak_gpu_fraction <= 1.0:
        parser.error("--max-peak-gpu-fraction must be in (0, 1]")
    if args.select_config and not _NAME_RE.fullmatch(args.select_config):
        parser.error("--select-config has an unsafe name")
    if args.final_overlap_split_writes and not args.run_final_full:
        parser.error("--final-overlap-split-writes requires --run-final-full")
    if args.results_dir is None:
        args.results_dir = Path(__file__).resolve().parent / "results" / "aws" / args.run_id
    else:
        args.results_dir = args.results_dir.expanduser().resolve()
    return args


def configure_logging(results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "fast-sweep.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path)):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    return log_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    log_path = configure_logging(args.results_dir)
    LOGGER.info("Starting focused fast S3 sweep; log=%s", log_path)
    try:
        report = execute_sweep(args)
    except BaseException:
        raise SystemExit(1)
    LOGGER.info(
        "Selected %s; final=%s",
        report["selected_config"]["name"],
        report["final_run"]["status"],
    )


if __name__ == "__main__":
    main()
