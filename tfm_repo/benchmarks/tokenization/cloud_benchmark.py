# SPDX-License-Identifier: Apache-2.0
"""S3-to-S3 tokenization benchmark orchestration.

This module deliberately keeps cloud I/O and Ray imports lazy.  Its URI,
configuration, result, cleanup, and summary helpers are therefore usable in
unit tests without AWS credentials or a running Ray cluster.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.metadata
import inspect
import json
import logging
import math
import os
import platform
import posixpath
import re
import socket
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit


LOGGER = logging.getLogger("tfm.cloud_tokenization_benchmark")

SPLITS = ("train", "val", "test")
EXPECTED_ROWS = {
    "train": 191_957_576,
    "val": 24_031_181,
    "test": 24_011_243,
}
EXPECTED_SEQUENCES = {
    "train": 614_874,
    "val": 81_993,
    "test": 82_794,
}
SEQUENCE_LENGTH = 4096
MAX_TOKEN_ID = 6250
TUNED_TARGET_ROWS_PER_BLOCK = 2_097_152
DEFAULT_INPUT_URI = "s3://bobbwang-tfm-gpu-e2e/tfm_try_2/input/temporal_split_v3"
DEFAULT_OUTPUT_ROOT = "s3://bobbwang-tfm-gpu-e2e/tfm_try_2/benchmarks"
SUCCESS_MARKER = "_SUCCESS.json"
RESULT_SCHEMA_VERSION = 1
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}" + (f"/{self.key}" if self.key else "")

    @property
    def arrow_path(self) -> str:
        return self.bucket + (f"/{self.key}" if self.key else "")


def parse_s3_uri(uri: str, *, require_key: bool = False) -> S3Uri:
    """Parse a plain S3 URI without silently accepting unsafe components."""
    parsed = urlsplit(str(uri))
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3://bucket/key URI, got {uri!r}")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError(f"S3 URI must not contain auth, query, or fragment: {uri!r}")
    key = parsed.path.lstrip("/").rstrip("/")
    if any(part in {".", ".."} for part in key.split("/") if part):
        raise ValueError(f"S3 URI contains an unsafe path component: {uri!r}")
    if require_key and not key:
        raise ValueError(f"S3 URI must include a key prefix: {uri!r}")
    return S3Uri(parsed.netloc, key)


def s3_join(root: str, *parts: object) -> str:
    base = parse_s3_uri(root)
    clean: list[str] = []
    for raw in parts:
        value = str(raw).strip("/")
        if not value:
            continue
        if any(part in {".", ".."} for part in value.split("/")):
            raise ValueError(f"Unsafe S3 key component: {raw!r}")
        clean.append(value)
    key = posixpath.join(base.key, *clean) if (base.key or clean) else ""
    return S3Uri(base.bucket, key).uri


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "--run-id must start with an alphanumeric character and contain "
            "only alphanumerics, '.', '_', or '-'"
        )
    return run_id


def derive_override_num_blocks(
    row_count: int, target_rows_per_block: int = TUNED_TARGET_ROWS_PER_BLOCK
) -> int:
    if row_count < 1:
        raise ValueError("row_count must be positive")
    if target_rows_per_block < 1:
        raise ValueError("target_rows_per_block must be positive")
    return max(1, math.ceil(row_count / target_rows_per_block))


def split_input_uris(input_uri: str) -> dict[str, str]:
    parse_s3_uri(input_uri, require_key=True)
    return {split: s3_join(input_uri, f"{split}.parquet") for split in SPLITS}


def benchmark_uris(output_root: str, run_id: str, arm: str) -> dict[str, str]:
    validate_run_id(run_id)
    if arm not in {"original", "tuned", "fast"}:
        raise ValueError(f"Unknown benchmark arm: {arm!r}")
    base = s3_join(output_root, run_id)
    return {
        "run": base,
        "output": s3_join(base, "outputs", arm),
        "warmup": s3_join(base, "warmup", arm),
        "smoke": s3_join(base, "smoke", arm),
    }


def s3_prefixes_overlap(left: str, right: str) -> bool:
    """Return whether either S3 prefix contains the other on key boundaries."""
    left_uri = parse_s3_uri(left)
    right_uri = parse_s3_uri(right)
    if left_uri.bucket != right_uri.bucket:
        return False
    left_key = left_uri.key.rstrip("/")
    right_key = right_uri.key.rstrip("/")
    if left_key == right_key:
        return True
    return bool(
        (left_key and right_key.startswith(left_key + "/"))
        or (right_key and left_key.startswith(right_key + "/"))
        or not left_key
        or not right_key
    )


def validate_uri_isolation(input_uri: str, generated_uris: Mapping[str, str]) -> None:
    for label in ("output", "warmup", "smoke"):
        generated = generated_uris[label]
        if s3_prefixes_overlap(input_uri, generated):
            raise ValueError(
                f"Input prefix {input_uri!r} overlaps generated {label} prefix "
                f"{generated!r}; refusing any cleanup or writes"
            )


def arm_configuration(arm: str) -> dict[str, Any]:
    """Return the complete, auditable configuration for one benchmark arm."""
    common: dict[str, Any] = {
        "arm": arm,
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_chunk_size": 315,
        "per_operation_runtime_env": None,
    }
    if arm in {"original", "tuned"}:
        actors = 4 if arm == "original" else 8
        gpus = 1.0 if arm == "original" else 0.5
        common.update(
            {
                "engine": "ray-data-nb02",
                "actors": actors,
                "gpus_per_actor": gpus,
                "batch_size": 16_384,
                "batch_format": "cudf",
                "merchant_hash_mode": "string_hash",
                "column_projection": None,
                "output_format": "ray-arrow-tensor",
                "output_dtype": "int64",
                "actors_reused_after_warmup": False,
                "s3_backend": "ray-data/pyarrow-s3",
                "per_split": {
                    split: {
                        "raw_rows": rows,
                        "override_num_blocks": (
                            None
                            if arm == "original"
                            else derive_override_num_blocks(rows)
                        ),
                    }
                    for split, rows in EXPECTED_ROWS.items()
                },
            }
        )
        if arm == "original":
            common["block_policy"] = "implicit Ray 128 MiB target default"
        else:
            common["target_rows_per_block"] = TUNED_TARGET_ROWS_PER_BLOCK
        return common
    if arm == "fast":
        common.update(
            {
                "engine": "s3-key-range-gpu",
                "actors": 4,
                "cpus_per_actor": 16,
                "gpus_per_actor": 1.0,
                "merchant_hash_mode": "integer_mod",
                "row_groups_per_batch": 16,
                "write_threads_per_actor": 4,
                "output_shard_size_bytes": 128 << 20,
                "output_format": "binary-tensor",
                "output_dtype": "uint16",
                "kvikio_remote_connections": 8,
                "kvikio_task_size_bytes": 4 << 20,
                "overlap_split_writes": False,
                "whole_object_fallback": False,
                "actors_reused_after_warmup": True,
                "s3_backend": "cudf-kvikio-read/pyarrow-s3-write",
            }
        )
        return common
    raise ValueError(f"Unknown benchmark arm: {arm!r}")


def realized_arm_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Apply run-specific CLI settings to the arm's recorded configuration."""

    config = arm_configuration(args.arm)
    config["aws_region"] = args.aws_region
    if args.arm == "fast":
        config.update(
            {
                "row_groups_per_batch": args.fast_row_groups_per_batch,
                "write_threads_per_actor": args.fast_write_threads_per_actor,
                "kvikio_remote_connections": args.fast_kvikio_remote_connections,
                "kvikio_task_size_bytes": args.fast_kvikio_task_size_bytes,
                "output_shard_size_bytes": args.fast_output_shard_size_bytes,
                "overlap_split_writes": args.fast_overlap_split_writes,
                "actors_reused_after_warmup": not args.skip_warmup,
                "actor_lifecycle": (
                    "fresh_actor_inclusive"
                    if args.skip_warmup
                    else "prewarmed_actor_reuse"
                ),
            }
        )
    return config


def assert_unique_output_keys(objects: Iterable[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in objects:
        key = str(item.get("uri") or item.get("key") or item.get("path") or "")
        if not key:
            raise ValueError(f"Output inventory item has no key: {item!r}")
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"Duplicate output object keys: {sorted(duplicates)!r}")


class S3Store:
    """Small Arrow S3 adapter with dependency injection for unit tests."""

    def __init__(
        self,
        filesystem: Any | None = None,
        s3_client: Any | None = None,
    ) -> None:
        if filesystem is None:
            import pyarrow.fs as pafs

            filesystem = pafs.S3FileSystem()
        self.filesystem = filesystem
        self._s3_client = s3_client

    def _selector(self, path: str):
        import pyarrow.fs as pafs

        return pafs.FileSelector(path, recursive=True, allow_not_found=True)

    @staticmethod
    def _is_file(info: Any) -> bool:
        type_name = getattr(getattr(info, "type", None), "name", "")
        return type_name.lower() == "file" or str(getattr(info, "type", "")).endswith("File")

    def list_files(self, prefix_uri: str) -> list[dict[str, Any]]:
        parsed = parse_s3_uri(prefix_uri)
        infos = self.filesystem.get_file_info(self._selector(parsed.arrow_path))
        result = [
            {
                "uri": f"s3://{info.path}",
                "key": info.path.split("/", 1)[1] if "/" in info.path else "",
                "bytes": int(info.size),
            }
            for info in infos
            if self._is_file(info)
        ]
        result.sort(key=lambda item: item["uri"])
        assert_unique_output_keys(result)
        return result

    def write_json(self, uri: str, document: Mapping[str, Any]) -> None:
        payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
        parsed = parse_s3_uri(uri, require_key=True)
        with self.filesystem.open_output_stream(parsed.arrow_path) as stream:
            stream.write(payload)

    def read_json(self, uri: str) -> dict[str, Any]:
        parsed = parse_s3_uri(uri, require_key=True)
        with self.filesystem.open_input_file(parsed.arrow_path) as stream:
            return json.loads(stream.read().decode())

    def delete_prefix(self, uri: str) -> int:
        parsed = parse_s3_uri(uri, require_key=True)
        files = self.list_files(uri)
        for item in files:
            self.filesystem.delete_file(parse_s3_uri(item["uri"]).arrow_path)
        return len(files)

    def abort_multipart_uploads(self, prefix_uri: str) -> int:
        """Abort every incomplete multipart upload below an exact key prefix."""
        parsed = parse_s3_uri(prefix_uri, require_key=True)
        if self._s3_client is None:
            import boto3

            self._s3_client = boto3.client(
                "s3",
                region_name=os.environ.get(
                    "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
                ),
            )
        prefix = parsed.key.rstrip("/") + "/"
        aborted = 0
        paginator = self._s3_client.get_paginator("list_multipart_uploads")
        for page in paginator.paginate(Bucket=parsed.bucket, Prefix=prefix):
            for upload in page.get("Uploads", ()):
                self._s3_client.abort_multipart_upload(
                    Bucket=parsed.bucket,
                    Key=upload["Key"],
                    UploadId=upload["UploadId"],
                )
                aborted += 1
        return aborted


def output_inventory(store: Any, split_uri: str) -> list[dict[str, Any]]:
    marker_uri = s3_join(split_uri, SUCCESS_MARKER)
    items = [item for item in store.list_files(split_uri) if item["uri"] != marker_uri]
    assert_unique_output_keys(items)
    return items


def commit_split(
    store: Any,
    split_uri: str,
    *,
    arm: str,
    split: str,
    run_id: str,
) -> dict[str, Any]:
    inventory = output_inventory(store, split_uri)
    parquet = [item for item in inventory if item["uri"].endswith(".parquet")]
    if not parquet:
        raise RuntimeError(f"No Parquet output objects found under {split_uri}")
    if len(parquet) != len(inventory):
        unexpected = [item["uri"] for item in inventory if not item["uri"].endswith(".parquet")]
        raise RuntimeError(
            f"Unexpected non-Parquet objects under {split_uri}: {unexpected!r}"
        )
    if any(int(item["bytes"]) <= 0 for item in inventory):
        raise RuntimeError(f"Empty output object found under {split_uri}")
    marker = {
        "schema_version": 1,
        "run_id": run_id,
        "arm": arm,
        "split": split,
        "committed_at": utc_now(),
        "object_count": len(inventory),
        "output_bytes": sum(int(item["bytes"]) for item in inventory),
        "objects": inventory,
    }
    store.write_json(s3_join(split_uri, SUCCESS_MARKER), marker)
    return marker


def cleanup_benchmark_prefixes(
    store: Any,
    *,
    warmup_uri: str,
    output_uri: str,
    succeeded: bool,
    retain_output: bool,
) -> dict[str, Any]:
    """Apply the benchmark retention policy without ever touching input data."""
    actions: dict[str, Any] = {
        "warmup_deleted_objects": 0,
        "warmup_aborted_multipart_uploads": 0,
        "output_deleted_objects": 0,
        "output_aborted_multipart_uploads": 0,
    }
    errors: list[str] = []
    warmup_started = time.perf_counter()
    try:
        actions["warmup_deleted_objects"] = store.delete_prefix(warmup_uri)
    except Exception as exc:
        errors.append(f"warmup: {type(exc).__name__}: {exc}")
    abort_uploads = getattr(store, "abort_multipart_uploads", None)
    if callable(abort_uploads):
        try:
            actions["warmup_aborted_multipart_uploads"] = abort_uploads(warmup_uri)
        except Exception as exc:
            errors.append(f"warmup multipart: {type(exc).__name__}: {exc}")
    actions["warmup_cleanup_elapsed_s"] = time.perf_counter() - warmup_started

    output_started = time.perf_counter()
    if not succeeded or not retain_output:
        try:
            actions["output_deleted_objects"] = store.delete_prefix(output_uri)
        except Exception as exc:
            errors.append(f"output: {type(exc).__name__}: {exc}")
    # A completed run should not have MPUs, but sweeping the unique run prefix
    # verifies that worker death cannot leave chargeable hidden uploads behind.
    if callable(abort_uploads):
        try:
            actions["output_aborted_multipart_uploads"] = abort_uploads(output_uri)
        except Exception as exc:
            errors.append(f"output multipart: {type(exc).__name__}: {exc}")
    actions["output_cleanup_elapsed_s"] = time.perf_counter() - output_started
    if errors:
        raise RuntimeError("; ".join(errors))
    return actions


def new_result(
    *,
    run_id: str,
    arm: str,
    input_uri: str,
    output_uri: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "arm": arm,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "uris": {"input": input_uri, "output": output_uri},
        "config": dict(config),
        "expected": {
            "raw_rows": dict(EXPECTED_ROWS),
            "sequences": dict(EXPECTED_SEQUENCES),
        },
        "cluster": {},
        "environment": {},
        "smoke_test": {},
        "warmup": {},
        "pipeline": {},
        "validation": {},
        "cleanup": {},
        "failure": None,
    }


def validate_result_schema(document: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "run_id",
        "arm",
        "status",
        "started_at",
        "finished_at",
        "uris",
        "config",
        "expected",
        "cluster",
        "environment",
        "smoke_test",
        "warmup",
        "pipeline",
        "validation",
        "cleanup",
        "failure",
    }
    missing = required - set(document)
    if missing:
        raise ValueError(f"Benchmark result is missing fields: {sorted(missing)}")
    if document["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported result schema: {document['schema_version']!r}")
    if document["arm"] not in {"original", "tuned", "fast"}:
        raise ValueError(f"Invalid result arm: {document['arm']!r}")


def _package_versions() -> dict[str, str]:
    packages = ("ray", "pyarrow", "numpy", "cudf-cu12", "cupy-cuda12x", "kvikio-cu12")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def git_sha() -> str:
    configured = os.environ.get("TFM_BENCH_GIT_SHA")
    if configured:
        return configured
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unavailable"


def _worker_environment_probe() -> dict[str, Any]:
    gpu = "unavailable"
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception:
        pass
    try:
        import ray

        context = ray.get_runtime_context()
        node_id = context.get_node_id()
    except Exception:
        node_id = "unavailable"
    return {
        "node_id": node_id,
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "gpu": gpu,
    }


def collect_cluster_metadata(ray_module: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    nodes = [node for node in ray_module.nodes() if node.get("Alive")]
    node_records = [
        {
            "node_id": node.get("NodeID"),
            "node_name": node.get("NodeName"),
            "address": node.get("NodeManagerAddress"),
            "resources": node.get("Resources", {}),
        }
        for node in nodes
    ]
    probes: list[dict[str, Any]] = []
    try:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        remote_probe = ray_module.remote(num_cpus=0)(_worker_environment_probe)
        refs = [
            remote_probe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"], soft=False
                )
            ).remote()
            for node in nodes
        ]
        probes = ray_module.get(refs)
    except Exception as exc:
        probes = [{"probe_error": f"{type(exc).__name__}: {exc}"}]
    cluster = {
        "alive_nodes": len(nodes),
        "resources": dict(ray_module.cluster_resources()),
        "available_resources": dict(ray_module.available_resources()),
        "nodes": node_records,
    }
    environment = {
        "driver": _worker_environment_probe(),
        "workers": probes,
        "git_sha": git_sha(),
    }
    return cluster, environment


def collect_actor_placement(ray_module: Any) -> dict[str, Any]:
    """Best-effort placement snapshot for Ray Data and fast GPU actors."""
    try:
        from ray.util.state import list_actors

        job_id = ray_module.get_runtime_context().get_job_id()
        states = list_actors(
            filters=[("job_id", "=", job_id)],
            detail=True,
            limit=1000,
            timeout=15,
        )
        records = []
        for state in states:
            class_name = state.class_name or ""
            if "MapWorker" not in class_name and "GPUParquet" not in class_name:
                continue
            records.append(
                {
                    "actor_id": state.actor_id,
                    "class_name": class_name,
                    "state": state.state,
                    "node_id": state.node_id,
                    "pid": state.pid,
                    "required_resources": state.required_resources,
                }
            )
        return {"status": "collected", "actors": records}
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def release_fast_actors(
    ray_module: Any,
    actors: Sequence[Any],
    *,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Terminate full-node fast actors and wait for validation resources.

    Each measured fast actor reserves all 16 logical CPUs on its L4 node.  The
    output validator uses Ray Data CPU tasks, so merely dropping Python handles
    can deadlock validation behind still-live actors.
    """
    started = time.perf_counter()
    unique: list[Any] = []
    seen: set[int] = set()
    for actor in actors:
        identity = id(actor)
        if identity not in seen:
            unique.append(actor)
            seen.add(identity)
    for actor in unique:
        ray_module.kill(actor, no_restart=True)

    cluster = ray_module.cluster_resources()
    expected_cpus = float(cluster.get("CPU", 0.0))
    expected_gpus = float(cluster.get("GPU", 0.0))
    deadline = time.monotonic() + timeout_s
    available: Mapping[str, Any] = {}
    while time.monotonic() < deadline:
        available = ray_module.available_resources()
        if (
            float(available.get("CPU", 0.0)) >= expected_cpus - 1e-9
            and float(available.get("GPU", 0.0)) >= expected_gpus - 1e-9
        ):
            return {
                "actors_released": len(unique),
                "elapsed_s": time.perf_counter() - started,
                "available_resources": dict(available),
            }
        time.sleep(0.25)
    raise TimeoutError(
        "Fast actors did not release cluster resources before validation: "
        f"available={dict(available)}, cluster={dict(cluster)}"
    )


def validate_cluster_shape(
    cluster: Mapping[str, Any], *, expected_nodes: int, expected_cpus: float, expected_gpus: float
) -> None:
    actual_nodes = int(cluster.get("alive_nodes", 0))
    resources = cluster.get("resources", {})
    actual_cpus = float(resources.get("CPU", 0))
    actual_gpus = float(resources.get("GPU", 0))
    errors = []
    if actual_nodes != expected_nodes:
        errors.append(f"expected {expected_nodes} alive nodes, found {actual_nodes}")
    if abs(actual_cpus - expected_cpus) > 1e-9:
        errors.append(f"expected {expected_cpus:g} CPUs, found {actual_cpus:g}")
    if abs(actual_gpus - expected_gpus) > 1e-9:
        errors.append(f"expected {expected_gpus:g} GPUs, found {actual_gpus:g}")
    if errors:
        raise RuntimeError("Cluster shape check failed: " + "; ".join(errors))


def _s3_smoke_worker(
    read_uri: str,
    write_uri: str,
    require_kvikio: bool,
    row_group: int,
) -> dict[str, Any]:
    if require_kvikio:
        import botocore.session

        credentials = botocore.session.get_session().get_credentials()
        if credentials is None:
            raise RuntimeError("Instance-role credentials are unavailable")
        frozen = credentials.get_frozen_credentials()
        os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
        if frozen.token:
            os.environ["AWS_SESSION_TOKEN"] = frozen.token
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
        os.environ["KVIKIO_NTHREADS"] = "8"

    import pyarrow.fs as pafs

    read_path = parse_s3_uri(read_uri, require_key=True).arrow_path
    write_path = parse_s3_uri(write_uri, require_key=True).arrow_path
    fs = pafs.S3FileSystem()
    with fs.open_input_file(read_path) as stream:
        if not stream.read(1):
            raise RuntimeError(f"S3 smoke input is empty: {read_uri}")
    with fs.open_output_stream(write_path) as stream:
        stream.write(b"tfm-s3-smoke\n")
    fs.delete_file(write_path)
    if fs.get_file_info(write_path).type != pafs.FileType.NotFound:
        raise RuntimeError(f"S3 smoke object still exists after delete: {write_uri}")
    kvikio: dict[str, Any] = {"required": require_kvikio}
    if require_kvikio:
        import kvikio as kvikio_module
        from kvikio.remote_file import is_remote_file_available

        kvikio["version"] = getattr(kvikio_module, "__version__", "unknown")
        kvikio["remote_file_available"] = bool(is_remote_file_available())
        if not kvikio["remote_file_available"]:
            raise RuntimeError("KvikIO RemoteFile support is unavailable")
        import cudf

        cudf.set_option("kvikio_remote_io", True)
        if not cudf.get_option("kvikio_remote_io"):
            raise RuntimeError("cuDF KvikIO remote I/O is inactive")
        frame = cudf.read_parquet(
            read_uri,
            engine="cudf",
            columns=["User"],
            row_groups=[row_group],
            dataset_kwargs={"partitioning": None},
            use_pandas_metadata=False,
            categorical_partitions=False,
        )
        kvikio["rows"] = len(frame)
        kvikio["cudf_option"] = True
        if not len(frame):
            raise RuntimeError("KvikIO smoke read returned no rows")
    return {
        "hostname": socket.gethostname(),
        "read_uri": read_uri,
        "write_uri": write_uri,
        "kvikio": kvikio,
    }


def run_s3_smoke_test(
    ray_module: Any,
    *,
    read_uri: str,
    write_prefix: str,
    require_kvikio: bool,
) -> dict[str, Any]:
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    nodes = [node for node in ray_module.nodes() if node.get("Alive")]
    remote_probe = ray_module.remote(
        num_cpus=1,
        num_gpus=1 if require_kvikio else 0,
    )(_s3_smoke_worker)
    refs = []
    for index, node in enumerate(nodes):
        write_uri = s3_join(write_prefix, f"{node['NodeID']}-{uuid.uuid4().hex}.txt")
        refs.append(
            remote_probe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"], soft=False
                )
            ).remote(read_uri, write_uri, require_kvikio, index)
        )
    started = time.perf_counter()
    probes = ray_module.get(refs)
    return {"status": "passed", "elapsed_s": time.perf_counter() - started, "nodes": probes}


def _legacy_tokenization_pipeline(
    ray_module: Any,
    *,
    input_uri: str,
    output_uri: str,
    config: Mapping[str, Any],
    override_num_blocks: int | None,
    warmup_rows: int | None,
) -> dict[str, Any]:
    from src.tokenization.legacy import GPUTokenizer, build_sequences

    read_kwargs: dict[str, Any] = {}
    if override_num_blocks is not None:
        read_kwargs["override_num_blocks"] = override_num_blocks
    dataset = ray_module.data.read_parquet(input_uri, **read_kwargs)
    if warmup_rows is not None:
        actors = int(config["actors"])
        dataset = dataset.limit(warmup_rows * actors).repartition(actors)
    tokenized = dataset.map_batches(
        GPUTokenizer,
        fn_constructor_kwargs={
            "merchant_hash_size": 2000,
            "merchant_hash_mode": "string_hash",
        },
        batch_size=int(config["batch_size"]),
        compute=ray_module.data.ActorPoolStrategy(size=int(config["actors"])),
        num_gpus=float(config["gpus_per_actor"]),
        batch_format="cudf",
        runtime_env=None,
    )
    sequences = tokenized.groupby("uc_key").map_groups(
        build_sequences, batch_format="numpy"
    )
    started = time.perf_counter()
    sequences.write_parquet(output_uri)
    elapsed = time.perf_counter() - started
    try:
        stats = sequences.stats()
    except Exception:
        stats = "unavailable"
    del sequences, tokenized, dataset
    gc.collect()
    return {"write_execution_s": elapsed, "ray_stats": stats}


def run_legacy_warmup(
    ray_module: Any,
    *,
    input_uris: Mapping[str, str],
    warmup_uri: str,
    config: Mapping[str, Any],
    warmup_rows: int,
) -> dict[str, Any]:
    # Override block count only to guarantee at least one warmup block per actor;
    # it is not part of the measured arm's block policy.
    return _legacy_tokenization_pipeline(
        ray_module,
        input_uri=input_uris["train"],
        output_uri=s3_join(warmup_uri, "train"),
        config=config,
        override_num_blocks=int(config["actors"]),
        warmup_rows=warmup_rows,
    )


def run_legacy_measured(
    ray_module: Any,
    store: Any,
    *,
    arm: str,
    run_id: str,
    input_uris: Mapping[str, str],
    output_uri: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    split_results: dict[str, Any] = {}
    markers: dict[str, Any] = {}
    commit_total_s = 0.0
    started = time.perf_counter()
    for split in SPLITS:
        split_started = time.perf_counter()
        split_config = config["per_split"][split]
        stage = _legacy_tokenization_pipeline(
            ray_module,
            input_uri=input_uris[split],
            output_uri=s3_join(output_uri, split),
            config=config,
            override_num_blocks=split_config["override_num_blocks"],
            warmup_rows=None,
        )
        commit_started = time.perf_counter()
        markers[split] = commit_split(
            store,
            s3_join(output_uri, split),
            arm=arm,
            split=split,
            run_id=run_id,
        )
        commit_s = time.perf_counter() - commit_started
        commit_total_s += commit_s
        split_results[split] = {
            "elapsed_s": time.perf_counter() - split_started,
            "raw_rows": EXPECTED_ROWS[split],
            "expected_sequences": EXPECTED_SEQUENCES[split],
            "stage_timings": stage,
            "commit_s": commit_s,
            "output_bytes": markers[split]["output_bytes"],
        }
    elapsed = time.perf_counter() - started
    return {
        "elapsed_s": elapsed,
        "commit_s": commit_total_s,
        "splits": split_results,
        "success_markers": markers,
        "output_bytes": sum(marker["output_bytes"] for marker in markers.values()),
    }


def _fast_api():
    from src.tokenization.parquet_runner import run_s3_tokenization

    return run_s3_tokenization


def run_fast_stage(
    ray_module: Any,
    *,
    input_uris: Mapping[str, str],
    output_uri: str,
    config: Mapping[str, Any],
    actor_handles: Sequence[Any] | None,
    warmup: bool,
) -> dict[str, Any]:
    """Narrow adapter around the application-local S3 fast runner."""
    api = _fast_api()
    kwargs: dict[str, Any] = {
        "actors": int(config["actors"]),
        "cpus_per_actor": int(config["cpus_per_actor"]),
        "gpus_per_actor": float(config["gpus_per_actor"]),
        "row_groups_per_batch": int(config["row_groups_per_batch"]),
        "write_threads": int(config["write_threads_per_actor"]),
        "output_shard_size_bytes": int(config["output_shard_size_bytes"]),
        "actor_handles": actor_handles,
        "splits": ("train",) if warmup else SPLITS,
        "processor_kwargs": {
            "output_format": "binary-tensor",
            "output_dtype": "uint16",
        },
        "aws_region": config.get("aws_region", "us-west-2"),
        "s3_connections": int(config["kvikio_remote_connections"]),
        "kvikio_task_size_bytes": int(config["kvikio_task_size_bytes"]),
        "overlap_split_writes": bool(config.get("overlap_split_writes", False)),
    }
    parameters = inspect.signature(api).parameters
    required_parameters = set(kwargs)
    required_parameters.add("max_row_groups_per_partition")
    missing_parameters = sorted(required_parameters - set(parameters))
    if missing_parameters:
        raise RuntimeError(
            "Fast S3 API is incompatible with the audited benchmark adapter; "
            f"missing parameters: {missing_parameters}"
        )
    if warmup:
        kwargs["max_row_groups_per_partition"] = 1
    else:
        kwargs["max_row_groups_per_partition"] = None
    result = api(ray_module, input_uris, output_uri, **kwargs)
    if not isinstance(result, dict):
        raise TypeError("run_s3_tokenization must return a dictionary")
    required_result_fields = {
        "actors",
        "sequence_counts",
        "raw_rows",
        "stage_timings",
        "actor_stats",
        "output_files",
        "backend_info",
        "config",
        "writes_success_markers",
    }
    missing_result_fields = sorted(required_result_fields - set(result))
    if missing_result_fields:
        raise ValueError(
            "run_s3_tokenization returned an incomplete result; missing "
            f"{missing_result_fields}"
        )
    if result["writes_success_markers"] is not False:
        raise ValueError(
            "Fast runner must leave _SUCCESS.json commits to the benchmark harness"
        )
    for field in ("sequence_counts", "raw_rows", "output_files"):
        missing_splits = [
            split for split in kwargs["splits"] if split not in result[field]
        ]
        if missing_splits:
            raise ValueError(
                f"run_s3_tokenization result {field!r} is missing splits: "
                f"{missing_splits}"
            )
    if len(result["actors"]) != int(config["actors"]):
        raise ValueError(
            f"Fast runner returned {len(result['actors'])} actors; "
            f"expected {config['actors']}"
        )
    return result


def validate_fast_output_files(
    store: Any,
    *,
    output_uri: str,
    output_files: Mapping[str, Sequence[str]],
    splits: Sequence[str],
) -> None:
    """Require the runner's completed shard list to equal the S3 inventory."""
    for split in splits:
        expected = sorted(str(uri) for uri in output_files.get(split, ()))
        if not expected:
            raise RuntimeError(f"Fast runner returned no output files for {split}")
        assert_unique_output_keys({"uri": uri} for uri in expected)
        actual = sorted(
            item["uri"]
            for item in output_inventory(store, s3_join(output_uri, split))
            if item["uri"].endswith(".parquet")
        )
        if actual != expected:
            raise RuntimeError(
                f"Fast runner output inventory mismatch for {split}: "
                f"reported={expected!r}, s3={actual!r}"
            )


def run_fast_measured(
    ray_module: Any,
    store: Any,
    *,
    run_id: str,
    input_uris: Mapping[str, str],
    output_uri: str,
    config: Mapping[str, Any],
    actor_handles: Sequence[Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    core: dict[str, Any] | None = None
    try:
        core = run_fast_stage(
            ray_module,
            input_uris=input_uris,
            output_uri=output_uri,
            config=config,
            actor_handles=actor_handles,
            warmup=False,
        )
        for split in SPLITS:
            raw_rows = int(core["raw_rows"][split])
            sequences = int(core["sequence_counts"][split])
            if raw_rows != EXPECTED_ROWS[split]:
                raise RuntimeError(
                    f"Fast runner raw-row mismatch for {split}: "
                    f"expected {EXPECTED_ROWS[split]:,}, got {raw_rows:,}"
                )
            if sequences != EXPECTED_SEQUENCES[split]:
                raise RuntimeError(
                    f"Fast runner sequence-count mismatch for {split}: "
                    f"expected {EXPECTED_SEQUENCES[split]:,}, got {sequences:,}"
                )
        inventory_started = time.perf_counter()
        validate_fast_output_files(
            store,
            output_uri=output_uri,
            output_files=core.get("output_files", {}),
            splits=SPLITS,
        )
        inventory_s = time.perf_counter() - inventory_started
        commit_started = time.perf_counter()
        # Each split has an independent prefix. Commit them concurrently so
        # S3 PUT/list latency is paid once on the end-to-end critical path,
        # while still waiting for every durable marker before returning.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(SPLITS)) as executor:
            commit_futures = {
                split: executor.submit(
                    commit_split,
                    store,
                    s3_join(output_uri, split),
                    arm="fast",
                    split=split,
                    run_id=run_id,
                )
                for split in SPLITS
            }
            markers = {
                split: commit_futures[split].result() for split in SPLITS
            }
        commit_s = time.perf_counter() - commit_started
    except BaseException:
        if core is not None:
            for actor in core.get("actors", ()):
                try:
                    ray_module.kill(actor, no_restart=True)
                except Exception:
                    pass
        raise
    assert core is not None
    elapsed = time.perf_counter() - started
    serializable_core = {key: value for key, value in core.items() if key != "actors"}
    flat_actor_stats = [
        stat
        for actor_result in core.get("actor_stats", ())
        for stat in actor_result
    ]
    split_results: dict[str, Any] = {}
    for split in SPLITS:
        split_stats = [stat for stat in flat_actor_stats if stat.get("split") == split]
        actor_critical_path_s = max(
            (float(stat.get("elapsed_s", 0.0)) for stat in split_stats), default=0.0
        )
        process_wall_s = float(
            core.get("stage_timings", {}).get("split_wall_s", {}).get(split, 0.0)
        )
        sequences = int(core["sequence_counts"][split])
        split_results[split] = {
            "raw_rows": int(core["raw_rows"][split]),
            "sequences": sequences,
            "actor_critical_path_s": actor_critical_path_s,
            "process_wall_s": process_wall_s,
            "sequences_per_s": (
                sequences / process_wall_s if process_wall_s else None
            ),
            "output_bytes": markers[split]["output_bytes"],
        }
    return {
        "elapsed_s": elapsed,
        "inventory_validation_s": inventory_s,
        "commit_s": commit_s,
        "splits": split_results,
        "success_markers": markers,
        "output_bytes": sum(marker["output_bytes"] for marker in markers.values()),
        "fast_core": serializable_core,
        "_actor_handles": list(core["actors"]),
    }


def _validate_token_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(batch["input_ids"])
    if values.dtype == object:
        values = np.stack(list(values)) if len(values) else np.empty((0, SEQUENCE_LENGTH))
    shape_ok = values.ndim == 2 and values.shape[1] == SEQUENCE_LENGTH
    return {
        "rows": np.asarray([values.shape[0]], dtype=np.int64),
        "shape_ok": np.asarray([shape_ok], dtype=np.bool_),
        "dtype": np.asarray([str(values.dtype)]),
        "minimum": np.asarray([int(values.min()) if values.size else 0], dtype=np.int64),
        "maximum": np.asarray([int(values.max()) if values.size else 0], dtype=np.int64),
    }


def validate_parquet_footer_contract(
    store: Any,
    parquet_uris: Sequence[str],
    *,
    arm: str,
) -> dict[str, Any]:
    """Inspect every physical Parquet schema, not a caller-supplied hint."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    total_rows = 0
    for uri in parquet_uris:
        parsed = parse_s3_uri(uri, require_key=True)
        with store.filesystem.open_input_file(parsed.arrow_path) as source:
            parquet = pq.ParquetFile(source)
            if parquet.metadata.num_rows <= 0:
                raise RuntimeError(f"Parquet object has no rows: {uri}")
            total_rows += int(parquet.metadata.num_rows)
            schema = parquet.schema_arrow
        if schema.names != ["input_ids"]:
            raise RuntimeError(
                f"Unexpected output columns in {uri}: {schema.names!r}"
            )
        field = schema.field("input_ids")
        metadata = field.metadata or {}
        if arm == "fast":
            if not pa.types.is_fixed_size_binary(field.type):
                raise RuntimeError(
                    f"Fast input_ids must be fixed_size_binary in {uri}, got {field.type}"
                )
            expected_width = SEQUENCE_LENGTH * np.dtype("uint16").itemsize
            if int(field.type.byte_width) != expected_width:
                raise RuntimeError(
                    f"Fast input_ids byte width in {uri} is {field.type.byte_width}, "
                    f"expected {expected_width}"
                )
            expected_metadata = {
                b"ray.data.fixed_size_binary_tensor.shape": b"[4096]",
                b"ray.data.fixed_size_binary_tensor.dtype": b"uint16",
            }
            if metadata != expected_metadata:
                raise RuntimeError(
                    f"Fast tensor metadata mismatch in {uri}: {metadata!r}"
                )
            continue

        field_type = field.type
        if isinstance(field_type, pa.ExtensionType):
            if field_type.extension_name != "ray.data.arrow_tensor_v2":
                raise RuntimeError(
                    f"Unexpected legacy tensor extension in {uri}: "
                    f"{field_type.extension_name!r}"
                )
            shape = tuple(getattr(field_type, "shape", ()))
            if shape != (SEQUENCE_LENGTH,):
                raise RuntimeError(
                    f"Legacy tensor extension mismatch in {uri}: "
                    f"shape={shape!r}"
                )
            storage_type = field_type.storage_type
        else:
            expected_metadata = {
                b"ARROW:extension:metadata": b"[4096]",
                b"ARROW:extension:name": b"ray.data.arrow_tensor_v2",
            }
            if metadata != expected_metadata:
                raise RuntimeError(
                    f"Legacy tensor metadata mismatch in {uri}: {metadata!r}"
                )
            storage_type = field_type
        if not (
            pa.types.is_list(storage_type)
            or pa.types.is_large_list(storage_type)
            or pa.types.is_fixed_size_list(storage_type)
        ) or storage_type.value_type != pa.int64():
            raise RuntimeError(
                f"Legacy input_ids storage mismatch in {uri}: {storage_type}"
            )
    return {"objects": len(parquet_uris), "rows": total_rows}


def validate_outputs(
    ray_module: Any,
    store: Any,
    *,
    arm: str,
    input_uris: Mapping[str, str],
    output_uri: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    input_count_s = 0.0
    listing_s = 0.0
    footer_s = 0.0
    content_s = 0.0
    split_results: dict[str, Any] = {}
    expected_dtype = "uint16" if arm == "fast" else "int64"
    for split in SPLITS:
        operation_started = time.perf_counter()
        raw_rows = int(ray_module.data.read_parquet(input_uris[split]).count())
        input_count_s += time.perf_counter() - operation_started
        if raw_rows != EXPECTED_ROWS[split]:
            raise RuntimeError(
                f"{split}: expected {EXPECTED_ROWS[split]:,} raw rows, got {raw_rows:,}"
            )
        split_uri = s3_join(output_uri, split)
        operation_started = time.perf_counter()
        inventory = output_inventory(store, split_uri)
        marker = store.read_json(s3_join(split_uri, SUCCESS_MARKER))
        if marker.get("objects") != inventory:
            raise RuntimeError(f"{split}: _SUCCESS.json inventory does not match S3")
        parquet_uris = [item["uri"] for item in inventory if item["uri"].endswith(".parquet")]
        if not parquet_uris:
            raise RuntimeError(f"{split}: no Parquet files")
        listing_s += time.perf_counter() - operation_started
        operation_started = time.perf_counter()
        footer = validate_parquet_footer_contract(store, parquet_uris, arm=arm)
        footer_s += time.perf_counter() - operation_started
        read_kwargs: dict[str, Any] = {}
        if arm == "fast":
            import numpy as np

            read_kwargs["tensor_column_schema"] = {
                "input_ids": (np.dtype("uint16"), (SEQUENCE_LENGTH,))
            }
        operation_started = time.perf_counter()
        checks = (
            ray_module.data.read_parquet(parquet_uris, **read_kwargs)
            .map_batches(_validate_token_batch, batch_format="numpy", batch_size=1024)
            .take_all()
        )
        content_s += time.perf_counter() - operation_started
        rows = sum(int(item["rows"]) for item in checks)
        dtypes = {str(item["dtype"]) for item in checks}
        minimum = min(int(item["minimum"]) for item in checks)
        maximum = max(int(item["maximum"]) for item in checks)
        if rows != EXPECTED_SEQUENCES[split]:
            raise RuntimeError(
                f"{split}: expected {EXPECTED_SEQUENCES[split]:,} sequences, got {rows:,}"
            )
        if not all(bool(item["shape_ok"]) for item in checks):
            raise RuntimeError(f"{split}: input_ids shape is not ({SEQUENCE_LENGTH},)")
        if dtypes != {expected_dtype}:
            raise RuntimeError(f"{split}: expected dtype {expected_dtype}, got {sorted(dtypes)}")
        if minimum < 0 or maximum > MAX_TOKEN_ID:
            raise RuntimeError(
                f"{split}: token range [{minimum}, {maximum}] is outside [0, {MAX_TOKEN_ID}]"
            )
        split_results[split] = {
            "raw_rows": raw_rows,
            "sequences": rows,
            "shape": [SEQUENCE_LENGTH],
            "dtype": expected_dtype,
            "token_min": minimum,
            "token_max": maximum,
            "object_count": len(inventory),
            "output_bytes": sum(int(item["bytes"]) for item in inventory),
            "footer": footer,
        }
    return {
        "status": "passed",
        "elapsed_s": time.perf_counter() - started,
        "stage_timings": {
            "input_row_count_s": input_count_s,
            "output_listing_s": listing_s,
            "footer_validation_s": footer_s,
            "content_validation_s": content_s,
        },
        "splits": split_results,
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(document), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def generate_comparison_summary(
    results_dir: Path, *, require_complete: bool = False
) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("original", "tuned", "fast"):
        candidates = (
            results_dir / f"{arm}.json",
            results_dir / arm / f"{arm}.json",
        )
        for path in candidates:
            if not path.exists():
                continue
            document = json.loads(path.read_text())
            if document.get("arm") == arm:
                validate_result_schema(document)
                arms[arm] = document
                break
    missing = [arm for arm in ("original", "tuned", "fast") if arm not in arms]
    if require_complete and missing:
        raise RuntimeError(
            f"Cannot summarize incomplete benchmark; missing arm results: {missing}"
        )
    unsuccessful = [
        arm for arm, document in arms.items() if document.get("status") != "succeeded"
    ]
    if require_complete and unsuccessful:
        raise RuntimeError(
            f"Cannot summarize unsuccessful benchmark arms: {unsuccessful}"
        )
    unvalidated = [
        arm
        for arm, document in arms.items()
        if document.get("validation", {}).get("status") != "passed"
    ]
    if require_complete and unvalidated:
        raise RuntimeError(
            f"Cannot summarize unvalidated benchmark arms: {unvalidated}"
        )
    run_ids = {document.get("run_id") for document in arms.values()}
    input_uris = {
        document.get("uris", {}).get("input") for document in arms.values()
    }
    if require_complete and len(run_ids) != 1:
        raise RuntimeError(
            f"Benchmark arm run IDs do not match: {sorted(map(str, run_ids))!r}"
        )
    if require_complete and len(input_uris) != 1:
        raise RuntimeError(
            f"Benchmark arm input URIs do not match: {sorted(map(str, input_uris))!r}"
        )
    original_document = arms.get("original", {})
    original_eligible = (
        original_document.get("status") == "succeeded"
        and original_document.get("validation", {}).get("status") == "passed"
    )
    original_s = (
        float(original_document.get("pipeline", {}).get("elapsed_s") or 0)
        if original_eligible
        else 0.0
    )
    rows = []
    for arm in ("original", "tuned", "fast"):
        document = arms.get(arm)
        if not document:
            continue
        pipeline = document.get("pipeline", {})
        elapsed = float(pipeline.get("elapsed_s") or 0)
        eligible = (
            document.get("status") == "succeeded"
            and document.get("validation", {}).get("status") == "passed"
        )
        rows.append(
            {
                "arm": arm,
                "status": document.get("status"),
                "warmup_s": document.get("warmup", {}).get("elapsed_s"),
                "pipeline_s": pipeline.get("elapsed_s"),
                "sequences_per_s": (
                    pipeline.get("sequences_per_s") if eligible else None
                ),
                "output_bytes": pipeline.get("output_bytes"),
                "speedup_vs_original": (
                    original_s / elapsed if eligible and original_s and elapsed else None
                ),
                "validation": document.get("validation", {}).get("status"),
            }
        )
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "rows": rows,
        "note": (
            "The fast arm uses integer-modulo merchant mapping and compact uint16 "
            "binary tensors; its end-to-end speedup includes representation and I/O "
            "contract improvements."
        ),
    }
    atomic_write_json(results_dir / "summary.json", summary)
    lines = [
        "# AWS tokenization benchmark",
        "",
        "| Arm | Status | Warmup (s) | Pipeline (s) | Sequences/s | Output GiB | Speedup | Validation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        output_gib = (float(row["output_bytes"]) / (1 << 30)) if row["output_bytes"] else None
        lines.append(
            "| {arm} | {status} | {warmup} | {pipeline} | {throughput} | {gib} | {speedup} | {validation} |".format(
                arm=row["arm"],
                status=row["status"],
                warmup=_format_number(row["warmup_s"]),
                pipeline=_format_number(row["pipeline_s"]),
                throughput=_format_number(row["sequences_per_s"], digits=0),
                gib=_format_number(output_gib),
                speedup=(f"{row['speedup_vs_original']:.2f}x" if row["speedup_vs_original"] else "—"),
                validation=row["validation"] or "—",
            )
        )
    lines.extend(["", summary["note"], ""])
    (results_dir / "summary.md").write_text("\n".join(lines))
    return summary


def _format_number(value: Any, *, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):,.{digits}f}"


def configure_logging(results_dir: Path, arm: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{arm}.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path)):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    return log_path


def execute_smoke_only(
    args: argparse.Namespace,
    *,
    ray_module: Any | None = None,
    store: Any | None = None,
) -> dict[str, Any]:
    """Run the same strict cluster/S3/KvikIO preflight without an arm pipeline."""
    uris = benchmark_uris(args.output_root, args.run_id, args.arm)
    validate_uri_isolation(args.input_uri, uris)
    input_uris = split_input_uris(args.input_uri)
    document: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "cluster": {},
        "environment": {},
        "smoke_test": {},
        "failure": None,
    }
    if store is None:
        store = S3Store()
    failure: BaseException | None = None
    try:
        if ray_module is None:
            import ray as ray_module

        if not ray_module.is_initialized():
            ray_module.init(address=args.ray_address)
        cluster, environment = collect_cluster_metadata(ray_module)
        document["cluster"] = cluster
        document["environment"] = environment
        if not args.skip_cluster_check:
            validate_cluster_shape(
                cluster,
                expected_nodes=args.expected_nodes,
                expected_cpus=args.expected_cpus,
                expected_gpus=args.expected_gpus,
            )
        document["smoke_test"] = run_s3_smoke_test(
            ray_module,
            read_uri=input_uris["train"],
            write_prefix=uris["smoke"],
            require_kvikio=True,
        )
        document["status"] = "succeeded"
    except BaseException as exc:
        failure = exc
        document["status"] = "failed"
        document["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        LOGGER.exception("Cluster/S3 smoke test failed")
    finally:
        try:
            store.delete_prefix(uris["smoke"])
        except Exception as cleanup_exc:
            document["smoke_cleanup_error"] = (
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            )
        document["finished_at"] = utc_now()
        atomic_write_json(args.results_dir / "smoke.json", document)
    if failure is not None:
        raise failure
    return document


def execute_benchmark(args: argparse.Namespace, *, ray_module: Any | None = None, store: Any | None = None) -> dict[str, Any]:
    config = realized_arm_configuration(args)
    uris = benchmark_uris(args.output_root, args.run_id, args.arm)
    validate_uri_isolation(args.input_uri, uris)
    input_uris = split_input_uris(args.input_uri)
    result = new_result(
        run_id=args.run_id,
        arm=args.arm,
        input_uri=args.input_uri,
        output_uri=uris["output"],
        config=config,
    )
    result["cluster"]["bootstrap_elapsed_s"] = args.cluster_bootstrap_seconds
    result_path = args.results_dir / f"{args.arm}.json"
    actor_handles: Sequence[Any] | None = None
    succeeded = False
    output_owned_by_attempt = False
    failure: BaseException | None = None
    if store is None:
        store = S3Store()
    try:
        if ray_module is None:
            import ray as ray_module

        ray_started = time.perf_counter()
        if not ray_module.is_initialized():
            ray_module.init(address=args.ray_address)
        result["cluster"]["ray_connect_elapsed_s"] = time.perf_counter() - ray_started
        cluster, environment = collect_cluster_metadata(ray_module)
        result["cluster"].update(cluster)
        result["environment"] = environment
        try:
            from ray.data.context import DataContext

            result["config"]["ray_target_max_block_size_bytes"] = (
                DataContext.get_current().target_max_block_size
            )
        except Exception as exc:
            result["config"]["ray_target_max_block_size_bytes"] = (
                f"unavailable: {type(exc).__name__}: {exc}"
            )
        if not args.skip_cluster_check:
            validate_cluster_shape(
                cluster,
                expected_nodes=args.expected_nodes,
                expected_cpus=args.expected_cpus,
                expected_gpus=args.expected_gpus,
            )
        existing = store.list_files(uris["output"])
        if existing:
            if not args.overwrite_output:
                raise RuntimeError(
                    f"Measured output already exists at {uris['output']}; use a new "
                    "--run-id or pass --overwrite-output"
                )
            store.delete_prefix(uris["output"])
        output_owned_by_attempt = True
        store.delete_prefix(uris["warmup"])
        if not args.skip_smoke_test:
            result["smoke_test"] = run_s3_smoke_test(
                ray_module,
                read_uri=input_uris["train"],
                write_prefix=uris["smoke"],
                require_kvikio=True,
            )
        else:
            result["smoke_test"] = {"status": "skipped"}

        if not args.skip_warmup:
            warmup_started = time.perf_counter()
            if args.arm == "fast":
                warmup_core = run_fast_stage(
                    ray_module,
                    input_uris=input_uris,
                    output_uri=uris["warmup"],
                    config=config,
                    actor_handles=None,
                    warmup=True,
                )
                actor_handles = warmup_core["actors"]
                commit_split(
                    store,
                    s3_join(uris["warmup"], "train"),
                    arm=args.arm,
                    split="train",
                    run_id=args.run_id,
                )
                warmup_details = {
                    key: value for key, value in warmup_core.items() if key != "actors"
                }
            else:
                warmup_details = run_legacy_warmup(
                    ray_module,
                    input_uris=input_uris,
                    warmup_uri=uris["warmup"],
                    config=config,
                    warmup_rows=args.warmup_rows_per_actor,
                )
                commit_split(
                    store,
                    s3_join(uris["warmup"], "train"),
                    arm=args.arm,
                    split="train",
                    run_id=args.run_id,
                )
            result["warmup"] = {
                "status": "passed",
                "elapsed_s": time.perf_counter() - warmup_started,
                "actors_initialized": config["actors"],
                "actors_reused_for_pipeline": args.arm == "fast",
                "details": warmup_details,
            }
        else:
            result["warmup"] = {"status": "skipped", "elapsed_s": 0.0}
            if args.arm == "fast":
                actor_handles = None

        if args.arm == "fast":
            result["pipeline"] = run_fast_measured(
                ray_module,
                store,
                run_id=args.run_id,
                input_uris=input_uris,
                output_uri=uris["output"],
                config=config,
                actor_handles=actor_handles or (),
            )
        else:
            result["pipeline"] = run_legacy_measured(
                ray_module,
                store,
                arm=args.arm,
                run_id=args.run_id,
                input_uris=input_uris,
                output_uri=uris["output"],
                config=config,
            )
        result["cluster"]["actor_placement"] = collect_actor_placement(ray_module)
        if args.arm == "fast":
            actor_handles = list(result["pipeline"].pop("_actor_handles"))
            result["cluster"]["fast_actor_release"] = release_fast_actors(
                ray_module, actor_handles
            )
            actor_handles = None
        elapsed = float(result["pipeline"]["elapsed_s"])

        if args.skip_validation:
            result["validation"] = {"status": "skipped"}
            result["pipeline"]["sequences_per_s"] = None
            result["pipeline"]["performance_claims"] = (
                "suppressed because output validation was skipped"
            )
            for pipeline_split in result["pipeline"]["splits"].values():
                pipeline_split["sequences_per_s"] = None
        else:
            result["validation"] = validate_outputs(
                ray_module,
                store,
                arm=args.arm,
                input_uris=input_uris,
                output_uri=uris["output"],
            )
            for split in SPLITS:
                validated = result["validation"]["splits"][split]
                pipeline_split = result["pipeline"]["splits"][split]
                reported_rows = int(pipeline_split["raw_rows"])
                if reported_rows != int(validated["raw_rows"]):
                    raise RuntimeError(
                        f"{split}: pipeline reported {reported_rows:,} raw rows, "
                        f"validation found {validated['raw_rows']:,}"
                    )
                reported_sequences = pipeline_split.get("sequences")
                if reported_sequences is not None and int(reported_sequences) != int(
                    validated["sequences"]
                ):
                    raise RuntimeError(
                        f"{split}: pipeline reported {reported_sequences:,} sequences, "
                        f"validation found {validated['sequences']:,}"
                    )
                pipeline_split["sequences"] = int(validated["sequences"])
                split_elapsed = pipeline_split.get(
                    "elapsed_s",
                    pipeline_split.get(
                        "process_wall_s", pipeline_split.get("actor_critical_path_s")
                    ),
                )
                pipeline_split["sequences_per_s"] = (
                    int(validated["sequences"]) / float(split_elapsed)
                    if split_elapsed
                    else None
                )
            total_sequences = sum(
                int(split_result["sequences"])
                for split_result in result["validation"]["splits"].values()
            )
            result["pipeline"]["sequences_per_s"] = total_sequences / elapsed
        result["status"] = "succeeded"
        succeeded = True
    except BaseException as exc:
        failure = exc
        result["status"] = "failed"
        result["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        LOGGER.exception("Benchmark arm %s failed", args.arm)
    finally:
        # Cover failures after fast actor creation (including count or commit
        # failures) so cleanup and process shutdown never leave full-node
        # actors occupying validation resources.
        if actor_handles:
            for actor in actor_handles:
                try:
                    ray_module.kill(actor, no_restart=True)
                except Exception:
                    pass
            actor_handles = None
        cleanup_started = time.perf_counter()
        cleanup_errors: list[BaseException] = []
        try:
            result["cleanup"] = cleanup_benchmark_prefixes(
                store,
                warmup_uri=uris["warmup"],
                output_uri=uris["output"],
                # Never delete a pre-existing successful prefix merely because
                # this invocation refused to overwrite it.
                succeeded=(succeeded or not output_owned_by_attempt),
                retain_output=args.retain_output,
            )
        except Exception as cleanup_exc:
            cleanup_errors.append(cleanup_exc)
            result["cleanup"] = {
                "error": f"{type(cleanup_exc).__name__}: {cleanup_exc}",
            }
        try:
            result["cleanup"]["smoke_deleted_objects"] = store.delete_prefix(
                uris["smoke"]
            )
        except Exception as smoke_cleanup_exc:
            cleanup_errors.append(smoke_cleanup_exc)
            result["cleanup"]["smoke_error"] = (
                f"{type(smoke_cleanup_exc).__name__}: {smoke_cleanup_exc}"
            )
        if cleanup_errors and failure is None:
            cleanup_exc = cleanup_errors[0]
            failure = cleanup_exc
            result["status"] = "failed"
            result["failure"] = {
                "type": type(cleanup_exc).__name__,
                "message": str(cleanup_exc),
                "traceback": "Cleanup failed after the benchmark pipeline",
            }
        result["cleanup"]["elapsed_s"] = time.perf_counter() - cleanup_started
        result["finished_at"] = utc_now()
        validate_result_schema(result)
        atomic_write_json(result_path, result)
        generate_comparison_summary(args.results_dir)
    if failure is not None:
        raise failure
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one four-node AWS S3-to-S3 tokenization benchmark arm."
    )
    parser.add_argument("--arm", choices=("original", "tuned", "fast"))
    parser.add_argument("--input-uri", default=DEFAULT_INPUT_URI)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=validate_run_id)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--summarize-results",
        type=Path,
        metavar="DIR",
        help="Only combine arm JSON files under DIR into summary.json and summary.md.",
    )
    parser.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", "auto"))
    parser.add_argument(
        "--aws-region",
        default=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")),
    )
    parser.add_argument("--warmup-rows-per-actor", type=int, default=16_384)
    warmup_group = parser.add_mutually_exclusive_group()
    warmup_group.add_argument(
        "--warmup", dest="skip_warmup", action="store_false", default=False
    )
    warmup_group.add_argument("--skip-warmup", action="store_true")
    validation_group = parser.add_mutually_exclusive_group()
    validation_group.add_argument(
        "--validate", dest="skip_validation", action="store_false", default=False
    )
    validation_group.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--skip-smoke",
        "--skip-smoke-test",
        dest="skip_smoke_test",
        action="store_true",
        help="Skip the per-arm smoke test when cluster preflight already ran.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run cluster/S3/KvikIO checks and write smoke.json without tokenizing.",
    )
    parser.add_argument("--skip-cluster-check", action="store_true")
    parser.add_argument("--expected-nodes", type=int, default=4)
    parser.add_argument("--expected-cpus", type=float, default=64)
    parser.add_argument("--expected-gpus", type=float, default=4)
    parser.add_argument("--cluster-bootstrap-seconds", type=float, default=None)
    parser.add_argument(
        "--fast-row-groups-per-batch",
        type=int,
        default=16,
        help="Fast arm only: row groups in each projected cuDF S3 read.",
    )
    parser.add_argument(
        "--fast-write-threads-per-actor",
        type=int,
        default=4,
        help="Fast arm only: concurrent Arrow S3 shard writers per actor.",
    )
    parser.add_argument(
        "--fast-kvikio-remote-connections",
        type=int,
        default=8,
        help="Fast arm only: KvikIO I/O threads and actor-local S3 connections.",
    )
    parser.add_argument(
        "--fast-kvikio-task-size-bytes",
        type=int,
        default=4 << 20,
        help="Fast arm only: maximum bytes in each KvikIO I/O task.",
    )
    parser.add_argument(
        "--fast-output-shard-size-bytes",
        type=int,
        default=128 << 20,
        help="Fast arm only: target uncompressed bytes in each output shard.",
    )
    parser.add_argument(
        "--fast-overlap-split-writes",
        action="store_true",
        help=(
            "Fast arm only: overlap each actor's host write for one split with "
            "GPU preparation of the next split."
        ),
    )
    parser.add_argument("--overwrite-output", action="store_true")
    retention_group = parser.add_mutually_exclusive_group()
    retention_group.add_argument(
        "--retain-output", dest="retain_output", action="store_true", default=True
    )
    retention_group.add_argument(
        "--no-retain-output", dest="retain_output", action="store_false"
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.summarize_results is not None:
        args.summarize_results = args.summarize_results.expanduser().resolve()
        return args
    if args.arm is None:
        parser.error("--arm is required unless --summarize-results is used")
    if args.run_id is None:
        parser.error("--run-id is required unless --summarize-results is used")
    parse_s3_uri(args.input_uri, require_key=True)
    parse_s3_uri(args.output_root, require_key=True)
    validate_uri_isolation(
        args.input_uri, benchmark_uris(args.output_root, args.run_id, args.arm)
    )
    if args.warmup_rows_per_actor < 1:
        raise ValueError("--warmup-rows-per-actor must be positive")
    fast_positive_options = {
        "--fast-row-groups-per-batch": args.fast_row_groups_per_batch,
        "--fast-write-threads-per-actor": args.fast_write_threads_per_actor,
        "--fast-kvikio-remote-connections": args.fast_kvikio_remote_connections,
        "--fast-kvikio-task-size-bytes": args.fast_kvikio_task_size_bytes,
        "--fast-output-shard-size-bytes": args.fast_output_shard_size_bytes,
    }
    invalid_fast_options = [
        option for option, value in fast_positive_options.items() if value < 1
    ]
    if invalid_fast_options:
        parser.error(f"{', '.join(invalid_fast_options)} must be positive")
    if args.results_dir is None:
        args.results_dir = (
            Path(__file__).resolve().parent / "results" / "aws" / args.run_id
        )
    else:
        args.results_dir = args.results_dir.expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.summarize_results is not None:
        generate_comparison_summary(args.summarize_results, require_complete=True)
        print(args.summarize_results / "summary.json")
        print(args.summarize_results / "summary.md")
        return
    log_name = "smoke" if args.smoke_only else args.arm
    log_path = configure_logging(args.results_dir, log_name)
    LOGGER.info("Starting %s arm; log=%s", args.arm, log_path)
    try:
        if args.smoke_only:
            execute_smoke_only(args)
            LOGGER.info("Cluster/S3 smoke test passed")
            return
        result = execute_benchmark(args)
    except BaseException:
        raise SystemExit(1)
    LOGGER.info(
        "Completed %s arm in %.2fs", args.arm, result["pipeline"]["elapsed_s"]
    )


__all__ = [
    "EXPECTED_ROWS",
    "EXPECTED_SEQUENCES",
    "S3Store",
    "S3Uri",
    "arm_configuration",
    "assert_unique_output_keys",
    "benchmark_uris",
    "build_parser",
    "cleanup_benchmark_prefixes",
    "commit_split",
    "derive_override_num_blocks",
    "execute_benchmark",
    "execute_smoke_only",
    "generate_comparison_summary",
    "main",
    "new_result",
    "parse_args",
    "parse_s3_uri",
    "s3_join",
    "s3_prefixes_overlap",
    "split_input_uris",
    "validate_cluster_shape",
    "validate_fast_output_files",
    "validate_result_schema",
    "validate_uri_isolation",
]
