#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create NB02-style tokenized train/val/test parquet from temporal splits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path


TFM_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
FAST_OUTPUT_IN_PROGRESS = "_IN_PROGRESS"
FAST_OUTPUT_SUCCESS = "_SUCCESS"
TOKENIZATION_MANIFEST = "_tokenization_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize temporal splits for NB03.")
    parser.add_argument(
        "split_dir",
        type=Path,
        help="Directory containing train/val/test parquet dirs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Tokenized output directory. Defaults to sibling tokenized_v*.",
    )
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--actors", type=int, default=4)
    parser.add_argument("--num-cpus-per-actor", type=int, default=16)
    parser.add_argument("--num-gpus-per-actor", type=float, default=1.0)
    parser.add_argument(
        "--local-num-cpus",
        type=int,
        default=None,
        help="Local Ray only: advertise this many CPUs to Ray.",
    )
    parser.add_argument(
        "--local-num-gpus",
        type=int,
        default=None,
        help="Local Ray only: advertise this many GPUs to Ray.",
    )
    parser.add_argument(
        "--ray-temp-dir",
        type=Path,
        default=Path(os.environ.get("TFM_RAY_TEMP_DIR", "/dev/shm/tfm-ray")),
        help=(
            "Local Ray only: root temp/session directory. Defaults to "
            "TFM_RAY_TEMP_DIR or /dev/shm/tfm-ray to avoid slow/full /tmp."
        ),
    )
    parser.add_argument(
        "--ray-spill-dir",
        type=Path,
        default=(
            Path(os.environ["TFM_RAY_SPILL_DIR"])
            if os.environ.get("TFM_RAY_SPILL_DIR")
            else None
        ),
        help=(
            "Local Ray only: filesystem object-spill directory. Defaults to "
            "<ray-temp-dir>/spill."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=("gpu-parquet", "legacy"),
        default="gpu-parquet",
        help=(
            "Use the application-local Ray Core GPU parquet runner or "
            "the legacy Ray Data groupby path."
        ),
    )
    parser.add_argument(
        "--merchant-hash-mode",
        choices=("integer_mod", "string_hash"),
        default="integer_mod",
        help=(
            "Merchant bucket mapping for the legacy Ray Data engine. The fast "
            "gpu-parquet engine always uses integer_mod."
        ),
    )
    parser.add_argument(
        "--row-groups-per-batch",
        type=int,
        default=64,
        help="gpu-parquet engine only: parquet row groups each GPU actor reads per batch.",
    )
    parser.add_argument(
        "--write-threads-per-actor",
        type=int,
        default=12,
        help=(
            "gpu-parquet engine only: concurrent parquet shard writers per actor. "
            "Large outputs are striped across these writers."
        ),
    )
    parser.add_argument(
        "--output-shard-size-mb",
        type=int,
        default=128,
        help=(
            "gpu-parquet engine only: target maximum uncompressed bytes per "
            "striped output shard, in MiB."
        ),
    )
    parser.add_argument(
        "--output-dtype",
        choices=("uint16", "int32", "int64"),
        default="uint16",
        help="gpu-parquet engine only: dtype for fixed-width input_ids tensors.",
    )
    parser.add_argument(
        "--output-format",
        choices=("binary-tensor", "arrow-tensor"),
        default="binary-tensor",
        help=(
            "gpu-parquet engine only: write fast fixed-size-binary tensor "
            "parquet or Ray's standard Arrow tensor extension."
        ),
    )
    parser.add_argument(
        "--use-dictionary",
        action="store_true",
        help="gpu-parquet engine only: enable parquet dictionary encoding.",
    )
    parser.add_argument(
        "--compression",
        default="none",
        help="gpu-parquet engine only: parquet compression codec for tokenized output.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=1,
        help="gpu-parquet engine only: parquet compression level; use -1 for codec default.",
    )
    parser.add_argument(
        "--validate-order",
        action="store_true",
        help="gpu-parquet engine only: fail if input rows are not sorted by User/Card.",
    )
    parser.add_argument(
        "--gpu-runtime-env",
        choices=("auto", "pip", "none"),
        default="auto",
        help=(
            "gpu-parquet engine only: use Ray's pip runtime_env, skip it, or "
            "auto-skip when cudf/cupy are already installed in this environment."
        ),
    )
    parser.add_argument(
        "--no-prewarm-actors",
        action="store_true",
        help="gpu-parquet engine only: include GPU actor startup in the timed path.",
    )
    parser.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_ADDRESS", "auto"),
        help="Ray address. Use 'local' to start a local Ray runtime.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def version_label(split_dir: Path) -> str:
    match = re.search(r"\b(v\d+)\b", split_dir.name)
    if match:
        return match.group(1)
    return re.sub(r"[^A-Za-z0-9]+", "_", split_dir.name).strip("_") or "run"


def default_output_dir(split_dir: Path) -> Path:
    return split_dir.parent / f"tokenized_{version_label(split_dir)}"


def nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    split_dir = args.split_dir.expanduser().resolve()
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    missing = [split for split in SPLITS if not (split_dir / split).exists()]
    if missing:
        raise FileNotFoundError(f"Missing split directories under {split_dir}: {missing}")
    if args.actors < 1:
        raise ValueError("--actors must be at least 1")
    if args.num_cpus_per_actor < 1:
        raise ValueError("--num-cpus-per-actor must be at least 1")
    if args.row_groups_per_batch < 1:
        raise ValueError("--row-groups-per-batch must be at least 1")
    if args.engine == "gpu-parquet":
        if args.merchant_hash_mode != "integer_mod":
            raise ValueError(
                "The gpu-parquet engine implements integer_mod merchant mapping; "
                "use --engine legacy for string_hash."
            )
        if args.write_threads_per_actor < 1:
            raise ValueError("--write-threads-per-actor must be at least 1")
        if args.write_threads_per_actor > args.num_cpus_per_actor:
            raise ValueError(
                "--write-threads-per-actor cannot exceed --num-cpus-per-actor"
            )
        if args.output_shard_size_mb < 1:
            raise ValueError("--output-shard-size-mb must be at least 1")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else default_output_dir(split_dir)
    )
    if (
        output_dir == split_dir
        or output_dir in split_dir.parents
        or split_dir in output_dir.parents
    ):
        raise ValueError(
            f"Input and output directories must not overlap: {split_dir}, {output_dir}"
        )
    auxiliary_paths = []
    if args.ray_temp_dir is not None:
        auxiliary_paths.append(("Ray temp", args.ray_temp_dir.expanduser().resolve()))
    if args.ray_spill_dir is not None:
        auxiliary_paths.append(("Ray spill", args.ray_spill_dir.expanduser().resolve()))
    for label, auxiliary in auxiliary_paths:
        for protected in (split_dir, output_dir):
            if (
                auxiliary == protected
                or auxiliary in protected.parents
                or protected in auxiliary.parents
            ):
                raise ValueError(
                    f"{label} path overlaps input/output: {auxiliary}, {protected}"
                )
    return split_dir, output_dir


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tokenization_config(C, split_dir: Path, args: argparse.Namespace) -> dict:
    config = {
        "schema_version": 1,
        "source": str(split_dir),
        "engine": args.engine,
        "merchant_hash_mode": args.merchant_hash_mode,
        "merchant_hash_size": C.MERCHANT_HASH_SIZE,
        "sequence_length": C.SEQ_LENGTH,
        "sequence_chunk_size": C.SEQ_CHUNK_SIZE,
        "actors": args.actors,
        "cpus_per_actor": args.num_cpus_per_actor,
        "gpus_per_actor": args.num_gpus_per_actor,
        "actors_prewarmed": (
            args.engine == "gpu-parquet" and not args.no_prewarm_actors
        ),
    }
    if args.engine == "gpu-parquet":
        config["output"] = {
            "format": args.output_format,
            "dtype": args.output_dtype,
            "compression": args.compression,
            "compression_level": args.compression_level,
            "dictionary": args.use_dictionary,
            "shard_size_mib": args.output_shard_size_mb,
            "writer_threads_per_actor": args.write_threads_per_actor,
        }
        config["row_groups_per_batch"] = args.row_groups_per_batch
        config["validate_order"] = args.validate_order
    else:
        config["batch_size"] = args.batch_size
    return config


def source_file_manifest(source_root: Path) -> dict[str, list[dict]]:
    return {
        split: [
            {
                "path": str(file.relative_to(source_root)),
                "bytes": file.stat().st_size,
                "sha256": file_sha256(file),
            }
            for file in parquet_files(source_root / split)
        ]
        for split in SPLITS
    }


def code_manifest() -> dict[str, str]:
    code_files = [
        Path(__file__).resolve(),
        TFM_ROOT / "src" / "gpu_parquet.py",
        TFM_ROOT / "src" / "ray_common.py",
        TFM_ROOT / "src" / "ray_tokenize.py",
        *sorted((TFM_ROOT / "src" / "tokenizer").glob("*.py")),
    ]
    manifest = {
        str(path.relative_to(TFM_ROOT)): file_sha256(path) for path in code_files
    }
    return manifest


def runtime_manifest() -> dict[str, str]:
    import numpy
    import pyarrow
    import ray

    versions = {
        "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__,
        "ray": ray.__version__,
        "python": sys.version.split()[0],
    }
    try:
        import cudf

        versions["cudf"] = cudf.__version__
    except ImportError:
        versions["cudf"] = "unavailable"
    try:
        import cupy

        versions["cupy"] = cupy.__version__
        name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
        versions["gpu"] = name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        versions["cupy"] = versions.get("cupy", "unavailable")
        versions["gpu"] = "unavailable"
    return versions


def output_file_manifest(output_dir: Path) -> dict[str, list[dict]]:
    import pyarrow.parquet as pq

    return {
        split: [
            {
                "path": str(file.relative_to(output_dir)),
                "bytes": file.stat().st_size,
                "rows": pq.ParquetFile(file).metadata.num_rows,
                "sha256": file_sha256(file),
            }
            for file in parquet_files(output_dir / split)
        ]
        for split in SPLITS
    }


def validate_existing_manifest(
    output_dir: Path,
    expected_config: dict,
    overwrite: bool,
) -> None:
    if overwrite or not nonempty_dir(output_dir):
        return
    manifest_path = output_dir / TOKENIZATION_MANIFEST
    if not manifest_path.exists():
        raise RuntimeError(
            f"Existing output has no tokenization manifest: {output_dir}. "
            "Pass --overwrite to regenerate it safely."
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config") != expected_config:
        raise RuntimeError(
            "Existing tokenization settings do not match this command. "
            f"Recorded={manifest.get('config')!r}, requested={expected_config!r}. "
            "Use a new --output-dir or pass --overwrite."
        )
    current_sources = source_file_manifest(Path(expected_config["source"]))
    if manifest.get("source_files") != current_sources:
        raise RuntimeError(
            "Source parquet files changed since this tokenized output was written. "
            "Use a new --output-dir or pass --overwrite."
        )
    if manifest.get("code_sha256") != code_manifest():
        raise RuntimeError(
            "Tokenizer implementation changed since this output was written. "
            "Use a new --output-dir or pass --overwrite."
        )
    if manifest.get("runtime") != runtime_manifest():
        raise RuntimeError(
            "Runtime package/GPU versions changed since this output was written. "
            "Use a new --output-dir or pass --overwrite."
        )
    current_outputs = output_file_manifest(output_dir)
    if manifest.get("output_files") != current_outputs:
        raise RuntimeError(
            "Tokenized parquet files changed since the manifest was written. "
            "Use a new --output-dir or pass --overwrite."
        )
    current_counts = {
        split: sum(item["rows"] for item in files)
        for split, files in current_outputs.items()
    }
    if manifest.get("sequence_counts") != current_counts:
        raise RuntimeError(
            f"Tokenized row counts changed: recorded={manifest.get('sequence_counts')}, "
            f"current={current_counts}."
        )


def write_tokenization_manifest(
    output_dir: Path,
    config: dict,
    seq_counts: dict[str, int],
    elapsed_s: float,
) -> None:
    source_root = Path(config["source"])
    manifest = {
        "config": config,
        "sequence_counts": seq_counts,
        "source_files": source_file_manifest(source_root),
        "elapsed_seconds": elapsed_s,
        "output_files": output_file_manifest(output_dir),
        "code_sha256": code_manifest(),
        "runtime": runtime_manifest(),
    }
    (output_dir / TOKENIZATION_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def init_ray(C, args: argparse.Namespace):
    import ray

    if ray.is_initialized():
        return ray
    if args.ray_address == "local":
        ray_kwargs = {"runtime_env": C.JOB_RUNTIME_ENV, "include_dashboard": False}
        if args.local_num_cpus is not None:
            ray_kwargs["num_cpus"] = args.local_num_cpus
        if args.local_num_gpus is not None:
            ray_kwargs["num_gpus"] = args.local_num_gpus
        if args.ray_temp_dir is not None:
            temp_dir = args.ray_temp_dir.expanduser().resolve()
            spill_dir = (
                args.ray_spill_dir.expanduser().resolve()
                if args.ray_spill_dir is not None
                else temp_dir / "spill"
            )
            temp_dir.mkdir(parents=True, exist_ok=True)
            spill_dir.mkdir(parents=True, exist_ok=True)
            ray_kwargs["_temp_dir"] = str(temp_dir)
            ray_kwargs["_system_config"] = {
                "object_spilling_config": json.dumps(
                    {
                        "type": "filesystem",
                        "params": {"directory_path": str(spill_dir)},
                    }
                )
            }
        ray.init(**ray_kwargs)
    else:
        ray.init(address=args.ray_address, runtime_env=C.JOB_RUNTIME_ENV)
    return ray


def tokenize_split(
    ray,
    C,
    split_dir: Path,
    output_dir: Path,
    split: str,
    args: argparse.Namespace,
) -> int:
    from src.ray_tokenize import GPUTokenizer, build_sequences

    out = output_dir / split
    if nonempty_dir(out) and not args.overwrite:
        n = ray.data.read_parquet(str(out)).count()
        print(f"[{split:5s}] tokenized exists: {n:,} sequences at {out}")
        return n
    if out.exists() and args.overwrite:
        shutil.rmtree(out)

    ds = ray.data.read_parquet(str(split_dir / split))
    toks = ds.map_batches(
        GPUTokenizer,
        fn_constructor_kwargs={
            "merchant_hash_size": C.MERCHANT_HASH_SIZE,
            "merchant_hash_mode": args.merchant_hash_mode,
        },
        batch_size=args.batch_size,
        compute=ray.data.ActorPoolStrategy(size=args.actors),
        num_gpus=args.num_gpus_per_actor,
        batch_format="cudf",
        runtime_env=(None if args.gpu_runtime_env == "none" else C.GPU_RUNTIME_ENV),
    )
    seqs = toks.groupby("uc_key").map_groups(build_sequences, batch_format="numpy")
    seqs.write_parquet(str(out))
    n = ray.data.read_parquet(str(out)).count()
    print(f"[{split:5s}] tokenized written: {n:,} sequences at {out}")
    return n


def parquet_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.parquet"))
    if files:
        return files
    return sorted(path.rglob("*.parquet"))


def parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    total = 0
    for file in parquet_files(path):
        total += pq.ParquetFile(file).metadata.num_rows
    return total


def prepare_fast_outputs(
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, int], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_counts: dict[str, int] = {}
    pending = []
    for split in SPLITS:
        out = output_dir / split
        if nonempty_dir(out) and not args.overwrite:
            in_progress = out / FAST_OUTPUT_IN_PROGRESS
            success = out / FAST_OUTPUT_SUCCESS
            if in_progress.exists() and not success.exists():
                raise RuntimeError(
                    f"Incomplete tokenized output found at {out}; "
                    "pass --overwrite to replace it."
                )
            if not parquet_files(out):
                raise RuntimeError(
                    f"Tokenized output contains no parquet files: {out}"
                )
            n = parquet_row_count(out)
            print(f"[{split:5s}] tokenized exists: {n:,} sequences at {out}")
            seq_counts[split] = n
            continue
        if out.exists() and args.overwrite:
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / FAST_OUTPUT_IN_PROGRESS).touch()
        seq_counts[split] = 0
        pending.append(split)
    return seq_counts, pending


def mark_fast_outputs_complete(output_dir: Path, pending: list[str]) -> None:
    for split in pending:
        out = output_dir / split
        (out / FAST_OUTPUT_SUCCESS).touch()
        (out / FAST_OUTPUT_IN_PROGRESS).unlink(missing_ok=True)


def select_gpu_runtime_env(C, args: argparse.Namespace):
    if args.gpu_runtime_env == "pip":
        return C.TOKENIZE_GPU_RUNTIME_ENV, "pip runtime_env"
    if args.gpu_runtime_env == "none":
        return None, "local environment"

    has_local_deps = all(
        importlib.util.find_spec(module) is not None
        for module in ("cudf", "cupy", "pyarrow")
    )
    if has_local_deps:
        return None, "local environment"
    return C.TOKENIZE_GPU_RUNTIME_ENV, "pip runtime_env"


def tokenize_gpu_parquet(
    ray,
    C,
    split_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, int], float]:
    from src.gpu_parquet import (
        create_gpu_parquet_actors,
        map_gpu_parquet_partitions,
        plan_parquet_key_range_partitions,
    )
    from src.ray_tokenize import FastParquetSplitTokenizer

    runtime_env, runtime_env_label = select_gpu_runtime_env(C, args)
    compression = None if args.compression.lower() == "none" else args.compression
    compression_level = None if args.compression_level < 0 or compression is None else args.compression_level
    processor_kwargs = {
        "merchant_hash_size": C.MERCHANT_HASH_SIZE,
        "seq_length": C.SEQ_LENGTH,
        "chunk_size": C.SEQ_CHUNK_SIZE,
        "output_dtype": args.output_dtype,
        "output_format": args.output_format,
        "compression": compression,
        "compression_level": compression_level,
        "use_dictionary": args.use_dictionary,
        "row_groups_per_batch": args.row_groups_per_batch,
        "arrow_cpu_threads": args.num_cpus_per_actor,
        "write_threads": args.write_threads_per_actor,
        "output_shard_size_bytes": args.output_shard_size_mb * 1024 * 1024,
        "validate_order": args.validate_order,
    }
    actors = None
    if not args.no_prewarm_actors:
        print(f"Prewarming {args.actors} GPU parquet actors...")
        actors = create_gpu_parquet_actors(
            args.actors,
            FastParquetSplitTokenizer,
            processor_kwargs=processor_kwargs,
            num_gpus_per_actor=args.num_gpus_per_actor,
            num_cpus_per_actor=args.num_cpus_per_actor,
            runtime_env=runtime_env,
        )

    seq_counts, pending = prepare_fast_outputs(output_dir, args)
    if not pending:
        return seq_counts, 0.0

    data_started = time.time()
    partitions = plan_parquet_key_range_partitions(
        {split: split_dir / split for split in pending},
        key_column="User",
        num_partitions=args.actors,
    )
    nonempty_partitions = 0
    min_user = None
    max_user = None
    for split, split_partitions in partitions.items():
        for partition in split_partitions:
            min_user = (
                partition["key_min"]
                if min_user is None
                else min(min_user, partition["key_min"])
            )
            max_user = (
                partition["key_max"]
                if max_user is None
                else max(max_user, partition["key_max"])
            )
            if partition["fragments"]:
                nonempty_partitions += 1
            partition["split"] = split
            partition["user_min"] = partition["key_min"]
            partition["user_max"] = partition["key_max"]
            partition["output_path"] = str(
                output_dir / split / f"part-{partition['partition_id']:05d}.parquet"
            )
    print(
        "Ray GPU parquet tokenizer: "
        f"{args.actors} actors x {args.num_gpus_per_actor:g} GPU, "
        f"{args.num_cpus_per_actor} CPU/actor, users {min_user:,}-{max_user:,}, "
        f"{nonempty_partitions} split partitions, "
        f"{args.row_groups_per_batch} row groups/batch, "
        f"{args.write_threads_per_actor} write threads/actor, "
        f"{args.output_shard_size_mb} MiB output shards, "
        f"output {args.output_format}/{args.output_dtype}, "
        f"compression {args.compression}, "
        f"dictionary {'on' if args.use_dictionary else 'off'}, "
        f"deps from {runtime_env_label} "
        "(application-local Ray Core runner)"
    )

    actor_results = map_gpu_parquet_partitions(
        partitions,
        FastParquetSplitTokenizer,
        processor_kwargs=processor_kwargs,
        num_gpus_per_actor=args.num_gpus_per_actor,
        num_cpus_per_actor=args.num_cpus_per_actor,
        runtime_env=runtime_env,
        actors=actors,
    )
    mark_fast_outputs_complete(output_dir, pending)
    data_elapsed_s = time.time() - data_started

    for actor_stats in actor_results:
        for stat in actor_stats:
            split = stat["split"]
            seq_counts[split] += int(stat["count"])
            elapsed_s = float(stat.get("elapsed_s", 0.0))
            read_s = float(stat.get("read_s", 0.0))
            tokenize_s = float(stat.get("tokenize_s", 0.0))
            sort_s = float(stat.get("sort_s", 0.0))
            sequence_s = float(stat.get("sequence_s", 0.0))
            write_s = float(stat.get("write_s", 0.0))
            output_files = int(stat.get("output_files", 1))
            output_location = (
                f"{output_files} files under {output_dir / split}"
                if output_files > 1
                else stat["output_path"]
            )
            print(
                f"[{split:5s}] output written: "
                f"{int(stat.get('rows', 0)):,} rows -> "
                f"{int(stat['count']):,} sequences in {elapsed_s:.2f}s "
                f"(read {read_s:.2f}s, tokenize {tokenize_s:.2f}s, "
                f"sort {sort_s:.2f}s, sequence {sequence_s:.2f}s, "
                f"write {write_s:.2f}s) "
                f"at {output_location}"
            )

    for split in pending:
        print(
            f"[{split:5s}] tokenized written: "
            f"{seq_counts[split]:,} sequences at {output_dir / split}"
        )
    return seq_counts, data_elapsed_s


def main() -> None:
    process_started = time.time()
    args = parse_args()
    split_dir, output_dir = validate_args(args)

    if str(TFM_ROOT) not in sys.path:
        sys.path.insert(0, str(TFM_ROOT))

    os.environ["TFM_SPLIT_DIR"] = str(split_dir)
    os.environ["TFM_TOKENIZED_DIR"] = str(output_dir)

    from src import ray_common as C

    manifest_config = tokenization_config(C, split_dir, args)
    validate_existing_manifest(output_dir, manifest_config, args.overwrite)
    all_outputs_existed = all(
        nonempty_dir(output_dir / split) for split in SPLITS
    )
    ray = init_ray(C, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Split input:    {split_dir}")
    print(f"Tokenized out:  {output_dir}")
    print(f"Ray GPUs:       {ray.cluster_resources().get('GPU', 0):.0f}")
    print(f"Ray CPUs:       {ray.cluster_resources().get('CPU', 0):.0f}")
    print(f"SEQ_LENGTH={C.SEQ_LENGTH}, SEQ_CHUNK_SIZE={C.SEQ_CHUNK_SIZE}\n")

    if args.engine == "gpu-parquet":
        seq_counts, elapsed_s = tokenize_gpu_parquet(ray, C, split_dir, output_dir, args)
    else:
        started = time.time()
        seq_counts = {}
        for split in SPLITS:
            seq_counts[split] = tokenize_split(ray, C, split_dir, output_dir, split, args)
        elapsed_s = time.time() - started

    manifest_path = output_dir / TOKENIZATION_MANIFEST
    manifest_elapsed_s = 0.0
    if args.overwrite or not all_outputs_existed or not manifest_path.exists():
        manifest_started = time.time()
        write_tokenization_manifest(output_dir, manifest_config, seq_counts, elapsed_s)
        manifest_elapsed_s = time.time() - manifest_started

    print("\nTokenized outputs for NB03:")
    for split in SPLITS:
        print(f"  {split:5s}: {seq_counts[split]:>10,} sequences -> {output_dir / split}")
    print(f"\nPoint NB03 at this output with:\n  export TFM_TOKENIZED_DIR={output_dir}")
    if args.engine == "legacy":
        timing_label = "Legacy tokenization data-path time"
    elif args.no_prewarm_actors:
        timing_label = "Tokenization data-path time including actor startup"
    else:
        timing_label = "Prewarmed tokenization data-path time"
    print(f"{timing_label}: {elapsed_s:.2f} s")
    if manifest_elapsed_s:
        print(f"Post-timer provenance manifest time: {manifest_elapsed_s:.2f} s")
    print(f"End-to-end process time (including Ray startup): {time.time()-process_started:.2f} s")


if __name__ == "__main__":
    main()
