#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create NB02-style tokenized train/val/test parquet from temporal splits."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--actors", type=int, default=8)
    parser.add_argument("--num-cpus-per-actor", type=int, default=8)
    parser.add_argument("--num-gpus-per-actor", type=float, default=0.5)
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
            "Use Ray's experimental GPU parquet key-range partition runner or "
            "the legacy Ray Data groupby path."
        ),
    )
    parser.add_argument(
        "--row-groups-per-batch",
        type=int,
        default=64,
        help="gpu-parquet engine only: parquet row groups each GPU actor reads per batch.",
    )
    parser.add_argument(
        "--output-dtype",
        choices=("uint16", "int32", "int64"),
        default="int32",
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
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else default_output_dir(split_dir)
    )
    return split_dir, output_dir


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
        fn_constructor_kwargs={"merchant_hash_size": C.MERCHANT_HASH_SIZE},
        batch_size=args.batch_size,
        compute=ray.data.ActorPoolStrategy(size=args.actors),
        num_gpus=args.num_gpus_per_actor,
        batch_format="cudf",
        runtime_env=C.GPU_RUNTIME_ENV,
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
            n = parquet_row_count(out)
            print(f"[{split:5s}] tokenized exists: {n:,} sequences at {out}")
            seq_counts[split] = n
            continue
        if out.exists() and args.overwrite:
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)
        seq_counts[split] = 0
        pending.append(split)
    return seq_counts, pending


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
    from ray.data.experimental.gpu_parquet import (
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

    data_started = time.time()
    seq_counts, pending = prepare_fast_outputs(output_dir, args)
    if not pending:
        return seq_counts, time.time() - data_started

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
        f"output {args.output_format}/{args.output_dtype}, "
        f"compression {args.compression}, "
        f"dictionary {'on' if args.use_dictionary else 'off'}, "
        f"deps from {runtime_env_label} "
        "(ray.data.experimental.gpu_parquet)"
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
            print(
                f"[{split:5s}] shard written: "
                f"{int(stat.get('rows', 0)):,} rows -> "
                f"{int(stat['count']):,} sequences in {elapsed_s:.2f}s "
                f"(read {read_s:.2f}s, tokenize {tokenize_s:.2f}s, "
                f"sort {sort_s:.2f}s, sequence {sequence_s:.2f}s, "
                f"write {write_s:.2f}s) "
                f"at {stat['output_path']}"
            )

    for split in pending:
        print(
            f"[{split:5s}] tokenized written: "
            f"{seq_counts[split]:,} sequences at {output_dir / split}"
        )
    return seq_counts, time.time() - data_started


def main() -> None:
    args = parse_args()
    split_dir, output_dir = validate_args(args)

    if str(TFM_ROOT) not in sys.path:
        sys.path.insert(0, str(TFM_ROOT))

    os.environ["TFM_SPLIT_DIR"] = str(split_dir)
    os.environ["TFM_TOKENIZED_DIR"] = str(output_dir)

    from src import ray_common as C

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

    print("\nTokenized outputs for NB03:")
    for split in SPLITS:
        print(f"  {split:5s}: {seq_counts[split]:>10,} sequences -> {output_dir / split}")
    print(f"\nPoint NB03 at this output with:\n  export TFM_TOKENIZED_DIR={output_dir}")
    print(f"Tokenization wall time: {elapsed_s:.2f} s")


if __name__ == "__main__":
    main()
