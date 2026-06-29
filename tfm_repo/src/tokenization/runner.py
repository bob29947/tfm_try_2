# SPDX-License-Identifier: Apache-2.0
"""Ray execution paths for the tokenization CLI."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

from .contract import SPLITS
from .outputs import (
    mark_fast_outputs_complete,
    nonempty_dir,
    prepare_fast_outputs,
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


def validate_cluster_resources(ray, args: argparse.Namespace) -> None:
    """Fail on fixed local runtimes before creating unschedulable actors.

    Remote Ray clusters can autoscale after actor demand is submitted, so their
    currently visible resources are not a safe capacity limit.
    """
    if args.engine != "gpu-parquet" or args.ray_address != "local":
        return

    resources = ray.cluster_resources()
    required_gpus = args.actors * args.num_gpus_per_actor
    required_cpus = args.actors * args.num_cpus_per_actor
    available_gpus = float(resources.get("GPU", 0.0))
    available_cpus = float(resources.get("CPU", 0.0))
    shortages = []
    if required_gpus > available_gpus + 1e-9:
        shortages.append(f"{required_gpus:g} GPUs requested, {available_gpus:g} available")
    if required_cpus > available_cpus + 1e-9:
        shortages.append(f"{required_cpus:g} CPUs requested, {available_cpus:g} available")
    if shortages:
        raise RuntimeError(
            "Insufficient Ray resources for all long-lived gpu-parquet actors: "
            + "; ".join(shortages)
            + ". Reduce --actors/resources per actor or use a matching benchmark profile."
        )


def tokenize_split(
    ray,
    C,
    split_dir: Path,
    output_dir: Path,
    split: str,
    args: argparse.Namespace,
) -> int:
    from .legacy import GPUTokenizer, build_sequences

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
    from .parquet_runner import (
        create_gpu_parquet_actors,
        map_gpu_parquet_partitions,
        plan_parquet_key_range_partitions,
    )
    from .fast_actor import FastParquetSplitTokenizer

    runtime_env, runtime_env_label = select_gpu_runtime_env(C, args)
    compression = None if args.compression.lower() == "none" else args.compression
    compression_level = (
        None
        if args.compression_level < 0 or compression is None
        else args.compression_level
    )
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
    if not args.overwrite and all(
        nonempty_dir(output_dir / split) for split in SPLITS
    ):
        seq_counts, _ = prepare_fast_outputs(output_dir, args)
        return seq_counts, 0.0

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


__all__ = [
    "init_ray",
    "select_gpu_runtime_env",
    "tokenize_gpu_parquet",
    "tokenize_split",
    "validate_cluster_resources",
]
