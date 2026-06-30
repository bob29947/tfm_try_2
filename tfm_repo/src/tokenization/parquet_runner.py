"""Application-local Ray Core runner for GPU-oriented Parquet fragments.

This module is intentionally small and experimental.  It covers a common GPU
ETL shape that is awkward to express efficiently with block-oriented APIs:

* The input is parquet with useful row-group statistics for a key column.
* Work can be split by key ranges without a shuffle.
* Each partition should run inside a long-lived GPU actor that reads parquet
  fragments, processes them, and writes its own output shard.

This deliberately lives with the TFM pipeline rather than in Ray Data. Inputs
may be shared POSIX paths or object-store URIs understood by ``pyarrow.fs``;
the partition key must have integral Parquet min/max statistics. Ranges have
equal key width rather than balanced row counts. A row group can overlap
multiple ranges, so every processor MUST filter its rows to the inclusive
``key_min``/``key_max`` bounds. The caller owns idempotent output naming,
incomplete-output cleanup, and commit.
"""

from __future__ import annotations

import math
import posixpath
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit

import ray

PathLike = Union[str, Path]
InputPaths = Union[PathLike, Sequence[PathLike]]
NamedInputs = Union[InputPaths, Mapping[str, InputPaths]]
Partition = Dict[str, Any]


def _is_uri(path: PathLike) -> bool:
    """Return whether *path* is a non-file URI rather than a local path."""
    return isinstance(path, str) and bool(urlsplit(path).scheme) and not path.startswith(
        "file:"
    )


def _as_paths(paths: InputPaths) -> List[str]:
    if isinstance(paths, (str, Path)):
        values = [paths]
    else:
        values = list(paths)

    out: List[str] = []
    for value in values:
        if _is_uri(value):
            # URI directory expansion is handled with the owning Arrow
            # filesystem later.  Treating it as pathlib would turn
            # ``s3://bucket/key`` into a bogus local ``s3:/bucket/key`` path.
            out.append(str(value).rstrip("/"))
            continue
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            files = sorted(path.glob("*.parquet")) or sorted(path.rglob("*.parquet"))
            out.extend(str(file) for file in files)
        else:
            out.append(str(path))
    return out


def _normalize_inputs(inputs: NamedInputs) -> Dict[str, List[str]]:
    if isinstance(inputs, Mapping):
        return {str(name): _as_paths(paths) for name, paths in inputs.items()}
    return {"default": _as_paths(inputs)}


def _arrow_input_files(path: str) -> List[Tuple[str, Any, str]]:
    """Resolve one path into ``(display_path, filesystem, fs_path)`` triples.

    Local paths deliberately return ``filesystem=None`` so the historical
    ``ParquetFile(path)`` path remains unchanged.  Remote directories are
    expanded exactly once on the driver and the original URI spelling is kept
    in planned fragments for the GPU actors.
    """
    if not _is_uri(path):
        return [(path, None, path)]

    import pyarrow.fs as pafs

    filesystem, fs_path = pafs.FileSystem.from_uri(path)
    info = filesystem.get_file_info(fs_path)
    if info.type == pafs.FileType.NotFound:
        raise ValueError(f"Parquet input does not exist: {path}")
    if info.type == pafs.FileType.File:
        return [(path, filesystem, fs_path)]
    if info.type != pafs.FileType.Directory:
        raise ValueError(f"Parquet input is neither a file nor directory: {path}")

    selector = pafs.FileSelector(fs_path, recursive=True)
    files = sorted(
        (
            child
            for child in filesystem.get_file_info(selector)
            if child.type == pafs.FileType.File and child.path.endswith(".parquet")
        ),
        key=lambda child: child.path,
    )
    out: List[Tuple[str, Any, str]] = []
    for child in files:
        relative = posixpath.relpath(child.path, fs_path)
        out.append((f"{path.rstrip('/')}/{relative}", filesystem, child.path))
    return out


def _key_ranges(key_min: int, key_max: int, num_partitions: int) -> List[Tuple[int, int]]:
    total = key_max - key_min + 1
    if num_partitions > total:
        raise ValueError(
            f"num_partitions ({num_partitions}) exceeds the integral key "
            f"cardinality ({total})"
        )
    return [
        (
            key_min + (idx * total) // num_partitions,
            key_min + ((idx + 1) * total) // num_partitions - 1,
        )
        for idx in range(num_partitions)
    ]


def plan_parquet_key_range_partitions(
    inputs: NamedInputs,
    *,
    key_column: str,
    num_partitions: int,
) -> Dict[str, List[Partition]]:
    """Plan parquet row groups into contiguous key-range partitions.

    Args:
        inputs: A parquet path, a sequence of parquet paths, or a mapping from
            logical input name to parquet path(s).  Directory inputs are expanded
            to parquet files recursively when needed.
        key_column: Column whose parquet row-group min/max statistics define
            the partition ranges.
        num_partitions: Number of key ranges to create.

    Returns:
        A mapping from input name to a list of partition dictionaries.  Each
        dictionary contains ``partition_id``, ``key_min``, ``key_max``, and
        ``fragments``.  Each fragment contains ``path`` and ``row_groups``.

    Raises:
        ValueError: If inputs are empty, ``num_partitions`` is invalid, or a row
            group is missing statistics for ``key_column``.
    """
    if num_partitions < 1:
        raise ValueError("num_partitions must be at least 1")

    import pyarrow.parquet as pq

    normalized = _normalize_inputs(inputs)
    row_groups_by_input: Dict[str, List[Dict[str, Any]]] = {}
    global_min = math.inf
    global_max = -math.inf

    for name, paths in normalized.items():
        if not paths:
            raise ValueError(f"No parquet files found for input {name!r}.")
        row_group_entries: List[Dict[str, Any]] = []
        for path in paths:
            resolved_files = _arrow_input_files(path)
            if not resolved_files:
                raise ValueError(f"No parquet files found for input path {path!r}.")
            for display_path, filesystem, fs_path in resolved_files:
                # Opening the Arrow random-access file makes footer reads
                # range-based for S3.  It avoids downloading the object and
                # avoids a second dataset-discovery pass.
                source = (
                    filesystem.open_input_file(fs_path)
                    if filesystem is not None
                    else None
                )
                try:
                    parquet = pq.ParquetFile(source if source is not None else fs_path)
                    try:
                        key_index = parquet.schema.names.index(key_column)
                    except ValueError as exc:
                        raise ValueError(
                            f"Column {key_column!r} not found in {display_path}."
                        ) from exc

                    for row_group_idx in range(parquet.metadata.num_row_groups):
                        row_group = parquet.metadata.row_group(row_group_idx)
                        stats = row_group.column(key_index).statistics
                        if stats is None or stats.min is None or stats.max is None:
                            raise ValueError(
                                f"Missing {key_column!r} statistics in {display_path} "
                                f"row group {row_group_idx}; key-range GPU planning "
                                "requires parquet row-group statistics."
                            )
                        key_min = int(stats.min)
                        key_max = int(stats.max)
                        global_min = min(global_min, key_min)
                        global_max = max(global_max, key_max)
                        row_group_entries.append(
                            {
                                "path": display_path,
                                "row_group": row_group_idx,
                                "key_min": key_min,
                                "key_max": key_max,
                                "num_rows": row_group.num_rows,
                            }
                        )
                finally:
                    if source is not None:
                        source.close()
        row_groups_by_input[name] = row_group_entries

    if global_min == math.inf:
        raise ValueError("No parquet row groups found.")

    ranges = _key_ranges(int(global_min), int(global_max), num_partitions)
    planned: Dict[str, List[Partition]] = {}
    for name, row_groups in row_groups_by_input.items():
        partitions: List[Partition] = []
        for partition_id, (key_min, key_max) in enumerate(ranges):
            by_path: Dict[str, List[int]] = {}
            rows_by_path: Dict[str, List[int]] = {}
            input_rows = 0
            for entry in row_groups:
                if entry["key_max"] >= key_min and entry["key_min"] <= key_max:
                    by_path.setdefault(entry["path"], []).append(entry["row_group"])
                    rows_by_path.setdefault(entry["path"], []).append(
                        int(entry["num_rows"])
                    )
                    input_rows += int(entry["num_rows"])
            fragments = [
                {
                    "path": path,
                    "row_groups": row_group_ids,
                    "row_group_num_rows": rows_by_path[path],
                }
                for path, row_group_ids in sorted(by_path.items())
            ]
            partitions.append(
                {
                    "input": name,
                    "partition_id": partition_id,
                    "key_min": key_min,
                    "key_max": key_max,
                    "fragments": fragments,
                    "estimated_input_rows": input_rows,
                }
            )
        planned[name] = partitions
    return planned


@ray.remote
class _GPUParquetPartitionActor:
    def __init__(
        self,
        processor_cls: type,
        processor_args: Optional[Sequence[Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        args = tuple(processor_args or ())
        kwargs = dict(processor_kwargs or {})
        self._processor = processor_cls(*args, **kwargs)

    def process(self, work_items: List[Partition]) -> Any:
        if hasattr(self._processor, "process_partitions"):
            return self._processor.process_partitions(work_items)
        return self._processor(work_items)

    def ready(self) -> Any:
        if hasattr(self._processor, "ready"):
            return self._processor.ready()
        return True


def _group_partitions_by_id(
    partitions: Mapping[str, Sequence[Partition]],
) -> Dict[int, List[Partition]]:
    grouped: Dict[int, List[Partition]] = {}
    for _, input_partitions in partitions.items():
        for partition in input_partitions:
            if not partition.get("fragments"):
                continue
            grouped.setdefault(int(partition["partition_id"]), []).append(dict(partition))
    return grouped


def map_gpu_parquet_partitions(
    partitions: Mapping[str, Sequence[Partition]],
    processor_cls: type,
    *,
    processor_args: Optional[Sequence[Any]] = None,
    processor_kwargs: Optional[Dict[str, Any]] = None,
    num_gpus_per_actor: float = 1.0,
    num_cpus_per_actor: float = 1.0,
    runtime_env: Optional[Dict[str, Any]] = None,
    actor_options: Optional[Dict[str, Any]] = None,
    actors: Optional[Sequence[Any]] = None,
) -> List[Any]:
    """Run planned parquet partitions on long-lived GPU actors.

    One actor is launched per non-empty ``partition_id``.  Each actor receives
    all logical inputs for that partition id, so applications with train/val/test
    splits can reuse the same key range and GPU allocation.

    The processor class is instantiated inside each actor.  It must either be
    callable with ``List[Partition]`` or expose ``process_partitions``.
    """
    grouped = _group_partitions_by_id(partitions)
    if not grouped:
        return []

    options = dict(actor_options or {})
    options.setdefault("num_gpus", num_gpus_per_actor)
    options.setdefault("num_cpus", num_cpus_per_actor)
    if runtime_env is not None:
        options.setdefault("runtime_env", runtime_env)

    refs = []
    actor_list = list(actors or [])
    partition_ids = sorted(grouped)
    if actor_list and max(partition_ids) >= len(actor_list):
        raise ValueError(
            "Actor handles are indexed by partition_id; expected a handle for "
            f"partition {max(partition_ids)}, got {len(actor_list)} handles."
        )
    for partition_id in partition_ids:
        actor = (
            actor_list[partition_id]
            if actor_list
            else _GPUParquetPartitionActor.options(**options).remote(
                processor_cls, processor_args, processor_kwargs
            )
        )
        refs.append(actor.process.remote(grouped[partition_id]))
    return ray.get(refs)


def create_gpu_parquet_actors(
    count: int,
    processor_cls: type,
    *,
    processor_args: Optional[Sequence[Any]] = None,
    processor_kwargs: Optional[Dict[str, Any]] = None,
    num_gpus_per_actor: float = 1.0,
    num_cpus_per_actor: float = 1.0,
    runtime_env: Optional[Dict[str, Any]] = None,
    actor_options: Optional[Dict[str, Any]] = None,
    wait_until_ready: bool = True,
) -> List[Any]:
    """Create long-lived GPU parquet actors, optionally waiting for init.

    Pre-creating actors lets applications measure data processing separately
    from CUDA/cuDF import and actor initialization latency.
    """
    if count < 1:
        raise ValueError("count must be at least 1")

    options = dict(actor_options or {})
    options.setdefault("num_gpus", num_gpus_per_actor)
    options.setdefault("num_cpus", num_cpus_per_actor)
    if runtime_env is not None:
        options.setdefault("runtime_env", runtime_env)

    actors = [
        _GPUParquetPartitionActor.options(**options).remote(
            processor_cls, processor_args, processor_kwargs
        )
        for _ in range(count)
    ]
    if wait_until_ready:
        ray.get([actor.ready.remote() for actor in actors])
    return actors


def _limit_partition_row_groups(
    partitions: Mapping[str, Sequence[Partition]],
    maximum: Optional[int],
) -> Dict[str, List[Partition]]:
    """Copy a plan, optionally retaining only its first row groups.

    This is used for a bounded first-touch warmup.  The cap applies per logical
    split and key-range partition, so every non-empty actor still performs a
    real S3 range read, GPU tokenize, sequence build, and S3 write.
    """
    if maximum is not None and maximum < 1:
        raise ValueError("max_row_groups_per_partition must be at least 1")

    limited: Dict[str, List[Partition]] = {}
    for name, input_partitions in partitions.items():
        copied_partitions: List[Partition] = []
        for original in input_partitions:
            partition = dict(original)
            fragments = []
            remaining = maximum
            estimated_rows = 0
            for original_fragment in original.get("fragments", []):
                fragment = dict(original_fragment)
                row_groups = list(fragment.get("row_groups", []))
                row_counts = list(fragment.get("row_group_num_rows", []))
                if remaining is not None:
                    row_groups = row_groups[:remaining]
                    row_counts = row_counts[: len(row_groups)]
                    remaining -= len(row_groups)
                if not row_groups:
                    if remaining == 0:
                        break
                    continue
                fragment["row_groups"] = row_groups
                if row_counts:
                    fragment["row_group_num_rows"] = row_counts
                    estimated_rows += sum(int(value) for value in row_counts)
                fragments.append(fragment)
                if remaining == 0:
                    break
            partition["fragments"] = fragments
            if maximum is not None:
                partition["estimated_input_rows"] = estimated_rows
            copied_partitions.append(partition)
        limited[name] = copied_partitions
    return limited


def run_s3_tokenization(
    ray_module: Any,
    input_uris: Mapping[str, InputPaths],
    output_uri: str,
    *,
    actors: int = 4,
    cpus_per_actor: float = 16,
    gpus_per_actor: float = 1.0,
    row_groups_per_batch: int = 16,
    write_threads: int = 4,
    output_shard_size_bytes: int = 128 * 1024 * 1024,
    actor_handles: Optional[Sequence[Any]] = None,
    splits: Optional[Sequence[str]] = None,
    processor_kwargs: Optional[Mapping[str, Any]] = None,
    max_row_groups_per_partition: Optional[int] = None,
    key_column: str = "User",
    aws_region: Optional[str] = None,
    s3_connections: int = 8,
    kvikio_task_size_bytes: int = 4 * 1024 * 1024,
    overlap_split_writes: bool = False,
) -> Dict[str, Any]:
    """Run the fast key-range tokenizer directly from S3 to S3.

    The returned ``actors`` can be passed back as ``actor_handles`` so a
    bounded warmup (set ``max_row_groups_per_partition=1``) and the measured
    run share already initialized CUDA, KvikIO, filesystem, and tokenization
    state.  This function intentionally does not create ``_SUCCESS.json``;
    callers should validate ``output_files`` before committing a split.

    ``overlap_split_writes`` is opt-in. When enabled for multiple splits, each
    actor writes split N on host threads while preparing split N+1 on its GPU.
    The actor drains every upload before returning, so callers may safely
    validate and commit only after this function succeeds.
    """
    if actors < 1:
        raise ValueError("actors must be at least 1")
    if row_groups_per_batch < 1:
        raise ValueError("row_groups_per_batch must be at least 1")
    if write_threads < 1:
        raise ValueError("write_threads must be at least 1")
    if s3_connections < 1:
        raise ValueError("s3_connections must be at least 1")
    if kvikio_task_size_bytes < 1:
        raise ValueError("kvikio_task_size_bytes must be at least 1")
    if not output_uri.startswith("s3://"):
        raise ValueError("run_s3_tokenization requires an s3:// output URI")
    if not input_uris:
        raise ValueError("input_uris must not be empty")

    selected = list(splits) if splits is not None else list(input_uris)
    missing = [split for split in selected if split not in input_uris]
    if missing:
        raise ValueError(f"Unknown requested splits: {missing}")
    selected_inputs = {split: input_uris[split] for split in selected}
    for split, paths in selected_inputs.items():
        for path in _as_paths(paths):
            if not path.startswith("s3://"):
                raise ValueError(
                    f"run_s3_tokenization requires s3:// inputs; {split!r} has {path!r}"
                )

    from .fast_actor import FastParquetSplitTokenizer

    total_started = time.perf_counter()
    plan_started = time.perf_counter()
    partitions = plan_parquet_key_range_partitions(
        selected_inputs,
        key_column=key_column,
        num_partitions=actors,
    )
    partitions = _limit_partition_row_groups(
        partitions, max_row_groups_per_partition
    )
    output_prefix = output_uri.rstrip("/")
    for split, split_partitions in partitions.items():
        for partition in split_partitions:
            partition["split"] = split
            partition["user_min"] = partition["key_min"]
            partition["user_max"] = partition["key_max"]
            partition["output_path"] = (
                f"{output_prefix}/{split}/"
                f"part-{int(partition['partition_id']):05d}.parquet"
            )
    plan_s = time.perf_counter() - plan_started

    kwargs: Dict[str, Any] = {
        "output_dtype": "uint16",
        "output_format": "binary-tensor",
        "row_groups_per_batch": row_groups_per_batch,
        "arrow_cpu_threads": int(cpus_per_actor),
        "write_threads": write_threads,
        "output_shard_size_bytes": output_shard_size_bytes,
        "s3_mode": True,
        "aws_region": aws_region,
        "s3_connections": s3_connections,
        "kvikio_task_size_bytes": kvikio_task_size_bytes,
        "overlap_split_writes": bool(overlap_split_writes),
        "require_kvikio": True,
    }
    kwargs.update(dict(processor_kwargs or {}))
    # An S3 benchmark must never accidentally turn the fail-closed path off via
    # a generic processor configuration dictionary.
    kwargs["s3_mode"] = True
    kwargs["require_kvikio"] = True
    kwargs["overlap_split_writes"] = bool(overlap_split_writes)

    actor_list = list(actor_handles or [])
    actors_reused = bool(actor_list)
    if actor_list and len(actor_list) != actors:
        raise ValueError(
            f"Expected exactly {actors} reusable actors, got {len(actor_list)}"
        )
    actor_create_and_ready_started = time.perf_counter()
    if not actor_list:
        actor_list = create_gpu_parquet_actors(
            actors,
            FastParquetSplitTokenizer,
            processor_kwargs=kwargs,
            num_gpus_per_actor=gpus_per_actor,
            num_cpus_per_actor=cpus_per_actor,
            # The readiness barrier immediately below is part of the same
            # measured actor-setup interval. Avoid querying each fresh actor
            # twice while still waiting for every constructor to finish.
            wait_until_ready=False,
        )

    # Querying readiness for reused actors also verifies that the actor-local
    # KvikIO and S3 clients survived the warmup.
    backend_info = ray_module.get([actor.ready.remote() for actor in actor_list])
    actor_create_and_ready_s = time.perf_counter() - actor_create_and_ready_started
    expected_kvikio = {
        "kvikio_num_threads": int(s3_connections),
        "kvikio_task_size_bytes": int(kvikio_task_size_bytes),
    }
    mismatched_kvikio = [
        {
            "hostname": info.get("hostname"),
            "realized": {
                key: info.get(key)
                for key in expected_kvikio
            },
        }
        for info in backend_info
        if any(info.get(key) != value for key, value in expected_kvikio.items())
    ]
    if mismatched_kvikio:
        raise RuntimeError(
            "Fast S3 actors did not realize the requested KvikIO settings: "
            f"requested={expected_kvikio!r}, actors={mismatched_kvikio!r}"
        )
    mismatched_overlap = [
        {
            "hostname": info.get("hostname"),
            "realized": info.get("overlap_split_writes"),
        }
        for info in backend_info
        if info.get("overlap_split_writes") != bool(overlap_split_writes)
    ]
    if mismatched_overlap:
        raise RuntimeError(
            "Fast S3 actors did not realize the requested split-write overlap: "
            f"requested={bool(overlap_split_writes)!r}, "
            f"actors={mismatched_overlap!r}"
        )
    process_started = time.perf_counter()
    actor_results: List[Any] = []
    split_wall_s: Dict[str, float] = {}
    if overlap_split_writes and len(selected) > 1:
        # One ordered call per actor is required for host writes from split N
        # to remain active while that same actor prepares split N+1 on its GPU.
        # The actor itself retains at most one pending write batch.
        actor_results = map_gpu_parquet_partitions(
            {split: partitions[split] for split in selected},
            FastParquetSplitTokenizer,
            processor_kwargs=kwargs,
            num_gpus_per_actor=gpus_per_actor,
            num_cpus_per_actor=cpus_per_actor,
            actors=actor_list,
        )
    else:
        # Barrier between splits: all four actors release train frames before
        # any actor begins val, then do the same before test. This is retained
        # as the deployment-safe default and yields disjoint split wall times.
        for split in selected:
            split_started = time.perf_counter()
            actor_results.extend(
                map_gpu_parquet_partitions(
                    {split: partitions[split]},
                    FastParquetSplitTokenizer,
                    processor_kwargs=kwargs,
                    num_gpus_per_actor=gpus_per_actor,
                    num_cpus_per_actor=cpus_per_actor,
                    actors=actor_list,
                )
            )
            split_wall_s[split] = time.perf_counter() - split_started
    process_s = time.perf_counter() - process_started

    flat_stats = [stat for result in actor_results for stat in result]
    sequence_counts = {split: 0 for split in selected}
    raw_rows = {split: 0 for split in selected}
    output_files: Dict[str, List[str]] = {split: [] for split in selected}
    component_timings = {
        name: 0.0
        for name in (
            "read_s",
            "tokenize_s",
            "sort_s",
            "sequence_s",
            "compute_s",
            "write_s",
            "write_wait_s",
            "write_overlap_s",
        )
    }
    for stat in flat_stats:
        split = stat["split"]
        sequence_counts[split] += int(stat.get("count", 0))
        raw_rows[split] += int(stat.get("rows", 0))
        output_files[split].extend(stat.get("output_paths", []))
        for name in component_timings:
            component_timings[name] += float(stat.get(name, 0.0))
    split_compute_wall_s = {
        split: max(
            (
                float(stat.get("compute_s", 0.0))
                for stat in flat_stats
                if stat.get("split") == split
            ),
            default=0.0,
        )
        for split in selected
    }
    if overlap_split_writes and len(selected) > 1:
        # These intervals can overlap by design and therefore do not sum to the
        # process wall time. They remain useful as each split's actor critical
        # path from preparation start through its durable write completion.
        split_wall_s = {
            split: max(
                (
                    float(stat.get("elapsed_s", 0.0))
                    for stat in flat_stats
                    if stat.get("split") == split
                ),
                default=0.0,
            )
            for split in selected
        }
    for split, paths in output_files.items():
        if len(paths) != len(set(paths)):
            raise RuntimeError(f"Fast S3 writers produced duplicate keys for {split}")

    total_s = time.perf_counter() - total_started
    return {
        "sequence_counts": sequence_counts,
        "raw_rows": raw_rows,
        "stage_timings": {
            "plan_s": plan_s,
            "actor_create_and_ready_s": actor_create_and_ready_s,
            "process_s": process_s,
            "split_wall_s": split_wall_s,
            "split_compute_wall_s": split_compute_wall_s,
            "total_s": total_s,
            **component_timings,
        },
        "actor_stats": actor_results,
        "output_files": output_files,
        "actors": actor_list,
        "backend_info": backend_info,
        "writes_success_markers": False,
        "config": {
            "actors": actors,
            "cpus_per_actor": cpus_per_actor,
            "gpus_per_actor": gpus_per_actor,
            "row_groups_per_batch": row_groups_per_batch,
            "write_threads": write_threads,
            "output_shard_size_bytes": output_shard_size_bytes,
            "max_row_groups_per_partition": max_row_groups_per_partition,
            "s3_connections": s3_connections,
            "kvikio_task_size_bytes": kvikio_task_size_bytes,
            "overlap_split_writes": bool(overlap_split_writes),
            "actors_reused": actors_reused,
            "fresh_actors_created": not actors_reused,
            "key_column": key_column,
        },
    }
