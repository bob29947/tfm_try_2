"""GPU-oriented parquet fragment execution helpers.

This module is intentionally small and experimental.  It covers a common GPU
ETL shape that is awkward to express efficiently with block-oriented APIs:

* The input is parquet with useful row-group statistics for a key column.
* Work can be split by key ranges without a shuffle.
* Each partition should run inside a long-lived GPU actor that reads parquet
  fragments, processes them, and writes its own output shard.

The helpers below are generic.  Ray handles key-range planning and GPU actor
scheduling; user code provides the domain-specific processor class.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import ray

PathLike = Union[str, Path]
InputPaths = Union[PathLike, Sequence[PathLike]]
NamedInputs = Union[InputPaths, Mapping[str, InputPaths]]
Partition = Dict[str, Any]


def _as_paths(paths: InputPaths) -> List[Path]:
    if isinstance(paths, (str, Path)):
        values = [paths]
    else:
        values = list(paths)

    out: List[Path] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            files = sorted(path.glob("*.parquet")) or sorted(path.rglob("*.parquet"))
            out.extend(files)
        else:
            out.append(path)
    return out


def _normalize_inputs(inputs: NamedInputs) -> Dict[str, List[Path]]:
    if isinstance(inputs, Mapping):
        return {str(name): _as_paths(paths) for name, paths in inputs.items()}
    return {"default": _as_paths(inputs)}


def _key_ranges(key_min: int, key_max: int, num_partitions: int) -> List[Tuple[int, int]]:
    total = key_max - key_min + 1
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
            parquet = pq.ParquetFile(path)
            try:
                key_index = parquet.schema.names.index(key_column)
            except ValueError as exc:
                raise ValueError(f"Column {key_column!r} not found in {path}.") from exc

            for row_group_idx in range(parquet.metadata.num_row_groups):
                row_group = parquet.metadata.row_group(row_group_idx)
                stats = row_group.column(key_index).statistics
                if stats is None or stats.min is None or stats.max is None:
                    raise ValueError(
                        f"Missing {key_column!r} statistics in {path} row group "
                        f"{row_group_idx}; key-range GPU planning requires parquet "
                        "row-group statistics."
                    )
                key_min = int(stats.min)
                key_max = int(stats.max)
                global_min = min(global_min, key_min)
                global_max = max(global_max, key_max)
                row_group_entries.append(
                    {
                        "path": str(path),
                        "row_group": row_group_idx,
                        "key_min": key_min,
                        "key_max": key_max,
                        "num_rows": row_group.num_rows,
                    }
                )
        row_groups_by_input[name] = row_group_entries

    if global_min == math.inf:
        raise ValueError("No parquet row groups found.")

    ranges = _key_ranges(int(global_min), int(global_max), num_partitions)
    planned: Dict[str, List[Partition]] = {}
    for name, row_groups in row_groups_by_input.items():
        partitions: List[Partition] = []
        for partition_id, (key_min, key_max) in enumerate(ranges):
            by_path: Dict[str, List[int]] = {}
            input_rows = 0
            for entry in row_groups:
                if entry["key_max"] >= key_min and entry["key_min"] <= key_max:
                    by_path.setdefault(entry["path"], []).append(entry["row_group"])
                    input_rows += int(entry["num_rows"])
            fragments = [
                {"path": path, "row_groups": row_group_ids}
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

    def ready(self) -> bool:
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
    if actor_list and len(actor_list) < len(partition_ids):
        raise ValueError(
            f"Expected at least {len(partition_ids)} actors, got {len(actor_list)}."
        )
    for idx, partition_id in enumerate(partition_ids):
        actor = (
            actor_list[idx]
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
