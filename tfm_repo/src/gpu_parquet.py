# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for the application-local Parquet actor runner."""

from .tokenization.parquet_runner import (
    Partition,
    _group_partitions_by_id,
    create_gpu_parquet_actors,
    map_gpu_parquet_partitions,
    plan_parquet_key_range_partitions,
)

__all__ = [
    "Partition",
    "create_gpu_parquet_actors",
    "map_gpu_parquet_partitions",
    "plan_parquet_key_range_partitions",
]
