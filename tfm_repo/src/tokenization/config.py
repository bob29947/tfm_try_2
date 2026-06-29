# SPDX-License-Identifier: Apache-2.0
"""Configuration normalization for tokenization runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def version_label(split_dir: Path) -> str:
    match = re.search(r"\b(v\d+)\b", split_dir.name)
    if match:
        return match.group(1)
    return re.sub(r"[^A-Za-z0-9]+", "_", split_dir.name).strip("_") or "run"


def default_output_dir(split_dir: Path) -> Path:
    return split_dir.parent / f"tokenized_{version_label(split_dir)}"


def tokenization_config(C, split_dir: Path, args: argparse.Namespace) -> dict:
    """Create the normalized config persisted in the artifact manifest.

    Keeping this separate from argument parsing gives benchmark profiles a
    single normalized representation regardless of how defaults are supplied.
    """
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


__all__ = ["default_output_dir", "tokenization_config", "version_label"]
