# SPDX-License-Identifier: Apache-2.0
"""Tokenized output discovery and transactional lifecycle helpers."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .contract import FAST_OUTPUT_IN_PROGRESS, FAST_OUTPUT_SUCCESS, SPLITS


def nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


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
    """Prepare split directories and mark newly scheduled outputs incomplete."""
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
    """Atomically advertise each successfully written split as complete."""
    for split in pending:
        out = output_dir / split
        (out / FAST_OUTPUT_SUCCESS).touch()
        (out / FAST_OUTPUT_IN_PROGRESS).unlink(missing_ok=True)


__all__ = [
    "mark_fast_outputs_complete",
    "nonempty_dir",
    "parquet_files",
    "parquet_row_count",
    "prepare_fast_outputs",
]
