#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create NB01-style temporal train/val/test splits from raw parquet.

This is the first half of the notebook preprocessing pipeline:

    raw parquet -> temporal_split_v*/{train,val,test}

The split logic mirrors NB01: build a YYYY-MM-DD date column, choose date
cutoffs by cumulative row count at 80% and 90%, then write the original
TabFormer columns back out as parquet. The writer normalizes raw parquet fields
that commonly arrive as Arrow-native types back to the notebook-compatible
shape: ``Time`` as ``HH:MM`` and ``Is Fraud?`` as ``Yes``/``No``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


TFM_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create temporal train/val/test parquet splits.")
    parser.add_argument("raw_parquet", type=Path, help="Raw TabFormer parquet file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output split directory. Defaults to tfm_repo/data/temporal_split_v*.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--test-ratio", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def version_label(raw_parquet: Path) -> str:
    match = re.search(r"\b(v\d+)\b", raw_parquet.stem)
    if match:
        return match.group(1)
    return re.sub(r"[^A-Za-z0-9]+", "_", raw_parquet.stem).strip("_") or "run"


def default_data_root(raw_parquet: Path) -> Path:
    raw = raw_parquet.resolve()
    if len(raw.parts) >= 3 and raw.parts[-3:-1] == ("raw", "parquet"):
        return raw.parents[2]
    return TFM_ROOT / "data"


def default_output_dir(raw_parquet: Path) -> Path:
    return default_data_root(raw_parquet) / f"temporal_split_{version_label(raw_parquet)}"


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    raw = args.raw_parquet.expanduser().resolve()
    if not raw.exists():
        raise FileNotFoundError(f"Raw parquet file does not exist: {raw}")
    if not 0 < args.train_ratio < args.test_ratio < 1:
        raise ValueError("--train-ratio and --test-ratio must satisfy 0 < train < test < 1")
    out = args.output_dir.expanduser().resolve() if args.output_dir else default_output_dir(raw)
    return raw, out


def split_dirs_ready(output_dir: Path) -> bool:
    return all((output_dir / split).is_dir() and any((output_dir / split).iterdir()) for split in SPLITS)


def date_array(table: pa.Table) -> pa.Array:
    year = pc.cast(table["Year"], pa.string())
    month = pc.utf8_lpad(pc.cast(table["Month"], pa.string()), 2, "0")
    day = pc.utf8_lpad(pc.cast(table["Day"], pa.string()), 2, "0")
    return pc.binary_join_element_wise(year, month, day, "-")


def fraud_bool_array(table: pa.Table) -> pa.Array:
    col = table["Is Fraud?"]
    if pa.types.is_boolean(col.type):
        return col
    as_str = pc.utf8_upper(pc.cast(col, pa.string()))
    return pc.or_(pc.equal(as_str, "YES"), pc.equal(as_str, "1"))


def scalar_count(mask: pa.Array) -> int:
    return int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)


def compute_daily_counts(raw: Path, batch_size: int) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    total = 0
    parquet = pq.ParquetFile(raw)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["Year", "Month", "Day"]):
        table = pa.Table.from_batches([batch])
        dates = date_array(table)
        value_counts = pc.value_counts(dates)
        for value, count in zip(value_counts.field("values").to_pylist(), value_counts.field("counts").to_pylist()):
            counts[value] += int(count)
            total += int(count)
    return counts, total


def cutoffs_from_counts(counts: Counter[str], total: int, train_ratio: float, test_ratio: float) -> tuple[str, str]:
    cumulative = 0
    train_cut = None
    test_cut = None
    for date, count in sorted(counts.items()):
        cumulative += count
        if train_cut is None and cumulative >= total * train_ratio:
            train_cut = date
        if test_cut is None and cumulative >= total * test_ratio:
            test_cut = date
            break
    if train_cut is None or test_cut is None:
        raise RuntimeError("Could not compute temporal cutoffs from daily counts")
    return train_cut, test_cut


def normalize_output_table(table: pa.Table) -> pa.Table:
    arrays = []
    for field in table.schema:
        col = table[field.name]
        if field.name == "Time" and (
            pa.types.is_time(col.type) or pa.types.is_timestamp(col.type) or pa.types.is_date(col.type)
        ):
            col = pc.strftime(col, format="%H:%M")
        elif field.name == "Is Fraud?" and pa.types.is_boolean(col.type):
            col = pc.if_else(col, pa.scalar("Yes"), pa.scalar("No"))
        arrays.append(col)
    return pa.Table.from_arrays(arrays, names=table.schema.names)


def write_table(writer_map, output_dir: Path, split: str, table: pa.Table, compression: str) -> int:
    if table.num_rows == 0:
        return 0
    table = normalize_output_table(table)
    if writer_map[split] is None:
        path = output_dir / split / "part-00000.parquet"
        writer_map[split] = pq.ParquetWriter(path, table.schema, compression=compression)
    writer_map[split].write_table(table)
    return table.num_rows


def write_splits(
    raw: Path,
    output_dir: Path,
    train_cut: str,
    test_cut: str,
    batch_size: int,
    compression: str,
) -> dict[str, dict[str, int]]:
    parquet = pq.ParquetFile(raw)
    writers = {split: None for split in SPLITS}
    stats = {split: {"rows": 0, "fraud": 0} for split in SPLITS}

    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            dates = date_array(table)
            fraud = fraud_bool_array(table)

            masks = {
                "train": pc.less(dates, train_cut),
                "val": pc.and_(pc.greater_equal(dates, train_cut), pc.less(dates, test_cut)),
                "test": pc.greater_equal(dates, test_cut),
            }
            for split, mask in masks.items():
                split_table = table.filter(mask)
                rows = write_table(writers, output_dir, split, split_table, compression)
                stats[split]["rows"] += rows
                stats[split]["fraud"] += scalar_count(pc.and_(mask, fraud))
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()

    return stats


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if split_dirs_ready(output_dir):
        if not overwrite:
            raise FileExistsError(f"Split output already exists; pass --overwrite to replace: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output_dir / split).mkdir(parents=True, exist_ok=True)


def write_metadata(
    output_dir: Path,
    raw: Path,
    total: int,
    train_cut: str,
    test_cut: str,
    stats: dict[str, dict[str, int]],
    elapsed_s: float,
) -> None:
    payload = {
        "raw_parquet": str(raw),
        "rows": total,
        "train_cut": train_cut,
        "test_cut": test_cut,
        "splits": stats,
        "elapsed_s": elapsed_s,
    }
    (output_dir / "_split_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    raw, output_dir = validate_args(args)

    started = time.time()
    print(f"Raw parquet:  {raw}")
    print(f"Output dir:   {output_dir}")
    print("Computing temporal cutoffs...")
    daily_counts, total = compute_daily_counts(raw, args.batch_size)
    train_cut, test_cut = cutoffs_from_counts(daily_counts, total, args.train_ratio, args.test_ratio)
    print(f"Rows: {total:,}")
    print(f"Train/Val cutoff: {train_cut}   Val/Test cutoff: {test_cut}")

    prepare_output_dir(output_dir, args.overwrite)
    print("Writing splits...")
    stats = write_splits(raw, output_dir, train_cut, test_cut, args.batch_size, args.compression)
    elapsed_s = time.time() - started
    write_metadata(output_dir, raw, total, train_cut, test_cut, stats, elapsed_s)

    for split in SPLITS:
        rows = stats[split]["rows"]
        fraud = stats[split]["fraud"]
        print(f"[{split:5s}] rows={rows:,} fraud={fraud:,} rate={fraud / max(rows, 1):.4%}")
    print(f"Done in {elapsed_s / 60:.1f} min")


if __name__ == "__main__":
    main()
