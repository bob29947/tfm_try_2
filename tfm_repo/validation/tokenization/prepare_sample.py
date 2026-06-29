#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a fixed, auditable transaction sample for NB04/NB05 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SPLITS = ("train", "val", "test")
LABEL_COLUMN = "Is Fraud?"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-fraud", type=int, default=6_250)
    parser.add_argument("--train-normal", type=int, default=243_750)
    parser.add_argument("--eval-normal", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def parquet_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.parquet")) or sorted(path.rglob("*.parquet"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def class_counts(path: Path, batch_size: int) -> tuple[int, int]:
    normal = fraud = 0
    for file in parquet_files(path):
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(
            columns=[LABEL_COLUMN], batch_size=batch_size, use_threads=True
        ):
            is_fraud = pc.equal(batch.column(0), "Yes")
            batch_fraud = int(pc.sum(is_fraud).as_py() or 0)
            fraud += batch_fraud
            normal += len(batch) - batch_fraud
    return normal, fraud


def selected_ordinals(total: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if count < 0 or count > total:
        raise ValueError(f"Cannot select {count:,} rows from class with {total:,} rows")
    if count == total:
        return np.arange(total, dtype=np.int64)
    return np.sort(rng.choice(total, size=count, replace=False).astype(np.int64))


def local_positions(
    selected: np.ndarray,
    class_offset: int,
    class_mask: np.ndarray,
) -> np.ndarray:
    class_count = int(class_mask.sum())
    lo = np.searchsorted(selected, class_offset, side="left")
    hi = np.searchsorted(selected, class_offset + class_count, side="left")
    if lo == hi:
        return np.empty(0, dtype=np.int64)
    class_rows = np.flatnonzero(class_mask)
    return class_rows[selected[lo:hi] - class_offset]


def write_sample_split(
    source: Path,
    destination: Path,
    selected_normal: np.ndarray,
    selected_fraud: np.ndarray,
    total_normal: int,
    total_fraud: int,
    batch_size: int,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "part-00000.parquet"
    writer = None
    normal_offset = fraud_offset = row_offset = written = 0
    row_ids_hash = hashlib.sha256()
    normal_weight = total_normal / max(len(selected_normal), 1)
    fraud_weight = total_fraud / max(len(selected_fraud), 1)

    try:
        for file in parquet_files(source):
            parquet = pq.ParquetFile(file)
            for batch in parquet.iter_batches(batch_size=batch_size, use_threads=True):
                labels = np.asarray(batch.column(batch.schema.get_field_index(LABEL_COLUMN)))
                fraud_mask = labels == "Yes"
                normal_mask = ~fraud_mask
                normal_rows = local_positions(
                    selected_normal, normal_offset, normal_mask
                )
                fraud_rows = local_positions(
                    selected_fraud, fraud_offset, fraud_mask
                )
                chosen = np.sort(np.concatenate([normal_rows, fraud_rows]))
                if len(chosen):
                    table = pa.Table.from_batches([batch]).take(pa.array(chosen))
                    row_ids = row_offset + chosen
                    chosen_fraud = fraud_mask[chosen]
                    weights = np.where(
                        chosen_fraud, fraud_weight, normal_weight
                    ).astype(
                        np.float64
                    )
                    table = table.append_column("row_id", pa.array(row_ids))
                    table = table.append_column("sample_weight", pa.array(weights))
                    if writer is None:
                        writer = pq.ParquetWriter(
                            output,
                            table.schema,
                            compression="zstd",
                            use_dictionary=True,
                        )
                    writer.write_table(table)
                    row_ids_hash.update(
                        np.asarray(row_ids, dtype="<i8", order="C").tobytes()
                    )
                    written += len(chosen)
                normal_offset += int(normal_mask.sum())
                fraud_offset += int(fraud_mask.sum())
                row_offset += len(batch)
    finally:
        if writer is not None:
            writer.close()

    expected = len(selected_normal) + len(selected_fraud)
    if written != expected:
        raise RuntimeError(f"Wrote {written:,} rows, expected {expected:,}")
    return {
        "rows": written,
        "normal": len(selected_normal),
        "fraud": len(selected_fraud),
        "normal_sample_weight": normal_weight,
        "fraud_sample_weight": fraud_weight,
        "row_ids_sha256": row_ids_hash.hexdigest(),
        "output": str(output),
        "output_sha256": file_sha256(output),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    split_dir = args.split_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if (
        output_dir == split_dir
        or output_dir in split_dir.parents
        or split_dir in output_dir.parents
    ):
        raise ValueError(
            f"Input and output directories must not overlap: {split_dir}, {output_dir}"
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    for split in SPLITS:
        if not parquet_files(split_dir / split):
            raise FileNotFoundError(f"No parquet files under {split_dir / split}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    counts = {
        split: class_counts(split_dir / split, args.batch_size) for split in SPLITS
    }
    results = {}
    for split_index, split in enumerate(SPLITS):
        normal, fraud = counts[split]
        rng = np.random.default_rng(args.seed + split_index)
        fraud_count = min(args.train_fraud, fraud) if split == "train" else fraud
        normal_count = args.train_normal if split == "train" else args.eval_normal
        selected_normal = selected_ordinals(normal, normal_count, rng)
        selected_fraud = selected_ordinals(fraud, fraud_count, rng)
        results[split] = write_sample_split(
            split_dir / split,
            output_dir / split,
            selected_normal,
            selected_fraud,
            normal,
            fraud,
            args.batch_size,
        )
        print(
            f"[{split:5s}] {results[split]['rows']:,} rows "
            f"({results[split]['fraud']:,} fraud) -> {output_dir / split}",
            flush=True,
        )

    metadata = {
        "source": str(split_dir),
        "seed": args.seed,
        "source_counts": {
            split: {"normal": normal, "fraud": fraud}
            for split, (normal, fraud) in counts.items()
        },
        "samples": results,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
