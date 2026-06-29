#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tokenize one fixed transaction sample under paired merchant mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


TFM_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")
MODES = ("string_hash", "integer_mod")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--actors", type=int, default=4)
    parser.add_argument("--num-cpus", type=int, default=64)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument(
        "--ray-temp-dir", type=Path, default=Path("/dev/shm/tfm-downstream-tokenize")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def count_parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    files = sorted(path.glob("*.parquet")) or sorted(path.rglob("*.parquet"))
    return sum(pq.ParquetFile(file).metadata.num_rows for file in files)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_manifest(path: Path) -> list[dict]:
    return [
        {"path": file.name, "bytes": file.stat().st_size, "sha256": file_sha256(file)}
        for file in sorted(path.glob("*.parquet"))
    ]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sample_dir = args.sample_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if (
        output_dir == sample_dir
        or output_dir in sample_dir.parents
        or sample_dir in output_dir.parents
    ):
        raise ValueError(
            f"Input and output directories must not overlap: {sample_dir}, {output_dir}"
        )
    ray_temp = args.ray_temp_dir.expanduser().resolve()
    for protected in (sample_dir, output_dir):
        if (
            ray_temp == protected
            or ray_temp in protected.parents
            or protected in ray_temp.parents
        ):
            raise ValueError(
                f"Ray temp path overlaps input/output: {ray_temp}, {protected}"
            )
    if args.actors < 1 or args.batch_size < 1:
        raise ValueError("--actors and --batch-size must be positive")
    if args.num_gpus < args.actors:
        raise ValueError("--num-gpus must be at least --actors")
    for split in SPLITS:
        files = sorted((sample_dir / split).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No sample parquet under {sample_dir / split}")
    sample_metadata_path = sample_dir / "metadata.json"
    sample_metadata = json.loads(sample_metadata_path.read_text())
    for split in SPLITS:
        files = sorted((sample_dir / split).glob("*.parquet"))
        if len(files) != 1:
            raise ValueError(f"Expected one sampled parquet for {split}")
        recorded_sha = sample_metadata.get("samples", {}).get(split, {}).get(
            "output_sha256"
        )
        if recorded_sha != file_sha256(files[0]):
            raise ValueError(f"Raw sample parquet hash mismatch for {split}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    if str(TFM_ROOT) not in sys.path:
        sys.path.insert(0, str(TFM_ROOT))

    import ray
    from ray.data import ActorPoolStrategy
    from src import ray_common as C
    from src.ray_tokenize import GPUTokenizer

    shutil.rmtree(ray_temp, ignore_errors=True)
    spill_dir = ray_temp / "spill"
    spill_dir.mkdir(parents=True, exist_ok=True)
    ray.init(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        include_dashboard=False,
        runtime_env=C.JOB_RUNTIME_ENV,
        _temp_dir=str(ray_temp),
        _system_config={
            "object_spilling_config": json.dumps(
                {
                    "type": "filesystem",
                    "params": {"directory_path": str(spill_dir)},
                }
            )
        },
    )

    metadata = {
        "sample_dir": str(sample_dir),
        "sample_metadata_sha256": file_sha256(sample_metadata_path),
        "merchant_hash_size": C.MERCHANT_HASH_SIZE,
        "arms": {},
    }
    try:
        for mode in MODES:
            arm = "legacy" if mode == "string_hash" else "fast"
            metadata["arms"][arm] = {"merchant_hash_mode": mode, "splits": {}}
            for split in SPLITS:
                destination = output_dir / arm / split
                ds = ray.data.read_parquet(str(sample_dir / split))
                tokenized = ds.map_batches(
                    GPUTokenizer,
                    fn_constructor_kwargs={
                        "merchant_hash_size": C.MERCHANT_HASH_SIZE,
                        "merchant_hash_mode": mode,
                        "carry_cols": ["row_id", "sample_weight"],
                    },
                    batch_size=args.batch_size,
                    compute=ActorPoolStrategy(size=args.actors),
                    # Keep half the CPU pool free for Ray's parquet read/write
                    # tasks; reserving every CPU for actors deadlocks the plan.
                    num_cpus=max(1, args.num_cpus // (2 * args.actors)),
                    num_gpus=1,
                    batch_format="cudf",
                ).select_columns(["token_ids", "label", "row_id", "sample_weight"])
                tokenized.write_parquet(str(destination))
                count = count_parquet_rows(destination)
                expected = count_parquet_rows(sample_dir / split)
                if count != expected:
                    raise RuntimeError(
                        f"{arm}/{split} wrote {count:,} rows; expected {expected:,}"
                    )
                metadata["arms"][arm]["splits"][split] = count
                metadata["arms"][arm].setdefault("output_files", {})[split] = (
                    parquet_manifest(destination)
                )
                print(
                    f"[{arm:6s}/{split:5s}] {count:,} tokenized transactions",
                    flush=True,
                )
    finally:
        ray.shutdown()

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
