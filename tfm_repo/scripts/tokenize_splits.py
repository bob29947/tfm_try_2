#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create NB02-style tokenized train/val/test parquet from temporal splits."""

from __future__ import annotations

import argparse
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
    parser.add_argument("split_dir", type=Path, help="Directory containing train/val/test parquet dirs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Tokenized output directory. Defaults to sibling tokenized_v*.",
    )
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--actors", type=int, default=2)
    parser.add_argument("--num-gpus-per-actor", type=float, default=1.0)
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
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else default_output_dir(split_dir)
    return split_dir, output_dir


def init_ray(C, address: str):
    import ray

    if ray.is_initialized():
        return ray
    if address == "local":
        ray.init(runtime_env=C.JOB_RUNTIME_ENV)
    else:
        ray.init(address=address, runtime_env=C.JOB_RUNTIME_ENV)
    return ray


def tokenize_split(ray, C, split_dir: Path, output_dir: Path, split: str, args: argparse.Namespace) -> int:
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


def main() -> None:
    args = parse_args()
    split_dir, output_dir = validate_args(args)

    if str(TFM_ROOT) not in sys.path:
        sys.path.insert(0, str(TFM_ROOT))

    os.environ["TFM_SPLIT_DIR"] = str(split_dir)
    os.environ["TFM_TOKENIZED_DIR"] = str(output_dir)

    from src import ray_common as C

    ray = init_ray(C, args.ray_address)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Split input:    {split_dir}")
    print(f"Tokenized out:  {output_dir}")
    print(f"Ray GPUs:       {ray.cluster_resources().get('GPU', 0):.0f}")
    print(f"SEQ_LENGTH={C.SEQ_LENGTH}, SEQ_CHUNK_SIZE={C.SEQ_CHUNK_SIZE}\n")

    started = time.time()
    seq_counts = {}
    for split in SPLITS:
        seq_counts[split] = tokenize_split(ray, C, split_dir, output_dir, split, args)

    print("\nTokenized outputs for NB03:")
    for split in SPLITS:
        print(f"  {split:5s}: {seq_counts[split]:>10,} sequences -> {output_dir / split}")
    print(f"\nPoint NB03 at this output with:\n  export TFM_TOKENIZED_DIR={output_dir}")
    print(f"Done in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
