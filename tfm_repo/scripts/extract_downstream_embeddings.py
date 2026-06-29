#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract paired NB04 transaction embeddings with Ray GPU actors."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TFM_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token_dir", type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="ARM=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--actors-per-arm", type=int, default=4)
    parser.add_argument("--num-cpus", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32_768)
    parser.add_argument("--max-length", type=int, default=14)
    parser.add_argument(
        "--ray-temp-dir", type=Path, default=Path("/dev/shm/tfm-downstream-embed")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    checkpoints = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected ARM=PATH, got {value!r}")
        arm, raw_path = value.split("=", 1)
        if arm in checkpoints:
            raise ValueError(f"Duplicate checkpoint arm {arm!r}")
        path = Path(raw_path).expanduser().resolve()
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"Missing checkpoint config: {path}")
        checkpoints[arm] = path
    if set(checkpoints) != {"legacy", "fast"}:
        raise ValueError("Exactly legacy=... and fast=... checkpoints are required")
    return checkpoints


def parquet_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.parquet")) or sorted(path.rglob("*.parquet"))


def row_count(path: Path) -> int:
    return sum(pq.ParquetFile(file).metadata.num_rows for file in parquet_files(path))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_manifest(path: Path) -> list[dict]:
    return [
        {
            "path": str(file.relative_to(path)),
            "bytes": file.stat().st_size,
            "sha256": file_sha256(file),
        }
        for file in parquet_files(path)
    ]


def scalar_columns(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tables = [
        pq.read_table(file, columns=["row_id", "label", "sample_weight"])
        for file in parquet_files(path)
    ]
    table = pa.concat_tables(tables)
    return tuple(
        table[column].combine_chunks().to_numpy(zero_copy_only=False)
        for column in ("row_id", "label", "sample_weight")
    )


def verify_paired_outputs(output_dir: Path) -> dict:
    summary = {}
    for split in SPLITS:
        arm_values = {
            arm: scalar_columns(output_dir / arm / split)
            for arm in ("legacy", "fast")
        }
        normalized = {}
        for arm, (row_id, label, weight) in arm_values.items():
            order = np.argsort(row_id, kind="stable")
            row_id = row_id[order]
            if len(row_id) != len(np.unique(row_id)):
                raise ValueError(f"Duplicate row_id in {arm}/{split}")
            normalized[arm] = (row_id, label[order], weight[order])
        legacy = normalized["legacy"]
        fast = normalized["fast"]
        for index, name in enumerate(("row_id", "label", "sample_weight")):
            if not np.array_equal(legacy[index], fast[index]):
                raise ValueError(f"Paired {split} {name} mismatch")
        summary[split] = {
            "rows": len(legacy[0]),
            "fraud": int(legacy[1].sum()),
            "sorted_row_ids_sha256": hashlib.sha256(
                np.asarray(legacy[0], dtype="<i8", order="C").tobytes()
            ).hexdigest(),
        }
    return summary


def main() -> None:
    args = parse_args()
    token_dir = args.token_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoints = parse_checkpoints(args.checkpoint)
    protected_inputs = {"token corpus": token_dir, **checkpoints}
    for label, protected in protected_inputs.items():
        if (
            output_dir == protected
            or output_dir in protected.parents
            or protected in output_dir.parents
        ):
            raise ValueError(
                f"Output overlaps {label} input: {output_dir}, {protected}"
            )
    ray_temp = args.ray_temp_dir.expanduser().resolve()
    for label, protected in {**protected_inputs, "output": output_dir}.items():
        if (
            ray_temp == protected
            or ray_temp in protected.parents
            or protected in ray_temp.parents
        ):
            raise ValueError(
                f"Ray temp path overlaps {label}: {ray_temp}, {protected}"
            )
    token_metadata = json.loads((token_dir / "metadata.json").read_text())
    for arm in ("legacy", "fast"):
        for split in SPLITS:
            recorded_files = (
                token_metadata.get("arms", {})
                .get(arm, {})
                .get("output_files", {})
                .get(split)
            )
            if recorded_files != parquet_manifest(token_dir / arm / split):
                raise ValueError(f"Token parquet manifest mismatch for {arm}/{split}")
    if args.actors_per_arm < 1 or args.batch_size < 1:
        raise ValueError("--actors-per-arm and --batch-size must be positive")
    if args.max_length < 14:
        raise ValueError("--max-length must fit BOS + 12 fields + EOS")

    checkpoint_metadata = {}
    for arm, checkpoint in checkpoints.items():
        config = json.loads((checkpoint / "config.json").read_text())
        metadata = json.loads((checkpoint / "tokenization_metadata.json").read_text())
        config_mode = config.get("merchant_hash_mode")
        metadata_mode = metadata.get("merchant_hash_mode")
        token_mode = token_metadata["arms"][arm]["merchant_hash_mode"]
        config_size = config.get("merchant_hash_size")
        metadata_size = metadata.get("merchant_hash_size")
        token_size = token_metadata.get("merchant_hash_size")
        if not config_mode == metadata_mode == token_mode:
            raise ValueError(
                f"Merchant mapping mismatch for {arm}: "
                f"config={config_mode}, checkpoint={metadata_mode}, tokens={token_mode}"
            )
        if not config_size == metadata_size == token_size:
            raise ValueError(
                f"Merchant hash-size mismatch for {arm}: "
                f"config={config_size}, checkpoint={metadata_size}, tokens={token_size}"
            )
        weights = checkpoint / "model.safetensors"
        checkpoint_metadata[arm] = {
            "path": str(checkpoint),
            "merchant_hash_mode": config_mode,
            "merchant_hash_size": config_size,
            "model_sha256": file_sha256(weights),
        }
        for split in SPLITS:
            if not parquet_files(token_dir / arm / split):
                raise FileNotFoundError(f"No token parquet for {arm}/{split}")

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
    from src.ray_embed import EmbeddingExtractor

    total_gpus = 2 * args.actors_per_arm
    shutil.rmtree(ray_temp, ignore_errors=True)
    spill_dir = ray_temp / "spill"
    spill_dir.mkdir(parents=True, exist_ok=True)
    ray.init(
        num_cpus=args.num_cpus,
        num_gpus=total_gpus,
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

    def run_arm(arm: str) -> dict:
        counts = {}
        for split in SPLITS:
            destination = output_dir / arm / split
            ds = ray.data.read_parquet(str(token_dir / arm / split))
            embeddings = ds.map_batches(
                EmbeddingExtractor,
                fn_constructor_kwargs={
                    "model_dir": str(checkpoints[arm]),
                    "max_length": args.max_length,
                    "pooling": "last_token",
                },
                batch_size=args.batch_size,
                compute=ActorPoolStrategy(size=args.actors_per_arm),
                num_cpus=max(1, args.num_cpus // (4 * args.actors_per_arm)),
                num_gpus=1,
                batch_format="numpy",
            ).select_columns(["embedding", "label", "row_id", "sample_weight"])
            embeddings.write_parquet(str(destination))
            counts[split] = row_count(destination)
            expected = row_count(token_dir / arm / split)
            if counts[split] != expected:
                raise RuntimeError(
                    f"{arm}/{split} embedded {counts[split]:,}; expected {expected:,}"
                )
            print(
                f"[{arm:6s}/{split:5s}] {counts[split]:,} embeddings",
                flush=True,
            )
        return counts

    counts = {}
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = {
                executor.submit(run_arm, arm): arm for arm in ("legacy", "fast")
            }
            for future in as_completed(pending):
                arm = pending[future]
                counts[arm] = future.result()
    finally:
        ray.shutdown()

    pairing = verify_paired_outputs(output_dir)
    metadata = {
        "token_dir": str(token_dir),
        "token_metadata_sha256": file_sha256(token_dir / "metadata.json"),
        "max_length": args.max_length,
        "pooling": "last_token",
        "embedding_dim": 512,
        "checkpoints": checkpoint_metadata,
        "counts": counts,
        "pairing": pairing,
        "output_files": {
            arm: {
                split: parquet_manifest(output_dir / arm / split)
                for split in SPLITS
            }
            for arm in ("legacy", "fast")
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
