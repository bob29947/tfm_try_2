# SPDX-License-Identifier: Apache-2.0
"""Creation and verification of tokenization provenance manifests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from .contract import SPLITS, TOKENIZATION_MANIFEST
from .outputs import nonempty_dir, parquet_files


TFM_ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_file_manifest(source_root: Path) -> dict[str, list[dict]]:
    return {
        split: [
            {
                "path": str(file.relative_to(source_root)),
                "bytes": file.stat().st_size,
                "sha256": file_sha256(file),
            }
            for file in parquet_files(source_root / split)
        ]
        for split in SPLITS
    }


def code_manifest() -> dict[str, str]:
    code_files = [
        TFM_ROOT / "scripts" / "tokenize_splits.py",
        TFM_ROOT / "src" / "gpu_parquet.py",
        TFM_ROOT / "src" / "ray_common.py",
        TFM_ROOT / "src" / "ray_tokenize.py",
        *sorted((TFM_ROOT / "src" / "tokenization").glob("*.py")),
        *sorted((TFM_ROOT / "src" / "tokenizer").glob("*.py")),
    ]
    return {
        str(path.relative_to(TFM_ROOT)): file_sha256(path) for path in code_files
    }


def runtime_manifest() -> dict[str, str]:
    import numpy
    import pyarrow
    import ray

    versions = {
        "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__,
        "ray": ray.__version__,
        "python": sys.version.split()[0],
    }
    try:
        import cudf

        versions["cudf"] = cudf.__version__
    except ImportError:
        versions["cudf"] = "unavailable"
    try:
        import cupy

        versions["cupy"] = cupy.__version__
        name = cupy.cuda.runtime.getDeviceProperties(0)["name"]
        versions["gpu"] = name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        versions["cupy"] = versions.get("cupy", "unavailable")
        versions["gpu"] = "unavailable"
    return versions


def output_file_manifest(output_dir: Path) -> dict[str, list[dict]]:
    import pyarrow.parquet as pq

    return {
        split: [
            {
                "path": str(file.relative_to(output_dir)),
                "bytes": file.stat().st_size,
                "rows": pq.ParquetFile(file).metadata.num_rows,
                "sha256": file_sha256(file),
            }
            for file in parquet_files(output_dir / split)
        ]
        for split in SPLITS
    }


def validate_existing_manifest(
    output_dir: Path,
    expected_config: dict,
    overwrite: bool,
) -> None:
    if overwrite or not nonempty_dir(output_dir):
        return
    manifest_path = output_dir / TOKENIZATION_MANIFEST
    if not manifest_path.exists():
        raise RuntimeError(
            f"Existing output has no tokenization manifest: {output_dir}. "
            "Pass --overwrite to regenerate it safely."
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config") != expected_config:
        raise RuntimeError(
            "Existing tokenization settings do not match this command. "
            f"Recorded={manifest.get('config')!r}, requested={expected_config!r}. "
            "Use a new --output-dir or pass --overwrite."
        )
    current_sources = source_file_manifest(Path(expected_config["source"]))
    if manifest.get("source_files") != current_sources:
        raise RuntimeError(
            "Source parquet files changed since this tokenized output was written. "
            "Use a new --output-dir or pass --overwrite."
        )
    if manifest.get("code_sha256") != code_manifest():
        raise RuntimeError(
            "Tokenizer implementation changed since this output was written. "
            "Use a new --output-dir or pass --overwrite."
        )
    if manifest.get("runtime") != runtime_manifest():
        raise RuntimeError(
            "Runtime package/GPU versions changed since this output was written. "
            "Use a new --output-dir or pass --overwrite."
        )
    current_outputs = output_file_manifest(output_dir)
    if manifest.get("output_files") != current_outputs:
        raise RuntimeError(
            "Tokenized parquet files changed since the manifest was written. "
            "Use a new --output-dir or pass --overwrite."
        )
    current_counts = {
        split: sum(item["rows"] for item in files)
        for split, files in current_outputs.items()
    }
    if manifest.get("sequence_counts") != current_counts:
        raise RuntimeError(
            f"Tokenized row counts changed: recorded={manifest.get('sequence_counts')}, "
            f"current={current_counts}."
        )


def write_tokenization_manifest(
    output_dir: Path,
    config: dict,
    seq_counts: dict[str, int],
    elapsed_s: float,
) -> None:
    source_root = Path(config["source"])
    manifest = {
        "config": config,
        "sequence_counts": seq_counts,
        "source_files": source_file_manifest(source_root),
        "elapsed_seconds": elapsed_s,
        "output_files": output_file_manifest(output_dir),
        "code_sha256": code_manifest(),
        "runtime": runtime_manifest(),
    }
    (output_dir / TOKENIZATION_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


__all__ = [
    "code_manifest",
    "file_sha256",
    "output_file_manifest",
    "runtime_manifest",
    "source_file_manifest",
    "validate_existing_manifest",
    "write_tokenization_manifest",
]
