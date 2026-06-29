#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fixed-seed Ray Train comparison for tokenized parquet corpora.

The runner is intentionally independent of the parquet tensor representation:
it decodes both Ray Arrow tensor columns (legacy path) and the fast fixed-size
binary tensors, then trains the same small Llama configuration on each corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TFM_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Named tokenized corpus; repeat to compare corpora concurrently.",
    )
    parser.add_argument(
        "--corpus-hash-mode",
        action="append",
        default=[],
        metavar="LABEL=MODE",
        help="Record a corpus merchant mapping (integer_mod/string_hash).",
    )
    parser.add_argument("--merchant-hash-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Checkpoint directory (defaults beside --output).",
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--workers-per-corpus", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--validation-sequences", type=int, default=512)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--num-cpus", type=int, default=64)
    parser.add_argument(
        "--align-nonmerchant",
        action="store_true",
        help=(
            "Pair corpora by all tokens except merchant positions before "
            "sampling. Use this when output row order differs between engines."
        ),
    )
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save each trained Hugging Face checkpoint for embedding validation.",
    )
    parser.add_argument("--ray-temp-dir", type=Path, default=Path("/dev/shm/tfm-downstream-ray"))
    return parser.parse_args()


def parse_corpora(values: list[str]) -> dict[str, Path]:
    corpora = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label):
            raise ValueError(
                f"Corpus label must contain only letters, digits, '_' or '-': {label!r}"
            )
        if label in corpora:
            raise ValueError(f"Duplicate or empty corpus label: {label!r}")
        for split in ("train", "val"):
            if not list((path / split).glob("*.parquet")):
                raise FileNotFoundError(f"No parquet files for {label}/{split}: {path / split}")
        corpora[label] = path
    return corpora


def parse_hash_modes(values: list[str], labels: set[str]) -> dict[str, str]:
    modes = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=MODE, got {value!r}")
        label, mode = value.split("=", 1)
        if label not in labels:
            raise ValueError(f"Hash mode names unknown corpus {label!r}")
        if label in modes:
            raise ValueError(f"Duplicate hash mode for corpus {label!r}")
        if mode not in {"integer_mod", "string_hash"}:
            raise ValueError(f"Unsupported merchant hash mode {mode!r}")
        modes[label] = mode
    missing = labels - set(modes)
    if missing:
        raise ValueError(
            "Explicit --corpus-hash-mode is required for: "
            + ", ".join(sorted(missing))
        )
    return modes


def validate_corpus_metadata(
    corpora: dict[str, Path], hash_modes: dict[str, str], hash_size: int
) -> None:
    for label, path in corpora.items():
        manifest_path = path / "_tokenization_manifest.json"
        if not manifest_path.exists():
            continue
        config = json.loads(manifest_path.read_text()).get("config", {})
        recorded = (config.get("merchant_hash_mode"), config.get("merchant_hash_size"))
        expected = (hash_modes[label], hash_size)
        if recorded != expected:
            raise ValueError(
                f"Corpus manifest mismatch for {label}: recorded={recorded}, "
                f"requested={expected}"
            )


def parquet_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.parquet")) or sorted(path.rglob("*.parquet"))


def paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def row_count(path: Path) -> int:
    return sum(pq.ParquetFile(file).metadata.num_rows for file in parquet_files(path))


def _decode_tensor_batch(batch: pa.RecordBatch) -> np.ndarray:
    array = batch.column(0)
    n = len(array)
    field = batch.schema.field(0)

    # Importing Ray registers its Arrow tensor extension type, so a legacy
    # tensor column can be exposed either as an extension array or as its list
    # storage depending on the process import order.
    if hasattr(array, "to_numpy_ndarray"):
        decoded = np.asarray(array.to_numpy_ndarray())
        if decoded.ndim != 2:
            raise ValueError(f"Expected rank-2 tensor batch, got {decoded.shape}")
        return decoded
    if isinstance(array, pa.ExtensionArray):
        array = array.storage

    if pa.types.is_fixed_size_binary(array.type):
        if array.null_count:
            raise ValueError("Null fixed-size tensor values are not supported")
        metadata = field.metadata or {}
        dtype_key = b"ray.data.fixed_size_binary_tensor.dtype"
        shape_key = b"ray.data.fixed_size_binary_tensor.shape"
        if dtype_key not in metadata or shape_key not in metadata:
            raise ValueError("Fixed-size tensor column is missing dtype/shape metadata")
        dtype = np.dtype(metadata[dtype_key].decode())
        shape = json.loads(metadata[shape_key].decode())
        width = array.type.byte_width
        elements = int(np.prod(shape))
        if elements * dtype.itemsize != width:
            raise ValueError(
                f"Tensor metadata {shape}/{dtype} does not match byte width {width}"
            )
        buffer = array.buffers()[1]
        start = array.offset * width
        return np.frombuffer(
            buffer,
            dtype=dtype,
            count=n * elements,
            offset=start,
        ).reshape((n, *shape))

    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        offsets = np.asarray(array.offsets)
        values = np.asarray(array.values)
        widths = np.diff(offsets)
        if len(widths) and not np.all(widths == widths[0]):
            raise ValueError("Variable-width input_ids rows are not supported")
        start = int(offsets[0])
        stop = int(offsets[-1])
        width = (stop - start) // max(n, 1)
        return values[start:stop].reshape(n, width)

    raise TypeError(f"Unsupported input_ids parquet type: {array.type}")


def load_selected_rows(path: Path, indices: np.ndarray) -> np.ndarray:
    """Load selected logical rows while preserving the requested index order."""
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        raise ValueError("At least one row index is required")
    if np.any(indices < 0):
        raise IndexError("Negative row indices are not supported")
    sort_order = np.argsort(indices, kind="stable")
    sorted_indices = indices[sort_order]
    output = None
    cursor = 0
    selected_cursor = 0

    for file in parquet_files(path):
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=2048, columns=["input_ids"]):
            decoded = _decode_tensor_batch(batch)
            end = cursor + len(decoded)
            hi = np.searchsorted(sorted_indices, end, side="left")
            if hi > selected_cursor:
                chosen = sorted_indices[selected_cursor:hi] - cursor
                if output is None:
                    output = np.empty(
                        (len(indices), decoded.shape[1]), dtype=decoded.dtype
                    )
                destinations = sort_order[selected_cursor:hi]
                output[destinations] = decoded[chosen]
                selected_cursor = hi
            cursor = end
            if selected_cursor == len(indices):
                return output

    raise IndexError(
        f"Could only resolve {selected_cursor}/{len(indices)} selected rows under {path}"
    )


def nonmerchant_fingerprints(path: Path) -> list[bytes]:
    """Fingerprint complete sequences after masking merchant token positions."""
    fingerprints: list[bytes] = []
    expected_width = None
    for file in parquet_files(path):
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=2048, columns=["input_ids"]):
            decoded = _decode_tensor_batch(batch)
            if decoded.ndim != 2:
                raise ValueError(f"Expected rank-2 input_ids, got {decoded.shape}")
            if expected_width is None:
                expected_width = decoded.shape[1]
            elif decoded.shape[1] != expected_width:
                raise ValueError(
                    f"Sequence width changed from {expected_width} to {decoded.shape[1]}"
                )
            if decoded.size and (decoded.min() < 0 or decoded.max() > np.iinfo(np.uint16).max):
                raise ValueError("Token IDs do not fit the canonical uint16 fingerprint")
            canonical = np.asarray(decoded, dtype="<u2", order="C").copy()
            canonical[:, 2::13] = 0
            fingerprints.extend(
                hashlib.blake2b(row, digest_size=16).digest() for row in canonical
            )
    return fingerprints


def build_nonmerchant_alignment(
    corpora: dict[str, Path], splits: tuple[str, ...] = ("train", "val")
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Return physical row indices paired in a canonical fingerprint order."""
    aligned = {label: {} for label in corpora}
    summary = {}
    for split in splits:
        buckets_by_label: dict[str, dict[bytes, list[int]]] = {}
        split_summary = {}
        for label, corpus in corpora.items():
            fingerprints = nonmerchant_fingerprints(corpus / split)
            buckets: dict[bytes, list[int]] = {}
            for index, fingerprint in enumerate(fingerprints):
                buckets.setdefault(fingerprint, []).append(index)
            buckets_by_label[label] = buckets
            duplicate_rows = sum(len(rows) - 1 for rows in buckets.values())
            split_summary[label] = {
                "rows": len(fingerprints),
                "unique_fingerprints": len(buckets),
                "duplicate_rows": duplicate_rows,
            }

        common = set.intersection(
            *(set(buckets) for buckets in buckets_by_label.values())
        )
        paired = {label: [] for label in corpora}
        canonical = []
        for fingerprint in sorted(common):
            multiplicity = min(
                len(buckets_by_label[label][fingerprint]) for label in corpora
            )
            canonical.extend([fingerprint] * multiplicity)
            for label in corpora:
                paired[label].extend(
                    buckets_by_label[label][fingerprint][:multiplicity]
                )

        if not canonical:
            raise ValueError(f"No non-merchant-aligned rows found for split {split}")
        canonical_digest = hashlib.blake2b(
            b"".join(canonical), digest_size=32
        ).hexdigest()
        for label in corpora:
            aligned[label][split] = np.asarray(paired[label], dtype=np.int64)
            split_summary[label]["paired_rows"] = len(paired[label])
            split_summary[label]["unmatched_rows"] = (
                split_summary[label]["rows"] - len(paired[label])
            )
        split_summary["paired_rows"] = len(canonical)
        split_summary["canonical_fingerprint_blake2b"] = canonical_digest
        summary[split] = split_summary
    return aligned, summary


def verify_selected_alignment(
    corpora: dict[str, Path],
    indices: dict[str, np.ndarray],
    split: str,
) -> dict:
    """Directly prove selected pairs differ only in merchant token slots."""
    loaded = {
        label: load_selected_rows(path / split, indices[label])
        for label, path in corpora.items()
    }
    reference_label = next(iter(corpora))
    reference = loaded[reference_label]
    merchant_mask = np.zeros(reference.shape[1], dtype=bool)
    merchant_mask[2::13] = True
    result = {}
    for label, rows in loaded.items():
        if rows.shape != reference.shape:
            raise ValueError(
                f"Aligned shape mismatch: {reference_label}={reference.shape}, "
                f"{label}={rows.shape}"
            )
        if not np.array_equal(reference[:, ~merchant_mask], rows[:, ~merchant_mask]):
            raise ValueError(
                f"Aligned {split} rows for {label} differ outside merchant positions"
            )
        normalized = np.asarray(rows, dtype="<u2", order="C")
        result[label] = {
            "selected_rows": len(rows),
            "selected_tensor_blake2b": hashlib.blake2b(
                normalized, digest_size=32
            ).hexdigest(),
            "merchant_tokens_different_from_reference": int(
                np.count_nonzero(reference[:, merchant_mask] != rows[:, merchant_mask])
            ),
        }
    return result


def corpus_manifest(path: Path) -> dict:
    manifest = {}
    for split in ("train", "val"):
        files = []
        for file in parquet_files(path / split):
            stat = file.stat()
            files.append(
                {
                    "path": str(file.relative_to(path)),
                    "bytes": stat.st_size,
                    "rows": pq.ParquetFile(file).metadata.num_rows,
                }
            )
        manifest[split] = files
    return manifest


def code_provenance() -> dict:
    tracked_paths = [
        Path(__file__).resolve(),
        TFM_ROOT / "scripts" / "tokenize_splits.py",
        TFM_ROOT / "src" / "ray_tokenize.py",
        TFM_ROOT / "src" / "tokenizer" / "financial_pipeline.py",
    ]
    hashes = {
        str(path.relative_to(TFM_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tracked_paths
    }
    repositories = {}
    for label, root in (("workspace", TFM_ROOT.parent), ("tfm_repo", TFM_ROOT)):
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--short"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            repositories[label] = {
                "root": str(root),
                "commit": commit,
                "branch": branch,
                "status": status,
            }
        except (OSError, subprocess.CalledProcessError):
            repositories[label] = {"root": str(root), "unavailable": True}
    return {"repositories": repositories, "file_sha256": hashes}


def model_config() -> dict:
    return {
        "vocab_size": 6251,
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 1408,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
        "hidden_act": "silu",
        "rms_norm_eps": 1.0e-5,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "use_cache": False,
    }


def train_loop(config: dict) -> None:
    import random
    import time

    import ray.train
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from ray.train.torch import prepare_model
    from transformers import LlamaConfig, LlamaForCausalLM

    context = ray.train.get_context()
    rank = context.get_world_rank()
    world_size = context.get_world_size()
    seed = int(config["seed"])

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")

    load_started = time.perf_counter()
    local_batch = int(config["local_batch_size"])
    train_indices = np.asarray(config["train_indices"], dtype=np.int64).reshape(
        config["steps"], world_size, local_batch
    )[:, rank, :]
    validation_indices = np.asarray(config["validation_indices"], dtype=np.int64)
    validation_indices = validation_indices[rank::world_size]

    corpus = Path(config["corpus"])
    train_rows = load_selected_rows(corpus / "train", train_indices.reshape(-1))
    validation_rows = load_selected_rows(corpus / "val", validation_indices)
    load_seconds = time.perf_counter() - load_started

    setup_started = time.perf_counter()
    llama_config = LlamaConfig(**config["model"])
    llama_config.merchant_hash_mode = config["merchant_hash_mode"]
    llama_config.merchant_hash_size = config["merchant_hash_size"]
    llama_config.tokenized_corpus = config["corpus"]
    model = LlamaForCausalLM(llama_config)
    initial_parameter_hash = hashlib.blake2b(digest_size=32)
    for parameter in model.parameters():
        initial_parameter_hash.update(parameter.detach().numpy())
    initial_parameter_hash = initial_parameter_hash.hexdigest()
    model = prepare_model(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.077
    )
    capability_major = torch.cuda.get_device_capability()[0]
    amp_dtype = (
        torch.bfloat16
        if capability_major >= 8 and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    device = torch.device("cuda")
    setup_seconds = time.perf_counter() - setup_started

    model.train()
    losses = []
    skipped_optimizer_steps = 0
    train_started = time.perf_counter()
    for step in range(config["steps"]):
        ids = torch.as_tensor(
            train_rows[step * local_batch : (step + 1) * local_batch],
            device=device,
            dtype=torch.long,
        )
        labels = ids.clone()
        labels[labels == config["pad_token_id"]] = -100
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            loss = model(input_ids=ids, labels=labels).loss
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        skipped_optimizer_steps += int(scaler.get_scale() < scale_before)

        valid_targets = (labels[:, 1:] != -100).sum().float()
        reduced = torch.stack(
            [loss.detach().float() * valid_targets, valid_targets]
        )
        if dist.is_initialized():
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        losses.append(float((reduced[0] / reduced[1]).item()))
        if rank == 0 and (step + 1) % config["report_every"] == 0:
            print(
                f"[{config['label']}] step {step + 1:4d}/{config['steps']} "
                f"loss={losses[-1]:.4f}",
                flush=True,
            )
    train_seconds = time.perf_counter() - train_started

    model.eval()
    evaluation_model = model.module if hasattr(model, "module") else model
    overall_sum = torch.zeros(1, device=device, dtype=torch.float64)
    overall_count = torch.zeros(1, device=device, dtype=torch.float64)
    merchant_sum = torch.zeros(1, device=device, dtype=torch.float64)
    merchant_count = torch.zeros(1, device=device, dtype=torch.float64)
    val_batch = int(config["validation_batch_size"])

    validation_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(validation_rows), val_batch):
            ids = torch.as_tensor(
                validation_rows[start : start + val_batch],
                device=device,
                dtype=torch.long,
            )
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = evaluation_model(input_ids=ids).logits
            shift_logits = logits[:, :-1].float()
            shift_labels = ids[:, 1:]
            valid = shift_labels != config["pad_token_id"]
            token_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="none",
            ).reshape_as(shift_labels)
            overall_sum += token_loss[valid].double().sum()
            overall_count += valid.sum()

            positions = torch.arange(1, ids.shape[1], device=device)
            merchant = valid & ((positions % 13) == 2).unsqueeze(0)
            merchant_sum += token_loss[merchant].double().sum()
            merchant_count += merchant.sum()

    metrics = torch.cat(
        [overall_sum, overall_count, merchant_sum, merchant_count]
    )
    if dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    validation_seconds = time.perf_counter() - validation_started

    if rank == 0:
        tail = losses[-min(25, len(losses)) :]
        checkpoint = None
        if config["save_checkpoint"]:
            checkpoint = Path(config["checkpoint_dir"])
            checkpoint.mkdir(parents=True, exist_ok=True)
            evaluation_model.save_pretrained(checkpoint, safe_serialization=True)
            (checkpoint / "tokenization_metadata.json").write_text(
                json.dumps(
                    {
                        "corpus": config["corpus"],
                        "label": config["label"],
                        "merchant_hash_mode": config["merchant_hash_mode"],
                        "merchant_hash_size": config["merchant_hash_size"],
                        "seed": seed,
                        "steps": config["steps"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        reported = {
            "label": config["label"],
            "steps": config["steps"],
            "final_train_loss": losses[-1],
            "median_last25_train_loss": float(np.median(tail)),
            "min_train_loss": min(losses),
            "validation_loss": float((metrics[0] / metrics[1]).item()),
            "merchant_validation_loss": float((metrics[2] / metrics[3]).item()),
            "validation_tokens": int(metrics[1].item()),
            "merchant_validation_tokens": int(metrics[3].item()),
            "load_seconds": load_seconds,
            "setup_seconds": setup_seconds,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "skipped_optimizer_steps": skipped_optimizer_steps,
            "initial_parameter_blake2b": initial_parameter_hash,
            "precision": "bf16" if amp_dtype == torch.bfloat16 else "fp16",
            "gpu": torch.cuda.get_device_name(),
            "checkpoint": str(checkpoint) if checkpoint else None,
        }
        metrics_path = Path(config["metrics_path"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(reported, indent=2, sort_keys=True) + "\n")
        ray.train.report(reported)
    else:
        ray.train.report({"label": config["label"], "steps": config["steps"]})


def main() -> None:
    args = parse_args()
    corpora = parse_corpora(args.corpus)
    hash_modes = parse_hash_modes(args.corpus_hash_mode, set(corpora))
    if args.merchant_hash_size < 1:
        raise ValueError("--merchant-hash-size must be positive")
    validate_corpus_metadata(corpora, hash_modes, args.merchant_hash_size)
    for name in (
        "steps",
        "workers_per_corpus",
        "global_batch_size",
        "validation_sequences",
        "validation_batch_size",
        "report_every",
        "num_cpus",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.global_batch_size % args.workers_per_corpus:
        raise ValueError("--global-batch-size must divide evenly across workers")
    local_batch = args.global_batch_size // args.workers_per_corpus
    total_gpus = len(corpora) * args.workers_per_corpus
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    counts = {
        label: {
            "train": row_count(path / "train"),
            "val": row_count(path / "val"),
        }
        for label, path in corpora.items()
    }

    alignment_summary = None
    if args.align_nonmerchant:
        aligned_indices, alignment_summary = build_nonmerchant_alignment(corpora)
    else:
        for split in ("train", "val"):
            split_counts = {item[split] for item in counts.values()}
            if len(corpora) > 1 and len(split_counts) != 1:
                raise ValueError(
                    f"Unaligned multi-corpus run requires equal {split} counts; "
                    "use --align-nonmerchant"
                )
        aligned_indices = {
            label: {
                split: np.arange(counts[label][split], dtype=np.int64)
                for split in ("train", "val")
            }
            for label in corpora
        }

    train_seed, validation_seed = np.random.SeedSequence(args.seed).spawn(2)
    train_rng = np.random.default_rng(train_seed)
    validation_rng = np.random.default_rng(validation_seed)
    reference_label = next(iter(corpora))
    reference_train_count = len(aligned_indices[reference_label]["train"])
    reference_val_count = len(aligned_indices[reference_label]["val"])
    train_size = args.steps * args.global_batch_size
    canonical_train_indices = train_rng.choice(
        reference_train_count,
        size=train_size,
        replace=train_size > reference_train_count,
    )
    validation_size = min(args.validation_sequences, reference_val_count)
    if validation_size < args.workers_per_corpus:
        raise ValueError(
            "Validation sample must contain at least one sequence per worker"
        )
    canonical_validation_indices = validation_rng.choice(
        reference_val_count, size=validation_size, replace=False
    )
    train_indices = {
        label: mapping["train"][canonical_train_indices]
        for label, mapping in aligned_indices.items()
    }
    validation_indices = {
        label: mapping["val"][canonical_validation_indices]
        for label, mapping in aligned_indices.items()
    }

    selection_verification = {
        "train": verify_selected_alignment(corpora, train_indices, "train"),
        "val": verify_selected_alignment(corpora, validation_indices, "val"),
    }

    args.output = args.output.expanduser().resolve()
    results_root = args.output.parent / f"{args.output.stem}_ray_runs"
    checkpoint_root = (
        args.checkpoint_root.expanduser().resolve()
        if args.checkpoint_root
        else args.output.parent / f"{args.output.stem}_checkpoints"
    )
    for label, corpus in corpora.items():
        for output_label, candidate in (
            ("Ray results", results_root),
            ("checkpoint", checkpoint_root),
        ):
            if paths_overlap(candidate, corpus):
                raise ValueError(
                    f"{output_label} path overlaps corpus {label}: "
                    f"{candidate}, {corpus}"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(results_root, ignore_errors=True)
    results_root.mkdir(parents=True, exist_ok=True)
    if args.save_checkpoints:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    args.ray_temp_dir = args.ray_temp_dir.expanduser().resolve()
    for output_label, candidate in (
        ("Ray results", results_root),
        ("checkpoint", checkpoint_root),
        ("result output directory", args.output.parent),
    ):
        if paths_overlap(args.ray_temp_dir, candidate):
            raise ValueError(
                f"Ray temp path overlaps {output_label}: "
                f"{args.ray_temp_dir}, {candidate}"
            )
    for label, corpus in corpora.items():
        if paths_overlap(args.ray_temp_dir, corpus):
            raise ValueError(
                f"Ray temp path overlaps corpus {label}: "
                f"{args.ray_temp_dir}, {corpus}"
            )
    shutil.rmtree(args.ray_temp_dir, ignore_errors=True)
    args.ray_temp_dir.mkdir(parents=True, exist_ok=True)

    import ray
    import torch
    import transformers
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    visible_gpus = torch.cuda.device_count()
    if total_gpus > visible_gpus:
        raise RuntimeError(
            f"Requested {total_gpus} GPUs but only {visible_gpus} are visible"
        )

    ray.init(
        num_cpus=args.num_cpus,
        num_gpus=total_gpus,
        include_dashboard=False,
        _temp_dir=str(args.ray_temp_dir),
    )

    trainers = {}
    metrics_paths = {}
    cpus_per_worker = max(1, args.num_cpus // total_gpus)
    for label, path in corpora.items():
        checkpoint_dir = checkpoint_root / label
        if args.save_checkpoints:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        metrics_paths[label] = results_root / f"{label}-metrics.json"
        config = {
            "label": label,
            "corpus": str(path),
            "merchant_hash_mode": hash_modes[label],
            "merchant_hash_size": args.merchant_hash_size,
            "model": model_config(),
            "pad_token_id": 0,
            "steps": args.steps,
            "local_batch_size": local_batch,
            "validation_batch_size": args.validation_batch_size,
            "train_indices": train_indices[label].tolist(),
            "validation_indices": validation_indices[label].tolist(),
            "seed": args.seed,
            "report_every": args.report_every,
            "save_checkpoint": args.save_checkpoints,
            "checkpoint_dir": str(checkpoint_dir),
            "metrics_path": str(metrics_paths[label]),
        }
        trainers[label] = TorchTrainer(
            train_loop,
            train_loop_config=config,
            scaling_config=ScalingConfig(
                num_workers=args.workers_per_corpus,
                use_gpu=True,
                resources_per_worker={"CPU": cpus_per_worker},
            ),
            run_config=RunConfig(
                name=f"tokenization-{label}",
                storage_path=str(results_root),
            ),
        )

    results = {}
    try:
        with ThreadPoolExecutor(max_workers=len(trainers)) as executor:
            pending = {
                executor.submit(trainer.fit): label
                for label, trainer in trainers.items()
            }
            for future in as_completed(pending):
                label = pending[future]
                future.result()
                results[label] = json.loads(metrics_paths[label].read_text())
                print(f"Completed {label}: {results[label]}", flush=True)
    finally:
        ray.shutdown()

    comparisons = {}
    labels = list(corpora)
    if len(labels) > 1:
        baseline = labels[0]
        for label in labels[1:]:
            comparisons[f"{label}_minus_{baseline}"] = {
                metric: results[label][metric] - results[baseline][metric]
                for metric in (
                    "final_train_loss",
                    "median_last25_train_loss",
                    "validation_loss",
                    "merchant_validation_loss",
                )
            }

    payload = {
        "seed": args.seed,
        "steps": args.steps,
        "global_batch_size": args.global_batch_size,
        "workers_per_corpus": args.workers_per_corpus,
        "validation_sequences": validation_size,
        "alignment": alignment_summary,
        "selection_verification": selection_verification,
        "counts": counts,
        "corpora": {label: str(path) for label, path in corpora.items()},
        "merchant_hash_modes": hash_modes,
        "merchant_hash_size": args.merchant_hash_size,
        "model_config": model_config(),
        "manifests": {
            label: corpus_manifest(path) for label, path in corpora.items()
        },
        "provenance": code_provenance(),
        "runtime": {
            "ray": ray.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "visible_gpus": visible_gpus,
        },
        "metrics": results,
        "comparisons": comparisons,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "alignment": {
                    split: details["paired_rows"]
                    for split, details in (alignment_summary or {}).items()
                },
                "metrics": results,
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
