#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create NB02-style tokenized train/val/test parquet from temporal splits."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


TFM_ROOT = Path(__file__).resolve().parents[2]
if str(TFM_ROOT) not in sys.path:
    sys.path.insert(0, str(TFM_ROOT))

from .config import default_output_dir, tokenization_config
from .contract import SPLITS, TOKENIZATION_MANIFEST
from .manifest import (
    validate_existing_manifest,
    write_tokenization_manifest,
)
from .outputs import nonempty_dir
from .profiles import available_profiles, load_profile
from .runner import (
    init_ray,
    select_gpu_runtime_env,
    tokenize_gpu_parquet,
    tokenize_split,
    validate_cluster_resources,
)


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokenize temporal splits for NB03.")
    profile_names = available_profiles()
    parser.add_argument(
        "--profile",
        default=None,
        metavar="NAME",
        help=(
            "Apply a bundled benchmark profile before explicit flags. "
            f"Available: {', '.join(profile_names) if profile_names else 'none'}."
        ),
    )
    parser.add_argument(
        "split_dir",
        type=Path,
        help="Directory containing train/val/test parquet dirs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Tokenized output directory. Defaults to sibling tokenized_v*.",
    )
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument(
        "--actors",
        type=int,
        default=2,
        help="Number of long-lived workers. Defaults to the documented 2-GPU deployment.",
    )
    parser.add_argument("--num-cpus-per-actor", type=int, default=16)
    parser.add_argument("--num-gpus-per-actor", type=float, default=1.0)
    parser.add_argument(
        "--local-num-cpus",
        type=int,
        default=None,
        help="Local Ray only: advertise this many CPUs to Ray.",
    )
    parser.add_argument(
        "--local-num-gpus",
        type=int,
        default=None,
        help="Local Ray only: advertise this many GPUs to Ray.",
    )
    parser.add_argument(
        "--ray-temp-dir",
        type=Path,
        default=Path(os.environ.get("TFM_RAY_TEMP_DIR", "/dev/shm/tfm-ray")),
        help=(
            "Local Ray only: root temp/session directory. Defaults to "
            "TFM_RAY_TEMP_DIR or /dev/shm/tfm-ray to avoid slow/full /tmp."
        ),
    )
    parser.add_argument(
        "--ray-spill-dir",
        type=Path,
        default=(
            Path(os.environ["TFM_RAY_SPILL_DIR"])
            if os.environ.get("TFM_RAY_SPILL_DIR")
            else None
        ),
        help=(
            "Local Ray only: filesystem object-spill directory. Defaults to "
            "<ray-temp-dir>/spill."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=("gpu-parquet", "legacy"),
        default="gpu-parquet",
        help=(
            "Use the application-local Ray Core GPU parquet runner or "
            "the legacy Ray Data groupby path."
        ),
    )
    parser.add_argument(
        "--merchant-hash-mode",
        choices=("integer_mod", "string_hash"),
        default="integer_mod",
        help=(
            "Merchant bucket mapping for the legacy Ray Data engine. The fast "
            "gpu-parquet engine always uses integer_mod."
        ),
    )
    parser.add_argument(
        "--row-groups-per-batch",
        type=int,
        default=64,
        help="gpu-parquet engine only: parquet row groups each GPU actor reads per batch.",
    )
    parser.add_argument(
        "--write-threads-per-actor",
        type=int,
        default=12,
        help=(
            "gpu-parquet engine only: concurrent parquet shard writers per actor. "
            "Large outputs are striped across these writers."
        ),
    )
    parser.add_argument(
        "--output-shard-size-mb",
        type=int,
        default=128,
        help=(
            "gpu-parquet engine only: target maximum uncompressed bytes per "
            "striped output shard, in MiB."
        ),
    )
    parser.add_argument(
        "--output-dtype",
        choices=("uint16", "int32", "int64"),
        default="uint16",
        help="gpu-parquet engine only: dtype for fixed-width input_ids tensors.",
    )
    parser.add_argument(
        "--output-format",
        choices=("binary-tensor", "arrow-tensor"),
        default="binary-tensor",
        help=(
            "gpu-parquet engine only: write fast fixed-size-binary tensor "
            "parquet or Ray's standard Arrow tensor extension."
        ),
    )
    parser.add_argument(
        "--use-dictionary",
        action="store_true",
        help="gpu-parquet engine only: enable parquet dictionary encoding.",
    )
    parser.add_argument(
        "--compression",
        default="none",
        help="gpu-parquet engine only: parquet compression codec for tokenized output.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=1,
        help="gpu-parquet engine only: parquet compression level; use -1 for codec default.",
    )
    parser.add_argument(
        "--validate-order",
        action="store_true",
        help="gpu-parquet engine only: fail if input rows are not sorted by User/Card.",
    )
    parser.add_argument(
        "--gpu-runtime-env",
        choices=("auto", "pip", "none"),
        default="auto",
        help=(
            "gpu-parquet engine only: use Ray's pip runtime_env, skip it, or "
            "auto-skip when cudf/cupy are already installed in this environment."
        ),
    )
    parser.add_argument(
        "--no-prewarm-actors",
        action="store_true",
        help="gpu-parquet engine only: include GPU actor startup in the timed path.",
    )
    parser.add_argument(
        "--ray-address",
        default=os.environ.get("RAY_ADDRESS", "auto"),
        help="Ray address. Use 'local' to start a local Ray runtime.",
    )
    parser.add_argument("--overwrite", action="store_true")
    if defaults:
        destinations = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - destinations)
        if unknown:
            raise ValueError(f"Profile contains unknown arguments: {unknown}")
        parser.set_defaults(**defaults)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--profile")
    selected, _ = selector.parse_known_args(arguments)
    defaults = None
    if selected.profile:
        try:
            defaults = load_profile(selected.profile)
        except ValueError as exc:
            selector.error(str(exc))
    return build_parser(defaults).parse_args(arguments)


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    split_dir = args.split_dir.expanduser().resolve()
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    missing = [split for split in SPLITS if not (split_dir / split).exists()]
    if missing:
        raise FileNotFoundError(f"Missing split directories under {split_dir}: {missing}")
    if args.actors < 1:
        raise ValueError("--actors must be at least 1")
    if args.num_cpus_per_actor < 1:
        raise ValueError("--num-cpus-per-actor must be at least 1")
    if args.row_groups_per_batch < 1:
        raise ValueError("--row-groups-per-batch must be at least 1")
    if args.engine == "gpu-parquet":
        if args.merchant_hash_mode != "integer_mod":
            raise ValueError(
                "The gpu-parquet engine implements integer_mod merchant mapping; "
                "use --engine legacy for string_hash."
            )
        if args.write_threads_per_actor < 1:
            raise ValueError("--write-threads-per-actor must be at least 1")
        if args.write_threads_per_actor > args.num_cpus_per_actor:
            raise ValueError(
                "--write-threads-per-actor cannot exceed --num-cpus-per-actor"
            )
        if args.output_shard_size_mb < 1:
            raise ValueError("--output-shard-size-mb must be at least 1")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else default_output_dir(split_dir)
    )
    if (
        output_dir == split_dir
        or output_dir in split_dir.parents
        or split_dir in output_dir.parents
    ):
        raise ValueError(
            f"Input and output directories must not overlap: {split_dir}, {output_dir}"
        )
    auxiliary_paths = []
    if args.ray_temp_dir is not None:
        auxiliary_paths.append(("Ray temp", args.ray_temp_dir.expanduser().resolve()))
    if args.ray_spill_dir is not None:
        auxiliary_paths.append(("Ray spill", args.ray_spill_dir.expanduser().resolve()))
    for label, auxiliary in auxiliary_paths:
        for protected in (split_dir, output_dir):
            if (
                auxiliary == protected
                or auxiliary in protected.parents
                or protected in auxiliary.parents
            ):
                raise ValueError(
                    f"{label} path overlaps input/output: {auxiliary}, {protected}"
                )
    return split_dir, output_dir


def main(argv: list[str] | None = None) -> None:
    process_started = time.time()
    args = parse_args(argv)
    split_dir, output_dir = validate_args(args)

    if str(TFM_ROOT) not in sys.path:
        sys.path.insert(0, str(TFM_ROOT))

    os.environ["TFM_SPLIT_DIR"] = str(split_dir)
    os.environ["TFM_TOKENIZED_DIR"] = str(output_dir)

    from src import ray_common as C

    manifest_config = tokenization_config(C, split_dir, args)
    validate_existing_manifest(output_dir, manifest_config, args.overwrite)
    all_outputs_existed = all(
        nonempty_dir(output_dir / split) for split in SPLITS
    )
    ray = init_ray(C, args)
    if not all_outputs_existed:
        validate_cluster_resources(ray, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Split input:    {split_dir}")
    print(f"Tokenized out:  {output_dir}")
    if args.profile:
        print(f"Profile:        {args.profile}")
    print(f"Ray GPUs:       {ray.cluster_resources().get('GPU', 0):.0f}")
    print(f"Ray CPUs:       {ray.cluster_resources().get('CPU', 0):.0f}")
    print(f"SEQ_LENGTH={C.SEQ_LENGTH}, SEQ_CHUNK_SIZE={C.SEQ_CHUNK_SIZE}\n")

    if args.engine == "gpu-parquet":
        seq_counts, elapsed_s = tokenize_gpu_parquet(ray, C, split_dir, output_dir, args)
    else:
        started = time.time()
        seq_counts = {}
        for split in SPLITS:
            seq_counts[split] = tokenize_split(ray, C, split_dir, output_dir, split, args)
        elapsed_s = time.time() - started

    manifest_path = output_dir / TOKENIZATION_MANIFEST
    manifest_elapsed_s = 0.0
    if args.overwrite or not all_outputs_existed or not manifest_path.exists():
        manifest_started = time.time()
        write_tokenization_manifest(output_dir, manifest_config, seq_counts, elapsed_s)
        manifest_elapsed_s = time.time() - manifest_started

    print("\nTokenized outputs for NB03:")
    for split in SPLITS:
        print(f"  {split:5s}: {seq_counts[split]:>10,} sequences -> {output_dir / split}")
    print(f"\nPoint NB03 at this output with:\n  export TFM_TOKENIZED_DIR={output_dir}")
    if args.engine == "legacy":
        timing_label = "Legacy tokenization data-path time"
    elif args.no_prewarm_actors:
        timing_label = "Tokenization data-path time including actor startup"
    else:
        timing_label = "Prewarmed tokenization data-path time"
    print(f"{timing_label}: {elapsed_s:.2f} s")
    if manifest_elapsed_s:
        print(f"Post-timer provenance manifest time: {manifest_elapsed_s:.2f} s")
    print(f"End-to-end process time (including Ray startup): {time.time()-process_started:.2f} s")


if __name__ == "__main__":
    main()
