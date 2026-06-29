#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paired NB05-style fraud evaluation for legacy and fast embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SPLITS = ("train", "val", "test")
FEATURE_COLUMNS = [
    "User",
    "Card",
    "Year",
    "Month",
    "Day",
    "Hour",
    "Amount",
    "Use Chip",
    "Merchant Name",
    "Merchant City",
    "Merchant State",
    "Zip",
    "MCC",
]
XGB_PARAMS = {
    "raw": dict(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.0023,
        colsample_bytree=0.95,
        min_child_weight=12,
        subsample=0.673,
        reg_alpha=0.01,
        reg_lambda=0.001,
        random_state=42,
    ),
    "embed": dict(
        n_estimators=435,
        max_depth=12,
        learning_rate=0.03774,
        colsample_bytree=0.587,
        min_child_weight=2.61,
        subsample=0.569,
        reg_alpha=0.01364,
        reg_lambda=9.7e-5,
        gamma=1.7,
        random_state=42,
    ),
    "combined": dict(
        n_estimators=512,
        max_depth=12,
        learning_rate=0.00305,
        colsample_bytree=0.768,
        min_child_weight=25.85,
        subsample=0.65,
        reg_alpha=0.01,
        reg_lambda=1e-4,
        gamma=4.8,
        random_state=42,
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_sample_dir", type=Path)
    parser.add_argument("embedding_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=20,
        help="Set to 0 to train the fixed notebook estimator budget.",
    )
    parser.add_argument("--num-cpus", type=int, default=64)
    parser.add_argument(
        "--ray-temp-dir", type=Path, default=Path("/dev/shm/tfm-downstream-xgb")
    )
    return parser.parse_args(argv)


def parquet_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.parquet")) or sorted(path.rglob("*.parquet"))


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


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import ray.data  # registers ArrowTensorTypeV2

    embeddings = []
    row_ids = []
    labels = []
    weights = []
    for file in parquet_files(path):
        table = pq.read_table(file)
        tensor = table["embedding"].combine_chunks()
        embeddings.append(tensor.to_numpy_ndarray())
        row_ids.append(table["row_id"].combine_chunks().to_numpy())
        labels.append(table["label"].combine_chunks().to_numpy())
        weights.append(table["sample_weight"].combine_chunks().to_numpy())
    row_id = np.concatenate(row_ids)
    order = np.argsort(row_id, kind="stable")
    return (
        np.concatenate(embeddings)[order],
        np.concatenate(labels)[order].astype(np.int8),
        np.concatenate(weights)[order].astype(np.float64),
        row_id[order].astype(np.int64),
    )


def load_raw(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(parquet_files(path))
    frame = table.to_pandas()
    frame["Hour"] = frame["Time"].str.split(":").str[0].astype(int)
    frame["Amount"] = (
        frame["Amount"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    label = frame["Is Fraud?"].isin(["Yes", "1"]).to_numpy(dtype=np.int8)
    row_id = frame["row_id"].to_numpy(dtype=np.int64)
    weight = frame["sample_weight"].to_numpy(dtype=np.float64)
    user = frame["User"].to_numpy(dtype=np.int64)
    raw = frame[FEATURE_COLUMNS].copy()
    for column in raw.columns:
        if raw[column].dtype == object or str(raw[column].dtype).startswith("string"):
            raw[column] = raw[column].astype(object).where(raw[column].notna(), np.nan)
    order = np.argsort(row_id, kind="stable")
    return raw.iloc[order].reset_index(drop=True), label[order], weight[order], row_id[order], user[order]


def assert_same_rows(
    split: str,
    raw_values: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    arm_values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    _, raw_label, raw_weight, raw_id, _ = raw_values
    if len(raw_id) != len(np.unique(raw_id)):
        raise ValueError(f"Duplicate raw row_id in {split}")
    for arm, (_, label, weight, row_id) in arm_values.items():
        if not np.array_equal(raw_id, row_id):
            raise ValueError(f"Raw/{arm} row_id mismatch in {split}")
        if not np.array_equal(raw_label, label):
            raise ValueError(f"Raw/{arm} label mismatch in {split}")
        if not np.array_equal(raw_weight, weight):
            raise ValueError(f"Raw/{arm} sample_weight mismatch in {split}")


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.percentile(values, [2.5, 50.0, 97.5])]


def paired_user_bootstrap(
    labels: np.ndarray,
    base_weights: np.ndarray,
    users: np.ndarray,
    legacy_score: np.ndarray,
    fast_score: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    unique_users, inverse = np.unique(users, return_inverse=True)
    rng = np.random.default_rng(seed)
    legacy_auc = np.empty(iterations)
    fast_auc = np.empty(iterations)
    legacy_ap = np.empty(iterations)
    fast_ap = np.empty(iterations)
    for index in range(iterations):
        drawn = rng.integers(0, len(unique_users), size=len(unique_users))
        multiplicity = np.bincount(drawn, minlength=len(unique_users))[inverse]
        weight = base_weights * multiplicity
        legacy_auc[index] = roc_auc_score(
            labels, legacy_score, sample_weight=weight
        )
        fast_auc[index] = roc_auc_score(labels, fast_score, sample_weight=weight)
        legacy_ap[index] = average_precision_score(
            labels, legacy_score, sample_weight=weight
        )
        fast_ap[index] = average_precision_score(
            labels, fast_score, sample_weight=weight
        )
    auc_difference = fast_auc - legacy_auc
    ap_difference = fast_ap - legacy_ap
    ap_ratio = fast_ap / legacy_ap
    return {
        "iterations": iterations,
        "clusters": len(unique_users),
        "legacy_auc_95ci": percentile_interval(legacy_auc),
        "fast_auc_95ci": percentile_interval(fast_auc),
        "fast_minus_legacy_auc_95ci": percentile_interval(auc_difference),
        "legacy_ap_95ci": percentile_interval(legacy_ap),
        "fast_ap_95ci": percentile_interval(fast_ap),
        "fast_minus_legacy_ap_95ci": percentile_interval(ap_difference),
        "fast_over_legacy_ap_95ci": percentile_interval(ap_ratio),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_root = args.raw_sample_dir.expanduser().resolve()
    embedding_root = args.embedding_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    for label, protected in (("raw sample", raw_root), ("embeddings", embedding_root)):
        if (
            args.output == protected
            or args.output in protected.parents
            or protected in args.output.parents
        ):
            raise ValueError(f"Output overlaps {label}: {args.output}, {protected}")
    ray_temp = args.ray_temp_dir.expanduser().resolve()
    for label, protected in (
        ("raw sample", raw_root),
        ("embeddings", embedding_root),
        ("output directory", args.output.parent),
    ):
        if (
            ray_temp == protected
            or ray_temp in protected.parents
            or protected in ray_temp.parents
        ):
            raise ValueError(f"Ray temp overlaps {label}: {ray_temp}, {protected}")
    if args.pca_dim < 1 or args.bootstrap < 1:
        raise ValueError("--pca-dim and --bootstrap must be positive")

    raw_metadata = json.loads((raw_root / "metadata.json").read_text())
    embedding_metadata = json.loads((embedding_root / "metadata.json").read_text())
    expected_modes = {"legacy": "string_hash", "fast": "integer_mod"}
    for arm, mode in expected_modes.items():
        recorded_mode = embedding_metadata.get("checkpoints", {}).get(arm, {}).get(
            "merchant_hash_mode"
        )
        if recorded_mode != mode:
            raise ValueError(
                f"Embedding checkpoint mode mismatch for {arm}: {recorded_mode!r}"
            )
    for split in SPLITS:
        files = parquet_files(raw_root / split)
        if len(files) != 1:
            raise ValueError(f"Expected one sampled raw parquet for {split}")
        recorded_sha = raw_metadata.get("samples", {}).get(split, {}).get(
            "output_sha256"
        )
        if recorded_sha != file_sha256(files[0]):
            raise ValueError(f"Raw sample parquet hash mismatch for {split}")
        for arm in expected_modes:
            recorded_files = (
                embedding_metadata.get("output_files", {})
                .get(arm, {})
                .get(split)
            )
            current_files = parquet_manifest(embedding_root / arm / split)
            if recorded_files != current_files:
                raise ValueError(f"Embedding parquet manifest mismatch for {arm}/{split}")

    raw = {split: load_raw(raw_root / split) for split in SPLITS}
    embeddings = {
        arm: {
            split: load_embeddings(embedding_root / arm / split)
            for split in SPLITS
        }
        for arm in ("legacy", "fast")
    }
    for split in SPLITS:
        assert_same_rows(
            split,
            raw[split],
            {arm: embeddings[arm][split] for arm in ("legacy", "fast")},
        )

    from sklearn.compose import make_column_selector, make_column_transformer
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import OrdinalEncoder

    encoder = make_column_transformer(
        (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            make_column_selector(dtype_include=["object", "category"]),
        ),
        remainder="passthrough",
    )
    raw_features = {
        "train": encoder.fit_transform(raw["train"][0]),
        "val": encoder.transform(raw["val"][0]),
        "test": encoder.transform(raw["test"][0]),
    }
    raw_features = {
        split: np.asarray(values, dtype=np.float32, order="C")
        for split, values in raw_features.items()
    }

    pca_features = {}
    pca_summary = {}
    for arm in ("legacy", "fast"):
        pca = PCA(
            n_components=args.pca_dim,
            svd_solver="randomized",
            random_state=args.seed,
        )
        pca_features[arm] = {
            "train": np.asarray(
                pca.fit_transform(embeddings[arm]["train"][0]),
                dtype=np.float32,
                order="C",
            )
        }
        for split in ("val", "test"):
            pca_features[arm][split] = np.asarray(
                pca.transform(embeddings[arm][split][0]),
                dtype=np.float32,
                order="C",
            )
        pca_summary[arm] = {
            "explained_variance": float(pca.explained_variance_ratio_.sum())
        }
        print(
            f"[{arm}] PCA explained variance "
            f"{pca_summary[arm]['explained_variance']:.2%}",
            flush=True,
        )

    datasets = {
        "raw": tuple(raw_features[split] for split in SPLITS),
    }
    for arm in ("legacy", "fast"):
        datasets[f"{arm}_embed"] = tuple(
            pca_features[arm][split] for split in SPLITS
        )
        datasets[f"{arm}_combined"] = tuple(
            np.ascontiguousarray(
                np.hstack([raw_features[split], pca_features[arm][split]]),
                dtype=np.float32,
            )
            for split in SPLITS
        )

    import ray

    shutil.rmtree(ray_temp, ignore_errors=True)
    (ray_temp / "spill").mkdir(parents=True, exist_ok=True)
    ray.init(
        num_cpus=args.num_cpus,
        num_gpus=len(datasets),
        include_dashboard=False,
        _temp_dir=str(ray_temp),
        _system_config={
            "object_spilling_config": json.dumps(
                {
                    "type": "filesystem",
                    "params": {"directory_path": str(ray_temp / "spill")},
                }
            )
        },
    )

    @ray.remote(num_gpus=1, num_cpus=8)
    def train_model(
        name,
        matrices,
        labels,
        val_weight,
        test_weight,
        params,
        early_stopping_rounds,
    ):
        import time

        import xgboost as xgb
        from sklearn.metrics import average_precision_score, roc_auc_score

        train_x, val_x, test_x = matrices
        train_y, val_y, test_y = labels
        started = time.perf_counter()
        model_kwargs = {
            **params,
            "tree_method": "hist",
            "device": "cuda",
            "eval_metric": "auc",
        }
        if early_stopping_rounds:
            model_kwargs["early_stopping_rounds"] = early_stopping_rounds
        model = xgb.XGBClassifier(**model_kwargs)
        model.fit(
            train_x,
            train_y,
            eval_set=[(val_x, val_y)],
            sample_weight_eval_set=[val_weight],
            verbose=False,
        )
        val_score = model.predict_proba(val_x)[:, 1]
        test_score = model.predict_proba(test_x)[:, 1]
        return {
            "name": name,
            "features": train_x.shape[1],
            "best_iteration": int(
                getattr(model, "best_iteration", params["n_estimators"] - 1)
            ),
            "train_seconds": time.perf_counter() - started,
            "val_auc": float(
                roc_auc_score(val_y, val_score, sample_weight=val_weight)
            ),
            "val_ap": float(
                average_precision_score(val_y, val_score, sample_weight=val_weight)
            ),
            "test_auc": float(
                roc_auc_score(test_y, test_score, sample_weight=test_weight)
            ),
            "test_ap": float(
                average_precision_score(test_y, test_score, sample_weight=test_weight)
            ),
            "test_score": test_score,
            "gpu": __import__("torch").cuda.get_device_name(),
        }

    labels = tuple(raw[split][1] for split in SPLITS)
    val_weight = raw["val"][2]
    test_weight = raw["test"][2]
    pending = {}
    for name, matrices in datasets.items():
        family = "raw" if name == "raw" else name.split("_", 1)[1]
        pending[name] = train_model.remote(
            name,
            matrices,
            labels,
            val_weight,
            test_weight,
            XGB_PARAMS[family],
            args.early_stopping_rounds,
        )
    try:
        trained = {name: ray.get(task) for name, task in pending.items()}
    finally:
        ray.shutdown()

    scores = {name: result.pop("test_score") for name, result in trained.items()}
    bootstrap = paired_user_bootstrap(
        raw["test"][1],
        raw["test"][2],
        raw["test"][4],
        scores["legacy_combined"],
        scores["fast_combined"],
        args.bootstrap,
        args.seed,
    )
    comparison = {
        "combined_fast_minus_legacy_auc": (
            trained["fast_combined"]["test_auc"]
            - trained["legacy_combined"]["test_auc"]
        ),
        "combined_fast_minus_legacy_ap": (
            trained["fast_combined"]["test_ap"]
            - trained["legacy_combined"]["test_ap"]
        ),
        "combined_fast_over_legacy_ap": (
            trained["fast_combined"]["test_ap"]
            / trained["legacy_combined"]["test_ap"]
        ),
    }
    comparison["point_noninferiority_pass"] = bool(
        comparison["combined_fast_minus_legacy_auc"] >= -0.005
        and comparison["combined_fast_over_legacy_ap"] >= 0.90
    )
    comparison["bootstrap_lower_bound_pass"] = bool(
        bootstrap["fast_minus_legacy_auc_95ci"][0] >= -0.005
        and bootstrap["fast_over_legacy_ap_95ci"][0] >= 0.90
    )

    payload = {
        "seed": args.seed,
        "sample_counts": {
            split: {
                "rows": len(raw[split][1]),
                "fraud": int(raw[split][1].sum()),
            }
            for split in SPLITS
        },
        "pca_dim": args.pca_dim,
        "early_stopping_rounds": args.early_stopping_rounds,
        "pca": pca_summary,
        "xgb_params": XGB_PARAMS,
        "metrics": trained,
        "comparison": comparison,
        "paired_user_bootstrap": bootstrap,
        "provenance": {
            "raw_sample_dir": str(raw_root),
            "raw_metadata_sha256": file_sha256(raw_root / "metadata.json"),
            "embedding_dir": str(embedding_root),
            "embedding_metadata_sha256": file_sha256(
                embedding_root / "metadata.json"
            ),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "runtime": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
                "ray": ray.__version__,
                "sklearn": __import__("sklearn").__version__,
                "xgboost": __import__("xgboost").__version__,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
