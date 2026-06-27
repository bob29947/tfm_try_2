# SPDX-License-Identifier: Apache-2.0
"""
Self-contained synthetic generator for the IBM TabFormer credit-card schema.

The real blueprint downloads ``card_transaction.v1.csv`` (~2.2 GB, ~24M rows)
from IBM Box. That file is impractical to process on a CPU head node and is not
reproducible. This module emits a Parquet dataset with the **exact same column
names, dtypes, string formats and ~0.1% fraud rate** so every Ray notebook runs
end-to-end anywhere, deterministically.

To use the REAL dataset instead, set ``TFM_REAL_CSV=/path/to/card_transaction.v1.csv``
and call :func:`materialize_real` — the downstream notebooks accept either source
because both produce identical schemas.

Columns (TabFormer): User, Card, Year, Month, Day, Time, Amount, Use Chip,
Merchant Name, Merchant City, Merchant State, Zip, MCC, Errors?, Is Fraud?

Fraud is injected with learnable structure (large/round amounts, online use,
odd hours, a few risky MCCs, bursty velocity) so both the XGBoost baseline and
the foundation-model embeddings have real signal to capture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Realistic-ish supporting vocabularies (subset of the TabFormer value space).
_MCCS = [
    5411, 5412, 5499, 5311, 5611, 5651, 5732, 5812, 5813, 5814, 5912, 5921,
    5941, 5942, 4111, 4121, 4814, 4829, 4900, 5541, 5542, 7011, 7230, 7995,
    7996, 8011, 8021, 8062, 3000, 3001, 3501, 3502, 6300, 5045, 5094,
]
# MCCs with elevated fraud propensity (e-commerce / cash-like).
_RISKY_MCCS = {7995, 5912, 4829, 6300, 5094}
_STATES = [
    "CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
]
_CITIES = [
    "San Jose", "New York", "Houston", "Miami", "Chicago", "Philadelphia",
    "Columbus", "Atlanta", "Charlotte", "Detroit", "Newark", "Seattle",
]
_CHIP = ["Swipe Transaction", "Chip Transaction", "Online Transaction"]


def generate(
    n_users: int = 2000,
    avg_txns_per_user: int = 750,
    start_year: int = 2015,
    end_year: int = 2019,
    fraud_rate: float = 0.0012,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a TabFormer-schema transaction DataFrame.

    Default ~1.5M rows across 2,000 users — large enough to exercise GPU
    tokenization / training, small enough to run in minutes.
    """
    rng = np.random.default_rng(seed)
    n_rows = n_users * avg_txns_per_user

    # Per-user attributes.
    user_ids = rng.integers(0, n_users, size=n_rows)
    card_ids = rng.integers(0, 5, size=n_rows)

    # Timestamps: spread across the year range, then we sort by (user, card, t).
    span_days = (end_year - start_year + 1) * 365
    day_offset = rng.integers(0, span_days, size=n_rows)
    base = np.datetime64(f"{start_year}-01-01")
    dates = base + day_offset.astype("timedelta64[D]")
    dates = dates + (rng.integers(0, 86400, size=n_rows)).astype("timedelta64[s]")
    dts = pd.to_datetime(dates)

    # Amounts: log-normal, occasional large round values.
    amounts = np.round(rng.lognormal(mean=3.2, sigma=1.1, size=n_rows), 2)
    big = rng.random(n_rows) < 0.02
    amounts[big] = rng.choice([500, 1000, 2000, 5000], size=big.sum())

    mcc = rng.choice(_MCCS, size=n_rows)
    chip = rng.choice(_CHIP, size=n_rows, p=[0.55, 0.35, 0.10])
    online = chip == "Online Transaction"

    states = np.array(_STATES)[rng.integers(0, len(_STATES), size=n_rows)]
    cities = np.array(_CITIES)[rng.integers(0, len(_CITIES), size=n_rows)]
    zip_nums = rng.integers(10000, 99999, size=n_rows)
    # Match the real CSV: Zip is a float-formatted STRING ("95113.0"); online
    # transactions have no physical location (empty string).
    zips = np.array([f"{z}.0" for z in zip_nums], dtype=object)
    # Online transactions have no physical location.
    states = np.where(online, "", states)
    cities = np.where(online, "ONLINE", cities)
    zips = np.where(online, "", zips)

    merchant_name = rng.integers(10**14, 10**15, size=n_rows).astype(str)

    hours = dts.hour.to_numpy()

    # ---- Fraud signal: combine several weak, learnable factors ------------
    score = (
        0.9 * (amounts > 500).astype(float)
        + 0.8 * online.astype(float)
        + 0.6 * np.isin(mcc, list(_RISKY_MCCS)).astype(float)
        + 0.5 * ((hours < 5) | (hours >= 23)).astype(float)
        + rng.normal(0, 0.5, size=n_rows)
    )
    # Pick a threshold that yields ~fraud_rate positives.
    thresh = np.quantile(score, 1.0 - fraud_rate)
    is_fraud = score >= thresh

    df = pd.DataFrame(
        {
            "User": user_ids,
            "Card": card_ids,
            "Year": dts.year.to_numpy(),
            "Month": dts.month.to_numpy(),
            "Day": dts.day.to_numpy(),
            "Time": dts.strftime("%H:%M").to_numpy(),
            "Amount": ["$" + format(a, ".2f") for a in amounts],
            "Use Chip": chip,
            "Merchant Name": merchant_name,
            "Merchant City": cities,
            "Merchant State": states,
            "Zip": zips,
            "MCC": mcc,
            "Errors?": "",
            "Is Fraud?": np.where(is_fraud, "Yes", "No"),
            # Helper ordering key (dropped before save to match real schema).
            "_ts": dts,
        }
    )
    # Order chronologically within each (user, card) — what the real data is.
    df = df.sort_values(["User", "Card", "_ts"]).drop(columns="_ts").reset_index(drop=True)
    return df


def materialize_synthetic(out_path, **kwargs) -> int:
    """Generate synthetic data and write a single Parquet file. Returns row count."""
    df = generate(**kwargs)
    df.to_parquet(out_path, index=False)
    return len(df)


def materialize_real(csv_path, out_path) -> int:
    """Convert the real ``card_transaction.v1.csv`` to Parquet (schema is identical)."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df.to_parquet(out_path, index=False)
    return len(df)
