# SPDX-License-Identifier: Apache-2.0
"""
Ray Data tokenization building blocks (run on GPU workers).

Module-level imports are head-safe (numpy only); cuDF and the cuDF-based
tokenizer are imported lazily inside the actor so this module can be referenced
from the CPU head and shipped to workers via `py_modules`.
"""

from __future__ import annotations

import numpy as np

from . import ray_common as C


class GPUTokenizer:
    """Stateful Ray Data actor: cuDF financial tokenizer over a batch of raw
    transactions. Emits, per transaction, the (data-independent) field token
    IDs plus grouping/order keys and the fraud label.

    Returned arrays are plain numpy so the CPU head can handle them.
    """

    def __init__(self, merchant_hash_size: int = C.MERCHANT_HASH_SIZE, carry_cols=None):
        import cudf  # lazy: worker-only
        from src.tokenizer import FinancialTokenizerPipeline

        self._cudf = cudf
        self.pipeline = FinancialTokenizerPipeline(
            merchant_hash_size=merchant_hash_size, use_streams=False,
        )
        self._fitted = False
        # Optional raw columns to pass through, aligned with each tokenized row
        # (used by NB04 so the embeddings table also holds raw features for the
        # NB05 "combined" model). Names are the original TabFormer column names.
        self.carry_cols = list(carry_cols) if carry_cols else []

    @staticmethod
    def _fraud_label(proc, n):
        for col in ("is_fraud?", "is_fraud", "fraud"):
            if col in proc.columns:
                s = proc[col].astype(str).to_pandas()
                return ((s == "Yes") | (s == "1")).astype("int64").to_numpy()
        return np.zeros(n, dtype="int64")

    def __call__(self, batch):
        cudf = self._cudf
        # Ray Data hands us a cudf.DataFrame directly via `batch_format="cudf"`
        # (the Arrow block is moved to GPU with `cudf.DataFrame.from_arrow`), so
        # the whole tokenizer runs on the worker GPU with zero host round-trips.
        # Fall back to constructing one if called with a pandas/dict batch.
        gdf = batch if isinstance(batch, cudf.DataFrame) else cudf.DataFrame(batch)
        proc = self.pipeline.preprocess(gdf)          # also sorts by user/card/time
        if not self._fitted:
            # vocab is data-independent (fixed bins/hash/ranges) -> consistent IDs.
            self.pipeline.fit(proc)
            self._fitted = True
        token_df = self.pipeline.transform(proc)
        vocab = self.pipeline.vocab
        cols = list(token_df.columns)
        id_cols = [
            token_df[c].to_pandas().map(vocab).fillna(C.UNK_TOKEN_ID).astype("int64").to_numpy()
            for c in cols
        ]
        token_ids = np.stack(id_cols, axis=1)          # (batch, n_fields)
        user = proc["user"].astype("int64").to_pandas().to_numpy()
        card = proc["card"].astype("int64").to_pandas().to_numpy()
        ts = (proc["time_full"].astype("int64") // 10**9).to_pandas().to_numpy()  # epoch s
        out = {
            "uc_key": user * 100 + card,               # unique (user, card) group key
            "ts": ts,
            "token_ids": token_ids,
            "label": self._fraud_label(proc, len(token_ids)),
        }
        for col in self.carry_cols:                    # raw features, row-aligned
            lc = col.strip().replace(" ", "_").lower()
            out[col] = proc[lc].to_pandas().to_numpy()
        return out


def build_sequences(group, seq_length: int = C.SEQ_LENGTH, chunk_size: int = C.SEQ_CHUNK_SIZE):
    """Ray Data `map_groups` fn: turn one (user, card)'s transactions into
    fixed-length causal-LM sequences:  <bos> txn1 <sep> txn2 ... <eos> <pad>...
    """
    ts, tok = group["ts"], group["token_ids"]
    order = np.argsort(ts, kind="stable")
    tok = tok[order]
    seqs = []
    for start in range(0, len(tok), chunk_size):
        chunk = tok[start:start + chunk_size]          # (m, n_fields)
        seq = [C.BOS_TOKEN_ID]
        for i, row in enumerate(chunk):
            seq.extend(int(x) for x in row)
            if i < len(chunk) - 1:
                seq.append(C.SEP_TOKEN_ID)
        seq.append(C.EOS_TOKEN_ID)
        seq = seq[:seq_length]
        arr = np.full(seq_length, C.PAD_TOKEN_ID, dtype="int64")
        arr[: len(seq)] = seq
        seqs.append(arr)
    if not seqs:
        return {"input_ids": np.zeros((0, seq_length), dtype="int64")}
    return {"input_ids": np.stack(seqs, axis=0)}
