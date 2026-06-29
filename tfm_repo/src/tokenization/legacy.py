# SPDX-License-Identifier: Apache-2.0
"""Legacy Ray Data tokenization building blocks."""

from __future__ import annotations

import numpy as np

from . import contract as C


class GPUTokenizer:
    """Stateful cuDF tokenizer used by Ray Data ``map_batches``.

    The returned arrays are NumPy arrays so the CPU driver can consume them.
    GPU dependencies remain worker-only imports.
    """

    def __init__(
        self,
        merchant_hash_size: int = C.MERCHANT_HASH_SIZE,
        carry_cols=None,
        merchant_hash_mode: str = C.MERCHANT_HASH_MODE,
    ):
        import cudf  # lazy: worker-only
        from src.tokenizer import FinancialTokenizerPipeline

        self._cudf = cudf
        self.pipeline = FinancialTokenizerPipeline(
            merchant_hash_size=merchant_hash_size, use_streams=False,
        )
        self.merchant_hash_mode = merchant_hash_mode
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
        # Ray Data normally supplies a cudf.DataFrame. Keep the fallback for
        # direct calls with pandas frames or dictionaries.
        gdf = batch if isinstance(batch, cudf.DataFrame) else cudf.DataFrame(batch)
        proc = self.pipeline.preprocess(
            gdf,
            merchant_hash_mode=self.merchant_hash_mode,
        )
        if not self._fitted:
            # The vocabulary is data-independent (fixed bins/hash/ranges).
            self.pipeline.fit(proc)
            self._fitted = True
        token_df = self.pipeline.transform(proc)
        vocab = self.pipeline.vocab
        cols = list(token_df.columns)
        id_cols = [
            token_df[c]
            .to_pandas()
            .map(vocab)
            .fillna(C.UNK_TOKEN_ID)
            .astype("int64")
            .to_numpy()
            for c in cols
        ]
        token_ids = np.stack(id_cols, axis=1)
        user = proc["user"].astype("int64").to_pandas().to_numpy()
        card = proc["card"].astype("int64").to_pandas().to_numpy()
        ts = (
            (proc["time_full"].astype("int64") // 10**9)
            .to_pandas()
            .to_numpy()
        )
        out = {
            "uc_key": user * 100 + card,
            "ts": ts,
            "token_ids": token_ids,
            "label": self._fraud_label(proc, len(token_ids)),
        }
        for col in self.carry_cols:
            lc = col.strip().replace(" ", "_").lower()
            out[col] = proc[lc].to_pandas().to_numpy()
        return out


def build_sequences(
    group,
    seq_length: int = C.SEQ_LENGTH,
    chunk_size: int = C.SEQ_CHUNK_SIZE,
):
    """Build fixed-length causal-LM sequences for one user/card group."""
    ts, tok = group["ts"], group["token_ids"]
    order = np.argsort(ts, kind="stable")
    tok = tok[order]
    seqs = []
    for start in range(0, len(tok), chunk_size):
        chunk = tok[start : start + chunk_size]
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


__all__ = ["GPUTokenizer", "build_sequences"]
