# SPDX-License-Identifier: Apache-2.0
"""
Ray Data tokenization building blocks (run on GPU workers).

Module-level imports are head-safe (numpy only); cuDF and the cuDF-based
tokenizer are imported lazily inside the actor so this module can be referenced
from the CPU head and shipped to workers via `py_modules`.
"""

from __future__ import annotations

import time

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


class FastParquetSplitTokenizer:
    """GPU actor for temporal split parquet.

    This bypasses Ray Data's groupby shuffle for the common TabFormer split
    layout produced by ``create_temporal_splits.py``.  The actor owns one
    contiguous user range, reads only overlapping parquet row groups, computes
    final token IDs directly on GPU, sorts by ``User/Card/time`` on GPU, creates
    fixed-width sequence tensors on GPU, and writes one parquet shard per split.
    """

    COLUMNS = [
        "User",
        "Card",
        "Year",
        "Month",
        "Day",
        "Time",
        "Amount",
        "Use Chip",
        "Merchant Name",
        "Merchant State",
        "Zip",
        "MCC",
    ]

    def __init__(
        self,
        merchant_hash_size: int = C.MERCHANT_HASH_SIZE,
        seq_length: int = C.SEQ_LENGTH,
        chunk_size: int = C.SEQ_CHUNK_SIZE,
        output_dtype: str = "int32",
        output_format: str = "binary-tensor",
        compression: str = "zstd",
        compression_level: int | None = 1,
        use_dictionary: bool = False,
        row_groups_per_batch: int = 4,
        arrow_cpu_threads: int | None = None,
        validate_order: bool = False,
    ):
        import cudf  # lazy: worker-only
        import cupy as cp
        import pyarrow as pa
        import pyarrow.parquet as pq
        from ray.data.extensions.tensor_extension import ArrowTensorArray
        from src.tokenizer.financial_pipeline import (
            ALL_STATES,
            CHIP_MAPPING,
            INDUSTRY_RANGES,
            KNOWN_MCCS,
        )

        if arrow_cpu_threads:
            pa.set_cpu_count(int(arrow_cpu_threads))

        self.cudf = cudf
        self.cp = cp
        self.pa = pa
        self.pq = pq
        self.ArrowTensorArray = ArrowTensorArray

        self.merchant_hash_size = int(merchant_hash_size)
        self.seq_length = int(seq_length)
        self.chunk_size = int(chunk_size)
        self.n_fields = 12
        self.output_np_dtype = np.dtype(output_dtype)
        self.output_cp_dtype = cp.dtype(output_dtype)
        self.output_format = output_format
        self.compression = compression
        self.compression_level = compression_level
        self.use_dictionary = bool(use_dictionary)
        self.row_groups_per_batch = max(1, int(row_groups_per_batch))
        self.validate_order = bool(validate_order)

        self.industry_ranges = list(INDUSTRY_RANGES)
        self.chip_mapping = dict(CHIP_MAPPING)

        cat_labels = sorted({label for _, _, label in INDUSTRY_RANGES})
        cat_labels.append("GENERAL")
        self.cat_idx = {label: idx for idx, label in enumerate(dict.fromkeys(cat_labels))}
        self.cat_default_idx = self.cat_idx["GENERAL"]

        mcc_labels = sorted({str(mcc) for mcc in KNOWN_MCCS})
        mcc_labels.append("-1")
        self.mcc_idx = {int(label): idx for idx, label in enumerate(dict.fromkeys(mcc_labels))}
        self.mcc_default_idx = self.mcc_idx[-1]

        chip_labels = sorted(set(CHIP_MAPPING.values()))
        chip_labels.append("UNK")
        self.chip_idx = {label: idx for idx, label in enumerate(dict.fromkeys(chip_labels))}
        self.chip_default_idx = self.chip_idx["UNK"]
        self.chip_raw_idx = {
            raw: self.chip_idx.get(label, self.chip_default_idx)
            for raw, label in CHIP_MAPPING.items()
        }

        state_labels = sorted(set(ALL_STATES))
        state_labels.append("XX")
        self.state_idx = {label: idx for idx, label in enumerate(dict.fromkeys(state_labels))}
        self.state_default_idx = self.state_idx["XX"]

        offset = C.UNK_TOKEN_ID + 1
        self.offset_amt = offset
        offset += 7
        self.offset_merch = offset
        offset += self.merchant_hash_size
        self.offset_cat = offset
        offset += len(self.cat_idx)
        self.offset_mcc = offset
        offset += len(self.mcc_idx)
        self.offset_hour = offset
        offset += 24
        self.offset_dow = offset
        offset += 7
        self.offset_month = offset
        offset += 12
        self.offset_card = offset
        offset += 10
        self.offset_chip = offset
        offset += len(self.chip_idx)
        self.offset_zip3 = offset
        offset += 1000
        self.offset_state = offset
        offset += len(self.state_idx)
        self.offset_cust = offset
        offset += 3000
        self.vocab_size = offset

        self._carry_key = None
        self._carry_tokens = None
        self._prev_key = None

    def tokenize(self, work_items: list[dict]) -> list[dict]:
        stats = []
        for work in work_items:
            stats.append(self._tokenize_one_split(work))
        return stats

    def __call__(self, work_items: list[dict]) -> list[dict]:
        return self.tokenize(work_items)

    def _tokenize_one_split(self, work: dict) -> dict:
        output_path = work["output_path"]
        writer = None
        count = 0
        rows = 0
        read_s = 0.0
        tokenize_s = 0.0
        sort_s = 0.0
        sequence_s = 0.0
        write_s = 0.0
        started = time.perf_counter()
        self._prev_key = None
        key_batches = []
        time_batches = []
        token_batches = []

        try:
            for fragment in work["fragments"]:
                row_groups = list(fragment["row_groups"])
                for start in range(0, len(row_groups), self.row_groups_per_batch):
                    batch_row_groups = row_groups[start:start + self.row_groups_per_batch]
                    op_started = time.perf_counter()
                    gdf = self._read_row_groups(fragment["path"], batch_row_groups)
                    read_s += time.perf_counter() - op_started
                    if len(gdf) == 0:
                        continue
                    gdf = self._filter_user_range(gdf, work["user_min"], work["user_max"])
                    if len(gdf) == 0:
                        continue
                    rows += len(gdf)

                    op_started = time.perf_counter()
                    keys, txn_order, token_ids = self._tokenize_frame(gdf)
                    tokenize_s += time.perf_counter() - op_started
                    key_batches.append(keys)
                    time_batches.append(txn_order)
                    token_batches.append(token_ids)

                    del gdf, keys, txn_order, token_ids

            if token_batches:
                op_started = time.perf_counter()
                keys = self.cp.concatenate(key_batches)
                txn_order = self.cp.concatenate(time_batches)
                token_ids = self.cp.concatenate(token_batches, axis=0)
                order = self.cp.lexsort(self.cp.stack([txn_order, keys]))
                keys = keys[order]
                token_ids = token_ids[order]
                sort_s += time.perf_counter() - op_started

                op_started = time.perf_counter()
                seqs = self._build_sequences_gpu(keys, token_ids)
                sequence_s += time.perf_counter() - op_started
                op_started = time.perf_counter()
                writer = self._write_sequences(writer, output_path, seqs)
                write_s += time.perf_counter() - op_started
                count += len(seqs)

                del keys, txn_order, token_ids, order, seqs
        finally:
            if writer is not None:
                writer.close()
            self._prev_key = None
            key_batches.clear()
            time_batches.clear()
            token_batches.clear()
            try:
                self.cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

        return {
            "split": work["split"],
            "count": count,
            "rows": rows,
            "elapsed_s": time.perf_counter() - started,
            "read_s": read_s,
            "tokenize_s": tokenize_s,
            "sort_s": sort_s,
            "sequence_s": sequence_s,
            "write_s": write_s,
            "output_path": output_path,
        }

    def _read_row_groups(self, path: str, row_groups: list[int]):
        cudf = self.cudf
        try:
            return cudf.read_parquet(path, columns=self.COLUMNS, row_groups=row_groups)
        except TypeError:
            frames = [
                cudf.read_parquet(path, columns=self.COLUMNS, row_groups=[row_group])
                for row_group in row_groups
            ]
            if not frames:
                return cudf.DataFrame()
            return cudf.concat(frames, ignore_index=True)

    def _filter_user_range(self, gdf, user_min: int, user_max: int):
        gdf.columns = [c.strip().replace(" ", "_").lower() for c in gdf.columns]
        user = gdf["user"]
        return gdf[(user >= user_min) & (user <= user_max)]

    def _tokenize_frame(self, gdf):
        cp = self.cp
        n = len(gdf)
        token_ids = cp.empty((n, self.n_fields), dtype=self.output_cp_dtype)

        user = gdf["user"].astype("int64")
        card = gdf["card"].astype("int32").clip(0, 9)
        user_cp = user.to_cupy()
        card_cp = card.to_cupy()
        keys = user_cp * 100 + card_cp
        if self.validate_order:
            self._validate_keys(keys)

        amt = gdf["amount"].str.slice(1).astype("float32")
        amt_val = (
            (amt >= 10).astype("int32")
            + (amt >= 50).astype("int32")
            + (amt >= 100).astype("int32")
            + (amt >= 500).astype("int32")
            + (amt >= 1000).astype("int32")
            + (amt >= 5000).astype("int32")
        )
        token_ids[:, 0] = self.offset_amt + amt_val.to_cupy()

        merch = gdf["merchant_name"].fillna(0).astype("int64")
        token_ids[:, 1] = self.offset_merch + (
            merch.abs() % self.merchant_hash_size
        ).astype("int32").to_cupy()

        mcc = gdf["mcc"].fillna(-1).astype("int64")
        mcc_cp = mcc.to_cupy()

        cat_idx = cp.full(n, self.cat_default_idx, dtype=cp.int32)
        for lo, hi, label in self.industry_ranges:
            idx = self.cat_idx[label]
            cat_idx = cp.where((mcc_cp >= lo) & (mcc_cp <= hi), idx, cat_idx)
        token_ids[:, 2] = self.offset_cat + cat_idx

        mcc_idx = mcc.map(self.mcc_idx).fillna(self.mcc_default_idx).astype("int32")
        token_ids[:, 3] = self.offset_mcc + mcc_idx.to_cupy()

        time_col = gdf["time"].fillna("00:00").astype(str)
        hour = time_col.str.slice(0, 2).astype("int32").clip(0, 23)
        minute = time_col.str.slice(3, 5).astype("int32").clip(0, 59)
        hour_cp = hour.to_cupy()
        token_ids[:, 4] = self.offset_hour + hour_cp

        year_cp = gdf["year"].astype("int32").to_cupy()
        month = gdf["month"].astype("int32").clip(1, 12)
        month_cp = month.to_cupy()
        day_cp = gdf["day"].astype("int32").clip(1, 31).to_cupy()
        # Sakamoto's Gregorian day-of-week algorithm.  It returns Sunday=0;
        # pandas/cuDF ``dt.dayofweek`` uses Monday=0.
        month_offsets = cp.asarray([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=cp.int32)
        y = year_cp - (month_cp < 3)
        dow = (y + y // 4 - y // 100 + y // 400 + month_offsets[month_cp - 1] + day_cp) % 7
        token_ids[:, 5] = self.offset_dow + ((dow + 6) % 7)

        # Match FixedVocabTokenizer's legacy global-ID layout: MONTH local IDs
        # are one-based because the configured range is [1, 12].
        token_ids[:, 6] = self.offset_month + month_cp
        token_ids[:, 7] = self.offset_card + card_cp

        chip = gdf["use_chip"].fillna("").astype(str).str.upper()
        chip_idx = chip.map(self.chip_raw_idx).fillna(self.chip_default_idx).astype("int32")
        token_ids[:, 8] = self.offset_chip + chip_idx.to_cupy()

        zip_code = gdf["zip"].fillna(0).astype("int64").clip(0, 99999).to_cupy()
        zip3 = cp.where(
            zip_code >= 10000,
            zip_code // 100,
            cp.where(zip_code >= 1000, zip_code // 10, zip_code),
        )
        token_ids[:, 9] = self.offset_zip3 + zip3.astype(self.output_cp_dtype, copy=False)

        state = gdf["merchant_state"].fillna("XX").astype(str).str.upper().str.strip()
        state = state.where(state != "", "XX")
        state_idx = state.map(self.state_idx).fillna(self.state_default_idx).astype("int32")
        token_ids[:, 10] = self.offset_state + state_idx.to_cupy()

        cust = user.astype("int32").clip(0, 2999)
        token_ids[:, 11] = self.offset_cust + cust.to_cupy()

        txn_order = (
            ((((year_cp.astype(cp.int64) * 13 + month_cp) * 32 + day_cp) * 24 + hour_cp)
            * 60)
            + minute.to_cupy()
        )

        return keys, txn_order, token_ids

    def _validate_keys(self, keys) -> None:
        cp = self.cp
        if len(keys) == 0:
            return
        first = int(keys[0].get())
        last = int(keys[-1].get())
        if self._prev_key is not None and first < self._prev_key:
            raise ValueError(
                "Input split is not sorted by User/Card across parquet row groups; "
                "use --engine legacy or regenerate the temporal split."
            )
        if len(keys) > 1 and bool(cp.any(keys[1:] < keys[:-1]).get()):
            raise ValueError(
                "Input split is not sorted by User/Card within a parquet row group; "
                "use --engine legacy or regenerate the temporal split."
            )
        self._prev_key = last

    def _build_sequences_gpu(self, keys, token_ids):
        cp = self.cp
        n = len(token_ids)
        if n == 0:
            return np.zeros((0, self.seq_length), dtype=self.output_np_dtype)

        starts_flag = cp.empty(n, dtype=cp.bool_)
        starts_flag[0] = True
        starts_flag[1:] = keys[1:] != keys[:-1]
        group_starts = cp.nonzero(starts_flag)[0]
        group_ends = cp.concatenate([
            group_starts[1:],
            cp.asarray([n], dtype=group_starts.dtype),
        ])
        group_lengths = group_ends - group_starts
        chunks_per_group = (group_lengths + self.chunk_size - 1) // self.chunk_size
        chunk_offsets = cp.empty_like(chunks_per_group)
        if len(chunks_per_group) == 1:
            chunk_offsets[0] = 0
        else:
            chunk_offsets[0] = 0
            chunk_offsets[1:] = cp.cumsum(chunks_per_group[:-1])
        nseq = int(cp.sum(chunks_per_group).get())

        out = cp.full((nseq, self.seq_length), C.PAD_TOKEN_ID, dtype=self.output_cp_dtype)
        out[:, 0] = C.BOS_TOKEN_ID

        group_ids = cp.cumsum(starts_flag.astype(cp.int32)) - 1
        row_pos = cp.arange(n, dtype=cp.int64) - group_starts[group_ids]
        seq_idx = chunk_offsets[group_ids] + row_pos // self.chunk_size
        txn_pos = row_pos % self.chunk_size
        base_pos = 1 + txn_pos * (self.n_fields + 1)

        field_offsets = cp.arange(self.n_fields, dtype=cp.int64)
        flat_pos = seq_idx[:, None] * self.seq_length + base_pos[:, None] + field_offsets[None, :]
        out.reshape(-1)[flat_pos.reshape(-1)] = token_ids.reshape(-1)

        sep_pos = base_pos + self.n_fields
        out.reshape(-1)[seq_idx * self.seq_length + sep_pos] = C.SEP_TOKEN_ID
        last_in_sequence = (txn_pos == self.chunk_size - 1) | cp.concatenate([
            keys[1:] != keys[:-1],
            cp.asarray([True], dtype=cp.bool_),
        ])
        eos_pos = (
            seq_idx[last_in_sequence] * self.seq_length
            + sep_pos[last_in_sequence]
        )
        out.reshape(-1)[eos_pos] = C.EOS_TOKEN_ID

        return out.get()

    def _write_sequences(self, writer, output_path: str, seqs: np.ndarray):
        if self.output_format == "binary-tensor":
            seqs = np.ascontiguousarray(seqs)
            byte_width = seqs.shape[1] * seqs.dtype.itemsize
            storage = self.pa.Array.from_buffers(
                self.pa.binary(byte_width),
                len(seqs),
                [None, self.pa.py_buffer(seqs.view("uint8"))],
            )
            field = self.pa.field(
                "input_ids",
                storage.type,
                metadata={
                    b"ray.data.fixed_size_binary_tensor.shape": f"[{self.seq_length}]".encode(),
                    b"ray.data.fixed_size_binary_tensor.dtype": str(seqs.dtype).encode(),
                },
            )
            table = self.pa.Table.from_arrays([storage], schema=self.pa.schema([field]))
        else:
            table = self.pa.Table.from_arrays(
                [self.ArrowTensorArray.from_numpy(seqs)],
                names=["input_ids"],
            )
        if writer is None:
            kwargs = {
                "compression": self.compression,
                "use_dictionary": self.use_dictionary,
                "write_statistics": False,
            }
            if self.compression_level is not None:
                kwargs["compression_level"] = self.compression_level
            writer = self.pq.ParquetWriter(output_path, table.schema, **kwargs)
        writer.write_table(table)
        return writer
