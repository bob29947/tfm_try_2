# SPDX-License-Identifier: Apache-2.0
"""Direct GPU Parquet tokenization actor.

Module-level imports are head-safe (numpy only). GPU libraries are imported
lazily when the actor starts so the module remains importable on a CPU driver.
"""

from __future__ import annotations

import gc
import os
import socket
import time
from urllib.parse import urlsplit

import numpy as np

from . import contract as C


def _configure_kvikio_runtime(
    defaults_module,
    *,
    num_threads: int,
    task_size_bytes: int,
) -> dict[str, int]:
    """Set and verify the KvikIO process defaults used by cuDF remote reads."""
    requested = {
        "num_threads": int(num_threads),
        "task_size_bytes": int(task_size_bytes),
    }
    defaults_module.set(
        {
            "num_threads": requested["num_threads"],
            "task_size": requested["task_size_bytes"],
        }
    )
    realized = {
        "num_threads": int(defaults_module.get("num_threads")),
        "task_size_bytes": int(defaults_module.get("task_size")),
    }
    if realized != requested:
        raise RuntimeError(
            "KvikIO runtime settings mismatch: "
            f"requested={requested!r}, realized={realized!r}"
        )
    return realized


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
        row_groups_per_batch: int = 16,
        arrow_cpu_threads: int | None = None,
        write_threads: int = 1,
        output_shard_size_bytes: int = 256 * 1024 * 1024,
        validate_order: bool = False,
        s3_mode: bool = False,
        aws_region: str | None = None,
        s3_connections: int = 8,
        kvikio_task_size_bytes: int = 4 * 1024 * 1024,
        overlap_split_writes: bool = False,
        require_kvikio: bool = True,
    ):
        self.s3_mode = bool(s3_mode)
        self.s3_connections = max(1, int(s3_connections))
        self.kvikio_task_size_bytes = int(kvikio_task_size_bytes)
        if self.kvikio_task_size_bytes < 1:
            raise ValueError("kvikio_task_size_bytes must be at least 1")
        self.overlap_split_writes = bool(overlap_split_writes)
        self.aws_region = aws_region
        self.arrow_s3_fs = None
        self.s3fs = None
        self._s3_backend = "local"
        self._kvikio_realized = None

        if self.s3_mode:
            # cuDF's KvikIO remote reader consumes AWS credentials from the
            # environment.  Resolve the instance-role chain once per actor,
            # before importing cuDF, and freeze it for this bounded benchmark.
            # s3fs and Arrow clients are likewise actor-local and reused.
            import botocore.session

            session = botocore.session.get_session()
            credentials = session.get_credentials()
            if credentials is None:
                raise RuntimeError(
                    "No AWS credentials available for the fail-closed S3 GPU path"
                )
            frozen = credentials.get_frozen_credentials()
            os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
            if frozen.token:
                os.environ["AWS_SESSION_TOKEN"] = frozen.token
            else:
                os.environ.pop("AWS_SESSION_TOKEN", None)

            self.aws_region = (
                self.aws_region
                or os.environ.get("AWS_DEFAULT_REGION")
                or os.environ.get("AWS_REGION")
                or session.get_config_variable("region")
                or "us-west-2"
            )
            os.environ["AWS_DEFAULT_REGION"] = self.aws_region
            os.environ["AWS_REGION"] = self.aws_region
            # KvikIO documents this as the remote reader's TCP concurrency
            # control; set it before importing cuDF/KvikIO.
            os.environ["KVIKIO_NTHREADS"] = str(self.s3_connections)
            os.environ["KVIKIO_TASK_SIZE"] = str(self.kvikio_task_size_bytes)

            import s3fs
            import pyarrow.fs as pafs

            self.s3fs = s3fs.S3FileSystem(
                anon=False,
                client_kwargs={"region_name": self.aws_region},
                config_kwargs={"max_pool_connections": self.s3_connections},
            )
            self.arrow_s3_fs = pafs.S3FileSystem(
                region=self.aws_region,
                background_writes=True,
            )

            if require_kvikio:
                try:
                    import kvikio.defaults as kvikio_defaults
                    from kvikio.remote_file import is_remote_file_available
                except (ImportError, ModuleNotFoundError) as exc:
                    raise RuntimeError(
                        "KvikIO remote I/O is required for S3 tokenization"
                    ) from exc
                if not is_remote_file_available():
                    raise RuntimeError(
                        "KvikIO was built without remote I/O support; refusing "
                        "to fall back to whole-object S3 prefetch"
                    )

                # Apply these settings through KvikIO's runtime API as well as
                # its environment variables, then record the values actually
                # realized by the actor.  This occurs during actor startup,
                # before cuDF can issue a remote read.  A mismatch is fatal so
                # a benchmark never silently measures a different thread pool
                # or transfer granularity than its requested configuration.
                self._kvikio_realized = _configure_kvikio_runtime(
                    kvikio_defaults,
                    num_threads=self.s3_connections,
                    task_size_bytes=self.kvikio_task_size_bytes,
                )

        # The fast path creates and releases several multi-GiB temporary arrays.
        # cudaMalloc/cudaFree serialization is otherwise a material part of the
        # data stopwatch, so give each long-lived, one-GPU actor its own RMM
        # pool before importing cuDF or CuPy.  Start at 8 GiB (large enough to
        # recycle the hot temporaries without reserving most of a 32 GiB GPU),
        # keep 2 GiB outside the pool for CUDA libraries, and adapt down on
        # smaller or partially occupied GPUs.
        import rmm

        self.rmm = rmm

        gib = 1 << 30
        free_bytes, _ = rmm.mr.available_device_memory()
        maximum_pool_size = min(30 * gib, max(0, free_bytes - 2 * gib))
        initial_pool_size = min(8 * gib, maximum_pool_size)
        if initial_pool_size >= gib:
            rmm.reinitialize(
                pool_allocator=True,
                initial_pool_size=initial_pool_size,
                maximum_pool_size=maximum_pool_size,
            )

        import cudf  # lazy: worker-only
        import cupy as cp
        import pyarrow as pa
        import pyarrow.parquet as pq
        from rmm.allocators.cupy import rmm_cupy_allocator
        from ray.data.extensions.tensor_extension import ArrowTensorArray
        from src.tokenizer.financial_pipeline import (
            ALL_STATES,
            CHIP_MAPPING,
            INDUSTRY_RANGES,
            KNOWN_MCCS,
        )

        cp.cuda.set_allocator(rmm_cupy_allocator)

        if self.s3_mode:
            try:
                cudf.set_option("kvikio_remote_io", True)
                kvikio_enabled = bool(cudf.get_option("kvikio_remote_io"))
            except (KeyError, AttributeError, ValueError) as exc:
                raise RuntimeError(
                    "Installed cuDF does not expose the KvikIO remote I/O option"
                ) from exc
            if require_kvikio and not kvikio_enabled:
                raise RuntimeError(
                    "cuDF rejected KvikIO remote I/O; refusing S3 fallback"
                )
            self._s3_backend = "cudf-kvikio"

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
        self.write_threads = max(1, int(write_threads))
        self.output_shard_size_bytes = max(1, int(output_shard_size_bytes))
        self.validate_order = bool(validate_order)

        output_cuda_type = {
            "uint16": "unsigned short",
            "int32": "int",
            "int64": "long long",
        }[self.output_np_dtype.name]
        self._sequence_scatter_kernel = cp.RawKernel(
            rf"""
            extern "C" __global__
            void scatter_sequences(
                const long long* keys,
                const long long* seq_idx,
                const long long* txn_pos,
                const {output_cuda_type}* token_ids,
                {output_cuda_type}* out,
                const long long n,
                const int n_fields,
                const int seq_length,
                const int chunk_size,
                const {output_cuda_type} sep_token,
                const {output_cuda_type} eos_token)
            {{
                const long long row =
                    (long long)blockDim.x * blockIdx.x + threadIdx.x;
                if (row >= n) return;

                const long long txn = txn_pos[row];
                const long long dst = seq_idx[row] * seq_length
                    + 1 + txn * (n_fields + 1);
                const long long src = row * n_fields;
                #pragma unroll
                for (int field = 0; field < {self.n_fields}; ++field) {{
                    out[dst + field] = token_ids[src + field];
                }}
                const bool last = txn == chunk_size - 1
                    || row + 1 == n || keys[row + 1] != keys[row];
                out[dst + n_fields] = last ? eos_token : sep_token;
            }}
            """,
            "scatter_sequences",
        )
        self._sequence_scatter_kernel.compile()

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

        cat_lookup = np.full(10_000, self.cat_default_idx, dtype=np.int32)
        for lo, hi, label in self.industry_ranges:
            cat_lookup[max(0, lo) : min(10_000, hi + 1)] = self.cat_idx[label]
        self.cat_lookup = cp.asarray(cat_lookup)

        mcc_lookup = np.full(10_000, self.mcc_default_idx, dtype=np.int32)
        for mcc, idx in self.mcc_idx.items():
            if 0 <= mcc < len(mcc_lookup):
                mcc_lookup[mcc] = idx
        self.mcc_lookup = cp.asarray(mcc_lookup)

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

    def ready(self) -> dict:
        """Return the realized actor I/O backend after initialization."""
        import ray

        context = ray.get_runtime_context()
        accelerator_ids = (
            context.get_accelerator_ids()
            if hasattr(context, "get_accelerator_ids")
            else {"GPU": [str(value) for value in ray.get_gpu_ids()]}
        )
        return {
            "ready": True,
            "node_id": str(context.get_node_id()),
            "hostname": socket.gethostname(),
            "accelerator_ids": {
                str(kind): [str(value) for value in values]
                for kind, values in accelerator_ids.items()
            },
            "s3_mode": self.s3_mode,
            "read_backend": self._s3_backend,
            "write_backend": (
                "pyarrow.fs.S3FileSystem" if self.s3_mode else "local-pyarrow"
            ),
            "aws_region": self.aws_region,
            "s3_connections": self.s3_connections if self.s3_mode else None,
            "kvikio_num_threads": (
                self._kvikio_realized["num_threads"]
                if self._kvikio_realized is not None
                else None
            ),
            "kvikio_task_size_bytes": (
                self._kvikio_realized["task_size_bytes"]
                if self._kvikio_realized is not None
                else None
            ),
            "row_groups_per_batch": self.row_groups_per_batch,
            "overlap_split_writes": getattr(self, "overlap_split_writes", False),
        }

    def tokenize(self, work_items: list[dict]) -> list[dict]:
        # A 24 GiB L4 cannot safely retain raw train/val/test frames together.
        # GPU frames are always released split by split; the opt-in path only
        # retains bounded host output while preparing the next split.
        if getattr(self, "overlap_split_writes", False) and len(work_items) > 1:
            return self._tokenize_with_overlapped_writes(work_items)
        return [self._tokenize_one_split(work) for work in work_items]

    def __call__(self, work_items: list[dict]) -> list[dict]:
        return self.tokenize(work_items)

    def _tokenize_one_split(self, work: dict) -> dict:
        stat, shards, started = self._prepare_one_split(work)
        if shards:
            op_started = time.perf_counter()
            self._write_shards_sync(shards)
            stat["write_s"] = time.perf_counter() - op_started
            stat["write_wait_s"] = stat["write_s"]
        stat["elapsed_s"] = time.perf_counter() - started
        return stat

    def _prepare_one_split(
        self, work: dict
    ) -> tuple[dict, list[tuple[str, np.ndarray]], float]:
        """Read, tokenize, and build one split without starting its writes."""
        output_path = work["output_path"]
        count = 0
        rows = 0
        read_s = 0.0
        tokenize_s = 0.0
        sort_s = 0.0
        sequence_s = 0.0
        started = time.perf_counter()
        self._prev_key = None
        key_batches = []
        time_batches = []
        token_batches = []
        output_paths = []
        shards: list[tuple[str, np.ndarray]] = []
        free_before, total_device_memory = self.rmm.mr.available_device_memory()
        reserved_before = int(total_device_memory - free_before)

        try:
            for fragment in work["fragments"]:
                row_groups = list(fragment["row_groups"])
                for start in range(0, len(row_groups), self.row_groups_per_batch):
                    batch_row_groups = row_groups[start:start + self.row_groups_per_batch]
                    op_started = time.perf_counter()
                    gdf = self._read_row_groups(fragment["path"], batch_row_groups)
                    read_s += time.perf_counter() - op_started
                    if len(gdf) == 0:
                        del gdf
                        continue
                    gdf = self._filter_user_range(gdf, work["user_min"], work["user_max"])
                    if len(gdf) == 0:
                        del gdf
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
                # The concatenated arrays own their data.  Drop all per-read
                # batches before allocating sort indices on a 24 GiB L4.
                key_batches.clear()
                time_batches.clear()
                token_batches.clear()
                # The temporal split is normally already ordered by
                # User/Card/time.  Avoid materializing and applying a full
                # permutation when that contract holds, but retain the exact
                # legacy ordering for any input that does not.
                order = None
                if not self._is_ordered(keys, txn_order):
                    if self._is_user_time_ordered(keys, txn_order):
                        # Stable key-only sorting preserves chronological
                        # order within each User/Card.
                        order = self.cp.argsort(keys)
                    else:
                        order = self.cp.lexsort(
                            self.cp.stack([txn_order, keys])
                        )
                    keys = keys[order]
                    token_ids = token_ids[order]
                sort_s += time.perf_counter() - op_started

                op_started = time.perf_counter()
                seqs = self._build_sequences_gpu(keys, token_ids)
                sequence_s += time.perf_counter() - op_started
                count += len(seqs)

                shards = self._sequence_output_shards(output_path, seqs)
                output_paths = [path for path, _ in shards]
                del keys, txn_order, token_ids, seqs
                if order is not None:
                    del order
        finally:
            self._prev_key = None
            key_batches.clear()
            time_batches.clear()
            token_batches.clear()

        compute_s = time.perf_counter() - started
        free_after, total_device_memory_after = self.rmm.mr.available_device_memory()
        reserved_after = int(total_device_memory_after - free_after)
        stat = {
            "split": work["split"],
            "count": count,
            "rows": rows,
            "elapsed_s": compute_s,
            "compute_s": compute_s,
            "read_s": read_s,
            "tokenize_s": tokenize_s,
            "sort_s": sort_s,
            "sequence_s": sequence_s,
            "write_s": 0.0,
            "write_wait_s": 0.0,
            "write_overlap_s": 0.0,
            "output_files": len(output_paths),
            "output_paths": output_paths,
            "output_path": output_paths[0] if output_paths else output_path,
            # The RMM pool retains acquired CUDA allocations until actor exit,
            # so its post-compute driver reservation is a stable high-water
            # signal for deciding whether a batching config is safe on L4.
            "gpu_memory_reserved_before_bytes": reserved_before,
            "peak_gpu_memory_bytes": max(reserved_before, reserved_after),
            "gpu_total_memory_bytes": int(total_device_memory_after),
        }
        return stat, shards, started

    def _write_shards_sync(self, shards: list[tuple[str, np.ndarray]]) -> None:
        """Write every shard and clean visible siblings if any writer fails."""
        paths = [path for path, _ in shards]
        try:
            if len(shards) == 1:
                path, shard = shards[0]
                self._write_sequences(None, path, shard)
                return

            from concurrent.futures import ThreadPoolExecutor, wait

            with ThreadPoolExecutor(max_workers=self.write_threads) as executor:
                futures = [
                    executor.submit(self._write_sequences, None, path, shard)
                    for path, shard in shards
                ]
                # Settle all streams before surfacing a failure. This guarantees
                # that cleanup never races an in-flight multipart close.
                wait(futures)
                for future in futures:
                    future.result()
        except BaseException:
            self._cleanup_output_paths(paths)
            raise

    def _tokenize_with_overlapped_writes(
        self, work_items: list[dict]
    ) -> list[dict]:
        """Overlap split N writes with split N+1 GPU work, one split at a time.

        Exactly one split may have writes in flight. The next split can be
        prepared concurrently, after which the prior writes are drained before
        another set is submitted. Thus host memory retains at most the prior
        and current split outputs, while GPU frames remain strictly split-local.
        """
        from concurrent.futures import ThreadPoolExecutor

        stats: list[dict] = []
        completed_paths: list[str] = []
        pending: dict | None = None
        prepared_shards: list[tuple[str, np.ndarray]] = []

        with ThreadPoolExecutor(max_workers=self.write_threads) as executor:
            try:
                for work in work_items:
                    stat, prepared_shards, started = self._prepare_one_split(work)

                    # The previous split has been writing while this split used
                    # the GPU. Drain it before submitting another split so the
                    # actor never accumulates an unbounded host-side queue.
                    if pending is not None:
                        previous = pending
                        pending = None
                        self._finish_pending_writes(previous)
                        completed_paths.extend(previous["paths"])
                        stats.append(previous["stat"])

                    if prepared_shards:
                        pending = self._submit_split_writes(
                            executor, stat, prepared_shards, started
                        )
                    else:
                        stats.append(stat)
                    prepared_shards = []

                if pending is not None:
                    previous = pending
                    pending = None
                    self._finish_pending_writes(previous)
                    completed_paths.extend(previous["paths"])
                    stats.append(previous["stat"])
            except BaseException:
                # If GPU preparation fails, settle the one allowed pending
                # batch before deleting its visible objects. If a writer fails,
                # _finish_pending_writes already settled and cleaned that batch.
                if pending is not None:
                    try:
                        self._finish_pending_writes(pending)
                    except BaseException:
                        pass
                    self._cleanup_output_paths(pending["paths"])
                self._cleanup_output_paths(completed_paths)
                prepared_shards.clear()
                raise

        return stats

    def _submit_split_writes(
        self,
        executor,
        stat: dict,
        shards: list[tuple[str, np.ndarray]],
        split_started: float,
    ) -> dict:
        from concurrent.futures import wait

        submitted = time.perf_counter()
        paths = [path for path, _ in shards]
        futures = []
        try:
            for path, shard in shards:
                futures.append(
                    executor.submit(self._timed_write_sequence, path, shard)
                )
        except BaseException:
            # Submission itself can fail after earlier futures have started.
            # Settle those writers before deleting their keys.
            wait(futures)
            self._cleanup_output_paths(paths)
            raise
        return {
            "stat": stat,
            "paths": paths,
            "futures": futures,
            "split_started": split_started,
            "submitted": submitted,
        }

    def _timed_write_sequence(self, path: str, shard: np.ndarray) -> dict:
        started = time.perf_counter()
        self._write_sequences(None, path, shard)
        return {
            "elapsed_s": time.perf_counter() - started,
            "finished": time.perf_counter(),
        }

    def _finish_pending_writes(self, pending: dict) -> None:
        from concurrent.futures import wait

        wait_started = time.perf_counter()
        wait(pending["futures"])
        write_wait_s = time.perf_counter() - wait_started
        results = []
        first_error: BaseException | None = None
        for future in pending["futures"]:
            try:
                results.append(future.result())
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            self._cleanup_output_paths(pending["paths"])
            raise first_error

        finished = max(
            (result["finished"] for result in results),
            default=pending["submitted"],
        )
        write_s = finished - pending["submitted"]
        stat = pending["stat"]
        stat["write_s"] = write_s
        stat["write_wait_s"] = min(write_wait_s, write_s)
        stat["write_overlap_s"] = max(0.0, write_s - stat["write_wait_s"])
        stat["elapsed_s"] = finished - pending["split_started"]

    def _cleanup_output_paths(self, output_paths: list[str]) -> None:
        """Best-effort deletion after a split or actor-level failure."""
        for output_path in dict.fromkeys(output_paths):
            try:
                if output_path.startswith("s3://"):
                    if self.arrow_s3_fs is None:
                        continue
                    parsed = urlsplit(output_path)
                    arrow_path = f"{parsed.netloc}/{parsed.path.lstrip('/')}"
                    info = self.arrow_s3_fs.get_file_info(arrow_path)
                    if info.type.name == "File":
                        self.arrow_s3_fs.delete_file(arrow_path)
                elif os.path.isfile(output_path):
                    os.remove(output_path)
            except Exception:
                # The benchmark harness performs a prefix-level fallback
                # cleanup, including multipart aborts, after actor failure.
                pass

    def _tokenize_combined_splits(self, work_items: list[dict]) -> list[dict]:
        output_paths = {work["split"]: work["output_path"] for work in work_items}
        row_counts = {work["split"]: 0 for work in work_items}
        counts = {work["split"]: 0 for work in work_items}
        write_s = {work["split"]: 0.0 for work in work_items}
        output_files = {work["split"]: [] for work in work_items}
        read_s = 0.0
        tokenize_s = 0.0
        sort_s = 0.0
        sequence_s = 0.0
        started = time.perf_counter()
        frames = []

        try:
            for work in work_items:
                split = work["split"]
                for fragment in work["fragments"]:
                    row_groups = list(fragment["row_groups"])
                    for start in range(0, len(row_groups), self.row_groups_per_batch):
                        batch_row_groups = row_groups[start:start + self.row_groups_per_batch]
                        op_started = time.perf_counter()
                        gdf = self._read_row_groups(fragment["path"], batch_row_groups)
                        read_s += time.perf_counter() - op_started
                        if len(gdf) == 0:
                            continue
                        gdf = self._filter_user_range(
                            gdf, work["user_min"], work["user_max"]
                        )
                        if len(gdf) == 0:
                            continue
                        row_counts[split] += len(gdf)
                        frames.append(gdf)

            if frames:
                gdf = self.cudf.concat(frames, ignore_index=True)
                frames.clear()

                op_started = time.perf_counter()
                keys, txn_order, token_ids = self._tokenize_frame(gdf)
                tokenize_s += time.perf_counter() - op_started
                del gdf

                row_offset = 0
                from concurrent.futures import ThreadPoolExecutor

                write_started = {}
                futures = []

                def write_shard(split: str, path: str, split_seqs: np.ndarray):
                    op_started = time.perf_counter()
                    self._write_sequences(None, path, split_seqs)
                    return split, time.perf_counter() - op_started, time.perf_counter()

                with ThreadPoolExecutor(max_workers=self.write_threads) as executor:
                    for work in work_items:
                        split = work["split"]
                        split_rows = row_counts[split]
                        if split_rows == 0:
                            continue
                        row_end = row_offset + split_rows
                        split_keys = keys[row_offset:row_end]
                        split_order = txn_order[row_offset:row_end]
                        split_tokens = token_ids[row_offset:row_end]
                        row_offset = row_end

                        op_started = time.perf_counter()
                        ordered = self._is_ordered(split_keys, split_order)
                        if not ordered:
                            if self._is_user_time_ordered(split_keys, split_order):
                                # Stable key-only sorting preserves the existing
                                # chronological order within each User/Card.
                                order = self.cp.argsort(split_keys)
                            else:
                                order = self.cp.lexsort(
                                    self.cp.stack([split_order, split_keys])
                                )
                            split_keys = split_keys[order]
                            split_tokens = split_tokens[order]
                        sort_s += time.perf_counter() - op_started

                        op_started = time.perf_counter()
                        split_seqs = self._build_sequences_gpu(
                            split_keys,
                            split_tokens,
                        )
                        sequence_s += time.perf_counter() - op_started
                        if len(split_seqs) == 0:
                            continue
                        counts[split] += len(split_seqs)
                        shards = self._sequence_output_shards(
                            output_paths[split], split_seqs
                        )
                        output_files[split] = [path for path, _ in shards]
                        write_started[split] = time.perf_counter()
                        futures.extend(
                            executor.submit(write_shard, split, path, shard)
                            for path, shard in shards
                        )

                    del keys, txn_order, token_ids
                    write_finished = {split: 0.0 for split in output_paths}
                    for future in futures:
                        split, _, finished = future.result()
                        write_finished[split] = max(write_finished[split], finished)
                    for split, started_at in write_started.items():
                        write_s[split] = write_finished[split] - started_at
        finally:
            frames.clear()

        elapsed_s = time.perf_counter() - started
        return [
            {
                "split": work["split"],
                "count": counts[work["split"]],
                "rows": row_counts[work["split"]],
                "elapsed_s": elapsed_s,
                "read_s": read_s,
                "tokenize_s": tokenize_s,
                "sort_s": sort_s,
                "sequence_s": sequence_s,
                "write_s": write_s[work["split"]],
                "output_files": len(output_files[work["split"]]),
                "combined_splits": len(work_items),
                "output_path": (
                    output_files[work["split"]][0]
                    if output_files[work["split"]]
                    else output_paths[work["split"]]
                ),
            }
            for work in work_items
        ]

    def _is_ordered(self, keys, txn_order) -> bool:
        if len(keys) < 2:
            return True
        out_of_order = (keys[1:] < keys[:-1]) | (
            (keys[1:] == keys[:-1]) & (txn_order[1:] < txn_order[:-1])
        )
        return not bool(self.cp.any(out_of_order).get())

    def _is_user_time_ordered(self, keys, txn_order) -> bool:
        if len(keys) < 2:
            return True
        users = keys // 100
        out_of_order = (users[1:] < users[:-1]) | (
            (users[1:] == users[:-1]) & (txn_order[1:] < txn_order[:-1])
        )
        return not bool(self.cp.any(out_of_order).get())

    def _sequence_output_shards(
        self,
        output_path: str,
        seqs: np.ndarray,
    ) -> list[tuple[str, np.ndarray]]:
        shard_count = min(
            len(seqs),
            max(
                1,
                (seqs.nbytes + self.output_shard_size_bytes - 1)
                // self.output_shard_size_bytes,
            ),
        )
        if shard_count == 1:
            return [(output_path, seqs)]

        path = output_path
        stem, suffix = path.rsplit(".", 1)
        shard_id_width = max(2, len(str(shard_count - 1)))
        shards = []
        for shard_id in range(shard_count):
            start = shard_id * len(seqs) // shard_count
            end = (shard_id + 1) * len(seqs) // shard_count
            shard_path = f"{stem}-{shard_id:0{shard_id_width}d}.{suffix}"
            shards.append((shard_path, seqs[start:end]))
        return shards

    def _read_row_groups(self, path: str, row_groups: list[int]):
        cudf = self.cudf
        if path.startswith("s3://"):
            if not self.s3_mode or self._s3_backend != "cudf-kvikio":
                raise RuntimeError(
                    "S3 input requires the actor's fail-closed cuDF/KvikIO mode"
                )
            if not bool(cudf.get_option("kvikio_remote_io")):
                raise RuntimeError(
                    "cuDF KvikIO remote I/O was disabled after actor startup"
                )
            # Do not pass ``filesystem=self.s3fs`` here: that explicitly routes
            # through fsspec and can prefetch a whole object into host memory.
            # With the option above, cuDF builds a KvikIO remote datasource and
            # issues byte-range reads only for the requested row groups/columns.
            return cudf.read_parquet(
                path,
                engine="cudf",
                columns=self.COLUMNS,
                row_groups=row_groups,
                dataset_kwargs={"partitioning": None},
                use_pandas_metadata=False,
                categorical_partitions=False,
            )

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
        valid_mcc = (mcc_cp >= 0) & (mcc_cp < len(self.mcc_lookup))
        safe_mcc = cp.clip(mcc_cp, 0, len(self.mcc_lookup) - 1)
        cat_idx = cp.where(
            valid_mcc,
            self.cat_lookup[safe_mcc],
            self.cat_default_idx,
        )
        token_ids[:, 2] = self.offset_cat + cat_idx

        mcc_idx = cp.where(
            valid_mcc,
            self.mcc_lookup[safe_mcc],
            self.mcc_default_idx,
        )
        token_ids[:, 3] = self.offset_mcc + mcc_idx

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
        if self.seq_length - 1 == self.chunk_size * (self.n_fields + 1):
            threads = 256
            blocks = (n + threads - 1) // threads
            self._sequence_scatter_kernel(
                (blocks,),
                (threads,),
                (
                    keys,
                    seq_idx,
                    txn_pos,
                    token_ids,
                    out,
                    np.int64(n),
                    np.int32(self.n_fields),
                    np.int32(self.seq_length),
                    np.int32(self.chunk_size),
                    self.output_np_dtype.type(C.SEP_TOKEN_ID),
                    self.output_np_dtype.type(C.EOS_TOKEN_ID),
                ),
            )
        else:
            last_in_sequence = (txn_pos == self.chunk_size - 1) | cp.concatenate([
                keys[1:] != keys[:-1],
                cp.asarray([True], dtype=cp.bool_),
            ])
            base_pos = 1 + txn_pos * (self.n_fields + 1)
            field_offsets = cp.arange(self.n_fields, dtype=cp.int64)
            flat_pos = (
                seq_idx[:, None] * self.seq_length
                + base_pos[:, None]
                + field_offsets[None, :]
            )
            out.reshape(-1)[flat_pos.reshape(-1)] = token_ids.reshape(-1)
            sep_pos = base_pos + self.n_fields
            out.reshape(-1)[seq_idx * self.seq_length + sep_pos] = C.SEP_TOKEN_ID
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
        kwargs = {
            "compression": self.compression,
            "use_dictionary": self.use_dictionary,
            "write_statistics": False,
        }
        if self.compression_level is not None:
            kwargs["compression_level"] = self.compression_level
        if output_path.startswith("s3://"):
            if not self.s3_mode or self.arrow_s3_fs is None:
                raise RuntimeError(
                    "S3 output requires an actor-local Arrow S3 filesystem"
                )
            parsed = urlsplit(output_path)
            if not parsed.netloc or not parsed.path.lstrip("/"):
                raise ValueError(f"Invalid S3 output URI: {output_path!r}")
            arrow_path = f"{parsed.netloc}/{parsed.path.lstrip('/')}"
            sink = self.arrow_s3_fs.open_output_stream(
                arrow_path,
                metadata={"Content-Type": "application/vnd.apache.parquet"},
            )
            try:
                self.pq.write_table(table, sink, **kwargs)
                # Arrow's S3 close completes and waits for the multipart upload.
                sink.close()
            except BaseException:
                # C++ OutputStream::Abort is invoked by the unclosed S3 stream's
                # destructor.  Newer PyArrow builds may expose it directly.
                abort = getattr(sink, "abort", None)
                if callable(abort):
                    try:
                        abort()
                    except Exception:
                        pass
                sink = None
                gc.collect()
                # A failed close can very rarely leave a visible partial key;
                # never let that object be mistaken for a completed shard.
                try:
                    info = self.arrow_s3_fs.get_file_info(arrow_path)
                    if info.type.name == "File":
                        self.arrow_s3_fs.delete_file(arrow_path)
                except Exception:
                    pass
                raise
        else:
            self.pq.write_table(table, output_path, **kwargs)
        return None
