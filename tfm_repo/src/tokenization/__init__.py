# SPDX-License-Identifier: Apache-2.0
"""TFM tokenization runtime components."""

__all__ = [
    "FastParquetSplitTokenizer",
    "GPUTokenizer",
    "build_sequences",
    "tokenized_parquet_read_kwargs",
]


def __getattr__(name: str):
    """Load runtime components lazily so config-only CLIs stay lightweight."""
    if name == "FastParquetSplitTokenizer":
        from .fast_actor import FastParquetSplitTokenizer

        return FastParquetSplitTokenizer
    if name in {"GPUTokenizer", "build_sequences"}:
        from .legacy import GPUTokenizer, build_sequences

        return {"GPUTokenizer": GPUTokenizer, "build_sequences": build_sequences}[name]
    if name == "tokenized_parquet_read_kwargs":
        from .tensor_io import tokenized_parquet_read_kwargs

        return tokenized_parquet_read_kwargs
    raise AttributeError(name)
