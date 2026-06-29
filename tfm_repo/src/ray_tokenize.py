# SPDX-License-Identifier: Apache-2.0
"""Compatibility facade for the tokenization runtime.

New code may import focused modules under :mod:`src.tokenization`. Existing
notebooks and scripts can continue importing these names from this module.
"""

from .tokenization.fast_actor import FastParquetSplitTokenizer
from .tokenization.legacy import GPUTokenizer, build_sequences
from .tokenization.tensor_io import tokenized_parquet_read_kwargs

__all__ = [
    "FastParquetSplitTokenizer",
    "GPUTokenizer",
    "build_sequences",
    "tokenized_parquet_read_kwargs",
]
