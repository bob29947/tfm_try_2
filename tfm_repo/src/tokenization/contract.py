# SPDX-License-Identifier: Apache-2.0
"""Logical and on-disk contracts shared by every tokenization stage."""

MERCHANT_HASH_SIZE = 2000
MERCHANT_HASH_MODE = "integer_mod"

# Match the original blueprint: 315 transactions at 12 fields plus separators
# fill one 4096-token causal-LM sequence.
SEQ_LENGTH = 4096
SEQ_CHUNK_SIZE = 315
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
UNK_TOKEN_ID = 4

SPLITS = ("train", "val", "test")
FAST_OUTPUT_IN_PROGRESS = "_IN_PROGRESS"
FAST_OUTPUT_SUCCESS = "_SUCCESS"
TOKENIZATION_MANIFEST = "_tokenization_manifest.json"

__all__ = [
    "FAST_OUTPUT_IN_PROGRESS",
    "FAST_OUTPUT_SUCCESS",
    "BOS_TOKEN_ID",
    "EOS_TOKEN_ID",
    "MERCHANT_HASH_MODE",
    "MERCHANT_HASH_SIZE",
    "PAD_TOKEN_ID",
    "SEQ_CHUNK_SIZE",
    "SEQ_LENGTH",
    "SEP_TOKEN_ID",
    "SPLITS",
    "TOKENIZATION_MANIFEST",
    "UNK_TOKEN_ID",
]
