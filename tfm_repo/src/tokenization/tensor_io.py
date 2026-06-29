# SPDX-License-Identifier: Apache-2.0
"""Interoperability helpers for tokenized Parquet tensors."""

from __future__ import annotations

import numpy as np


def tokenized_parquet_read_kwargs(document: dict) -> dict:
    """Return Ray's public tensor-schema option for a tokenization manifest.

    Legacy outputs already use an Arrow tensor extension and need no override.
    The fast writer stores each contiguous sequence as fixed-size binary, which
    ``ray.data.read_parquet`` can decode through ``tensor_column_schema``.
    """
    config = document.get("config", document)
    output = config.get("output", {})
    if output.get("format") != "binary-tensor":
        return {}

    return {
        "tensor_column_schema": {
            "input_ids": (
                np.dtype(output["dtype"]),
                (int(config["sequence_length"]),),
            )
        }
    }


__all__ = ["tokenized_parquet_read_kwargs"]
