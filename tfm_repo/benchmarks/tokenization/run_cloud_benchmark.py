#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Executable entry point for the four-node S3 tokenization benchmark."""

from __future__ import annotations

import sys
from pathlib import Path


TFM_ROOT = Path(__file__).resolve().parents[2]
if str(TFM_ROOT) not in sys.path:
    sys.path.insert(0, str(TFM_ROOT))

from benchmarks.tokenization.cloud_benchmark import main


if __name__ == "__main__":
    main()
