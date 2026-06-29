#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point for paired tokenization validation."""

from __future__ import annotations

import sys
from pathlib import Path


TFM_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if str(TFM_ROOT) not in sys.path:
        sys.path.insert(0, str(TFM_ROOT))
    from validation.tokenization import main as package_main

    package_main()


if __name__ == "__main__":
    main()
