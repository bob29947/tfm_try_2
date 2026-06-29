# SPDX-License-Identifier: Apache-2.0
"""Named benchmark profiles for the tokenization command line interface."""

from __future__ import annotations

import json
from pathlib import Path


TFM_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = TFM_ROOT / "benchmarks" / "tokenization" / "profiles"
PROFILE_SCHEMA_VERSION = 1


def available_profiles() -> tuple[str, ...]:
    """Return deterministic names for bundled tokenization profiles."""
    if not PROFILE_DIR.exists():
        return ()
    return tuple(path.stem for path in sorted(PROFILE_DIR.glob("*.json")))


def load_profile(name: str) -> dict:
    """Load validated argparse defaults from a bundled profile."""
    if not name or Path(name).name != name or name.endswith(".json"):
        raise ValueError(f"Invalid tokenization profile name: {name!r}")
    path = PROFILE_DIR / f"{name}.json"
    if not path.exists():
        choices = ", ".join(available_profiles()) or "none installed"
        raise ValueError(f"Unknown tokenization profile {name!r}; available: {choices}")

    document = json.loads(path.read_text())
    if document.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported profile schema in {path}: "
            f"{document.get('schema_version')!r}"
        )
    defaults = document.get("arguments")
    if not isinstance(defaults, dict):
        raise ValueError(f"Profile {path} must contain an 'arguments' object")
    return dict(defaults)


__all__ = ["PROFILE_DIR", "available_profiles", "load_profile"]
