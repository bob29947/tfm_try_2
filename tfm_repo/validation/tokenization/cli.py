"""Unified command line interface for paired tokenization validation."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    module: str
    help: str


COMMANDS = {
    "prepare-sample": Command(
        ".prepare_sample",
        "Create the fixed, auditable transaction sample.",
    ),
    "tokenize-sample": Command(
        ".tokenize_sample",
        "Tokenize the sample with both merchant mappings.",
    ),
    "compare-language-models": Command(
        ".compare_language_models",
        "Train and compare the fixed-seed language-model proxies.",
    ),
    "extract-embeddings": Command(
        ".extract_embeddings",
        "Extract paired transaction embeddings from both checkpoints.",
    ),
    "evaluate-fraud": Command(
        ".evaluate_fraud",
        "Run the paired downstream fraud-model evaluation.",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m validation.tokenization",
        description=__doc__,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        required=True,
    )
    for name, command in COMMANDS.items():
        subparsers.add_parser(name, add_help=False, help=command.help)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, command_argv = parser.parse_known_args(argv)
    command = COMMANDS[args.command]
    module = importlib.import_module(command.module, package=__package__)
    module.main(command_argv)
