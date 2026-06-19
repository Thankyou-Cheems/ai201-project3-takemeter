"""Export the reviewed JSONL dataset to the CSV shape expected by Colab."""

from __future__ import annotations

import argparse
from pathlib import Path

from ai201_project3_takemeter.data_io import read_rows, write_colab_csv

DEFAULT_INPUT = Path("data/labeled/takemeter_hn_labeled.jsonl")
DEFAULT_OUTPUT = Path("data/colab/takemeter_hn_labeled_for_colab.csv")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Export the JSONL source dataset to a minimal text,label CSV for "
            "the course starter Colab notebook."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    rows = read_rows(args.input)
    write_colab_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
