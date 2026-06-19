"""Validate a TakeMeter labeled dataset before uploading it to Colab."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ai201_project3_takemeter.data_io import read_rows
from ai201_project3_takemeter.labels import LABEL_NAMES

REQUIRED_COLUMNS = {"text", "label"}
DEFAULT_DATASET = Path("data/labeled/takemeter_hn_labeled.jsonl")


def validate_rows(
    rows: list[dict[str, str]],
    *,
    min_examples: int = 200,
    max_single_label_ratio: float = 0.70,
    allow_unreviewed: bool = False,
) -> list[str]:
    """Return validation errors for the assignment requirements."""
    errors: list[str] = []
    if not rows:
        return ["Dataset has no data rows."]

    columns = set(rows[0])
    missing = REQUIRED_COLUMNS - columns
    if missing:
        errors.append(f"Missing required columns: {', '.join(sorted(missing))}")
        return errors

    if len(rows) < min_examples:
        errors.append(f"Need at least {min_examples} examples; found {len(rows)}.")

    allowed = set(LABEL_NAMES)
    label_counts = Counter(row.get("label", "").strip() for row in rows)
    empty_labels = label_counts.pop("", 0)
    if empty_labels:
        errors.append(f"{empty_labels} rows are missing labels.")

    unknown = sorted(label for label in label_counts if label not in allowed)
    if unknown:
        errors.append(
            "Unknown labels: "
            + ", ".join(unknown)
            + f". Expected one of: {', '.join(LABEL_NAMES)}."
        )

    if rows and label_counts:
        most_common_label, most_common_count = label_counts.most_common(1)[0]
        ratio = most_common_count / len(rows)
        if ratio > max_single_label_ratio:
            errors.append(
                f"Label imbalance: {most_common_label} is {ratio:.1%} of rows; "
                "assignment requires no single label above "
                f"{max_single_label_ratio:.0%}."
            )

    if not allow_unreviewed and "review_status" in columns:
        unreviewed_statuses = {"needs_human_review", "unlabeled", ""}
        unreviewed = sum(
            1
            for row in rows
            if row.get("review_status", "").strip() in unreviewed_statuses
        )
        if unreviewed:
            errors.append(
                f"{unreviewed} rows still need human review. "
                "Set review_status to reviewed after checking each label."
            )

    return errors


def print_distribution(rows: list[dict[str, str]]) -> None:
    """Print label counts and percentages."""
    counts = Counter(row.get("label", "").strip() or "<missing>" for row in rows)
    total = len(rows)
    for label, count in counts.most_common():
        print(f"{label}: {count} ({count / total:.1%})")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Validate a TakeMeter JSONL dataset.")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--min-examples", type=int, default=200)
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="allow rough review queues to pass distribution checks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    rows = read_rows(args.path)
    print(f"Rows: {len(rows)}")
    print_distribution(rows)
    errors = validate_rows(
        rows,
        min_examples=args.min_examples,
        allow_unreviewed=args.allow_unreviewed,
    )
    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
