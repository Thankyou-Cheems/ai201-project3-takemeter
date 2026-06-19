"""Create a review queue with rough labels that must be human-reviewed."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ai201_project3_takemeter.data_io import read_rows, write_jsonl
from ai201_project3_takemeter.labels import LABEL_NAMES

DEFAULT_INPUT = Path("data/raw/hackernews_comments.jsonl")
DEFAULT_OUTPUT = Path("data/labeled/takemeter_hn_review_queue.jsonl")
EVIDENCE_RE = re.compile(
    r"\b(benchmark|latency|p95|because|for example|measured|data|study|"
    r"numbers?|percent|ms|seconds?|source|compare|tested|evidence)\b",
    re.IGNORECASE,
)
REASON_RE = re.compile(
    r"\b(i think|i would|in my experience|probably|seems|because|reason|"
    r"trade-?off|maintenance|risk|cost|benefit|should|would)\b",
    re.IGNORECASE,
)
LOW_RE = re.compile(
    r"\b(lol|lmao|this is dumb|nonsense|hype|trash|ridiculous|obviously|"
    r"never|always|everyone knows)\b",
    re.IGNORECASE,
)


def rough_label(text: str) -> tuple[str, str]:
    """Assign a rough pre-label for faster human review."""
    words = text.split()
    has_number = any(char.isdigit() for char in text)
    if len(words) < 18 or LOW_RE.search(text):
        return (
            "low_substance",
            "Heuristic: short or reaction-heavy; verify it lacks a concrete reason.",
        )
    if has_number or len(words) >= 80 or EVIDENCE_RE.search(text):
        return (
            "evidence_based",
            "Heuristic: contains evidence cues, detail, or length; "
            "verify evidence supports a claim.",
        )
    if REASON_RE.search(text) or len(words) >= 35:
        return (
            "reasoned_opinion",
            "Heuristic: contains rationale cues; "
            "verify no concrete evidence upgrades it.",
        )
    return (
        "low_substance",
        "Heuristic fallback: verify whether a real reason is present.",
    )


def prelabel(input_path: Path, output_path: Path) -> int:
    """Write rough pre-labels as JSONL for manual review."""
    rows = read_rows(input_path)

    for row in rows:
        label, note = rough_label(row.get("text", ""))
        row["label"] = label
        row["notes"] = note
        row["review_status"] = "needs_human_review"

    write_jsonl(rows, output_path)
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Create rough AI-assisted labels. You must review every row before "
            "submitting the final dataset."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    count = prelabel(args.input, args.output)
    print(f"Wrote {count} rough labels to {args.output}")
    print(f"Allowed final labels: {', '.join(LABEL_NAMES)}")
    print(
        "Review every row, then save the final file as "
        "data/labeled/takemeter_hn_labeled.jsonl"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
