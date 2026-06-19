"""Collect public Hacker News comments for TakeMeter annotation."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import requests

from ai201_project3_takemeter.data_io import write_jsonl

ALGOLIA_API = "https://hn.algolia.com/api/v1/search_by_date"
COMMENT_MIN_CHARS = 80
DEFAULT_OUTPUT = Path("data/raw/hackernews_comments.jsonl")
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_hn_text(raw: str) -> str:
    """Convert HN's small HTML subset to plain text."""
    text = raw.replace("<p>", "\n")
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return SPACE_RE.sub(" ", text).strip()


def collect_comments(
    *,
    target: int,
    min_chars: int,
    pages: int,
) -> list[dict[str, str]]:
    """Collect clean public HN comments from the public HN Search API."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for page in range(pages):
        response = requests.get(
            ALGOLIA_API,
            params={"tags": "comment", "hitsPerPage": 1000, "page": page},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        hits = payload.get("hits") or []
        if not isinstance(hits, list) or not hits:
            break

        for hit in hits:
            if not isinstance(hit, dict):
                continue
            object_id = str(hit.get("objectID") or "")
            if not object_id or object_id in seen:
                continue
            seen.add(object_id)

            raw_text = str(hit.get("comment_text") or "")
            text = clean_hn_text(raw_text)
            if len(text) < min_chars:
                continue

            story_id = str(hit.get("story_id") or "")
            rows.append(
                {
                    "id": object_id,
                    "source": f"https://news.ycombinator.com/item?id={object_id}",
                    "story_id": story_id,
                    "story_title": str(hit.get("story_title") or ""),
                    "author": str(hit.get("author") or ""),
                    "created_at": str(hit.get("created_at") or ""),
                    "text": text,
                    "label": "",
                    "notes": "",
                    "review_status": "unlabeled",
                }
            )
            if len(rows) >= target:
                return rows
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Collect public Hacker News comments for TakeMeter labeling."
    )
    parser.add_argument("--target", type=int, default=240, help="comments to collect")
    parser.add_argument("--pages", type=int, default=3, help="HN Search pages to scan")
    parser.add_argument(
        "--min-chars",
        type=int,
        default=COMMENT_MIN_CHARS,
        help="minimum cleaned comment length",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    rows = collect_comments(
        target=args.target,
        min_chars=args.min_chars,
        pages=args.pages,
    )
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} comments to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
