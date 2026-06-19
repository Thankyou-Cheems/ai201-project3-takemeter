"""Structured dataset I/O for TakeMeter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

JsonRow = dict[str, Any]


def read_jsonl(path: Path) -> list[JsonRow]:
    """Read a JSONL file into a list of dictionaries."""
    rows: list[JsonRow] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(payload)
    return rows


def write_jsonl(rows: list[JsonRow], path: Path) -> None:
    """Write dictionaries as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def read_csv(path: Path) -> list[JsonRow]:
    """Read a CSV file into dictionaries."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_rows(path: Path) -> list[JsonRow]:
    """Read rows from JSONL, with CSV kept only for Colab-export validation."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".csv":
        return read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def write_colab_csv(rows: list[JsonRow], path: Path) -> None:
    """Write the minimal text,label CSV expected by the starter Colab notebook."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "text": str(row.get("text", "")),
                    "label": str(row.get("label", "")),
                }
            )
