import csv

from ai201_project3_takemeter.collect_hn import clean_hn_text
from ai201_project3_takemeter.data_io import read_jsonl, write_colab_csv, write_jsonl
from ai201_project3_takemeter.validate_dataset import validate_rows


def test_clean_hn_text_removes_html() -> None:
    raw = "SQLite is fast.<p>See <a href='https://example.com'>this</a> benchmark."
    assert clean_hn_text(raw) == "SQLite is fast. See this benchmark."


def test_validate_rows_accepts_balanced_dataset() -> None:
    rows = []
    labels = ["evidence_based", "reasoned_opinion", "low_substance"]
    for index in range(210):
        rows.append({"text": f"example {index}", "label": labels[index % 3]})

    assert validate_rows(rows) == []


def test_validate_rows_rejects_missing_labels() -> None:
    rows = [{"text": "example", "label": ""} for _ in range(200)]

    errors = validate_rows(rows)

    assert any("missing labels" in error for error in errors)


def test_validate_rows_rejects_unreviewed_queue() -> None:
    rows = [
        {
            "text": f"example {index}",
            "label": "evidence_based",
            "review_status": "needs_human_review",
        }
        for index in range(200)
    ]

    errors = validate_rows(rows, allow_unreviewed=True)
    strict_errors = validate_rows(rows)

    assert not any("human review" in error for error in errors)
    assert any("human review" in error for error in strict_errors)


def test_jsonl_round_trip_preserves_text_metadata(tmp_path) -> None:
    path = tmp_path / "dataset.jsonl"
    rows = [
        {
            "text": 'Comma, quote " and newline\nstay inside one JSON object.',
            "label": "reasoned_opinion",
            "source": "https://news.ycombinator.com/item?id=1",
        }
    ]

    write_jsonl(rows, path)

    assert read_jsonl(path) == rows


def test_write_colab_csv_exports_minimal_columns(tmp_path) -> None:
    path = tmp_path / "colab.csv"
    rows = [
        {
            "text": "A detailed comment.",
            "label": "evidence_based",
            "source": "metadata stays out of Colab CSV",
        }
    ]

    write_colab_csv(rows, path)

    with path.open(newline="", encoding="utf-8") as file:
        exported = list(csv.DictReader(file))

    assert exported == [{"text": "A detailed comment.", "label": "evidence_based"}]
