from ai201_project3_takemeter.collect_hn import clean_hn_text
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
