from ai201_project3_takemeter.labels import LABEL_MAP, LABEL_NAMES


def test_label_map_is_stable() -> None:
    assert LABEL_NAMES == (
        "evidence_based",
        "reasoned_opinion",
        "low_substance",
    )
    assert LABEL_MAP == {
        "evidence_based": 0,
        "reasoned_opinion": 1,
        "low_substance": 2,
    }
