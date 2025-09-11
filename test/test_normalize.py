import pytest
from pipeline.normalize import normalize


def test_normalize_standardizes_keys_and_types():
    records = [
        {"nct_id": "NCT0", "title": ["A study"], "condition": "Condition A"},
        {"nct_id": "NCT1", "title": "Another study", "condition": ["Condition B"]},
    ]

    expected = [
        {"nct_id": "NCT0", "title": "A study", "condition": ["Condition A"]},
        {"nct_id": "NCT1", "title": "Another study", "condition": ["Condition B"]},
    ]

    normalized = normalize(records)
    assert normalized == expected
