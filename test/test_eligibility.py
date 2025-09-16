import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.retrieval import search_client
from app.routers import eligibility

client = TestClient(app)


def _age_based_stub(criteria, patient):
    """Simple helper that flags eligibility based on an age range."""

    inclusion = criteria.get("inclusion", []) if isinstance(criteria, dict) else []
    age_rule = next(
        (
            rule
            for rule in inclusion
            if isinstance(rule, str) and rule.lower().startswith("age")
        ),
        None,
    )

    reasons: list[str] = []
    eligible = True

    if age_rule is None:
        reasons.append("No age rule available to evaluate")
        return {"eligible": eligible, "reasons": reasons}

    numbers = [int(value) for value in re.findall(r"\d+", age_rule)]

    if hasattr(patient, "model_dump"):
        patient_data = patient.model_dump()
    elif isinstance(patient, dict):
        patient_data = patient
    else:
        patient_data = {}

    age = patient_data.get("age")
    if age is None or len(numbers) < 2:
        reasons.append(f"Unable to evaluate age against criterion: {age_rule}")
        return {"eligible": eligible, "reasons": reasons}

    lower, upper = numbers[:2]
    if lower <= age <= upper:
        reasons.append(f"Age {age} within allowed range ({age_rule})")
    else:
        eligible = False
        reasons.append(f"Age {age} violates eligibility criterion ({age_rule})")

    return {"eligible": eligible, "reasons": reasons}


def test_check_eligibility_success(monkeypatch):

    def mock_retrieve(nct_id):
        return ["criteria"]

    def mock_check(criteria, patient):
        return {"eligible": True, "reasons": []}

    monkeypatch.setattr(eligibility, "retrieve_criteria_for_trial", mock_retrieve)
    monkeypatch.setattr(eligibility, "check_eligibility", mock_check)
    response = client.post(
        "/check-eligibility/", json={"nct_id": "NCT1", "patient": {}}
    )
    assert response.status_code == 200
    assert response.json()["eligible"] is True


def test_check_eligibility_missing_criteria_returns_400(monkeypatch):
    monkeypatch.setattr(eligibility, "retrieve_criteria_for_trial", lambda nct_id: None)
    response = client.post(
        "/check-eligibility/", json={"nct_id": "NCT1", "patient": {}}
    )
    assert response.status_code == 400


def test_check_eligibility_in_range_age(monkeypatch):
    monkeypatch.setattr(eligibility, "check_eligibility", _age_based_stub)

    response = client.post(
        "/check-eligibility/",
        json={"nct_id": "NCT00000001", "patient": {"age": 30}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["eligible"] is True
    assert any("Age 18 to 65" in reason for reason in payload["reasons"])


def test_check_eligibility_out_of_range_age(monkeypatch):
    monkeypatch.setattr(eligibility, "check_eligibility", _age_based_stub)

    response = client.post(
        "/check-eligibility/",
        json={"nct_id": "NCT00000001", "patient": {"age": 70}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["eligible"] is False
    assert any("Age 18 to 65" in reason for reason in payload["reasons"])


def test_retrieve_criteria_returns_lists():
    path = Path(".data/processed/trials.jsonl")
    with path.open("r", encoding="utf-8") as f:
        search_client._FAKE_INDEX = [json.loads(line) for line in f]

    trial_counts: dict[str, dict[str, int]] = {}
    trial_with_both: str | None = None

    for chunk in search_client._FAKE_INDEX:
        section = chunk.get("section")
        if section not in {"eligibility.inclusion", "eligibility.exclusion"}:
            continue

        stats = trial_counts.setdefault(
            chunk.get("nct_id"), {"inclusion": 0, "exclusion": 0}
        )
        key = "inclusion" if section.endswith("inclusion") else "exclusion"
        stats[key] += 1
        if stats["inclusion"] and stats["exclusion"]:
            trial_with_both = chunk.get("nct_id")
            break

    assert trial_with_both is not None, "Expected a trial with both criteria sections"

    criteria = eligibility.retrieve_criteria_for_trial(trial_with_both)
    assert criteria is not None
    assert criteria["inclusion"], "Inclusion criteria should not be empty"
    assert criteria["exclusion"], "Exclusion criteria should not be empty"
