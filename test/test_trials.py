import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.retrieval import trial_store

client = TestClient(app)


def _sample_trial_id() -> str:
    trial_store.clear_trials_cache()
    path = Path(".data/processed/trials.jsonl")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            nct_id = chunk.get("nct_id")
            if nct_id:
                return nct_id
    raise AssertionError("No trial data available for tests")


def test_get_trial_returns_expected_structure():
    nct_id = _sample_trial_id()
    response = client.get(f"/trial/{nct_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == nct_id
    assert "title" in data
    assert "sections" in data


def test_get_trial_unknown_returns_400():
    trial_store.clear_trials_cache()
    response = client.get("/trial/UNKNOWN")
    assert response.status_code == 400
