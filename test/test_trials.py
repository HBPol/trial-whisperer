import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.routers import trials

client = TestClient(app)


def _load_trials() -> str:
    path = Path(".data/processed/trials.jsonl")
    trials.TRIALS.clear()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            trial = trials.TRIALS.setdefault(
                chunk["nct_id"], {"id": chunk["nct_id"], "sections": {}}
            )
            if chunk["section"] == "title":
                trial["title"] = chunk["text"]
            trial["sections"][chunk["section"]] = chunk["text"]
    return next(iter(trials.TRIALS))


def test_get_trial_returns_expected_structure():
    nct_id = _load_trials()
    response = client.get(f"/trial/{nct_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == nct_id
    assert "title" in data
    assert "sections" in data


def test_get_trial_unknown_returns_400():
    _load_trials()
    response = client.get("/trial/UNKNOWN")
    assert response.status_code == 400
