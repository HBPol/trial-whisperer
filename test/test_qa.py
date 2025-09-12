import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.retrieval import search_client
from app.routers import qa

client = TestClient(app)


def _load_index() -> None:
    path = Path(".data/processed/trials.jsonl")
    with path.open("r", encoding="utf-8") as f:
        search_client._FAKE_INDEX = [json.loads(line) for line in f]


def test_ask_returns_answer():
    _load_index()
    sample_id = search_client._FAKE_INDEX[0]["nct_id"]
    response = client.post(
        "/ask/", json={"query": "What is this study?", "nct_id": sample_id}
    )
    assert response.status_code == 200
    assert "answer" in response.json()


def test_ask_requires_nonempty_query():
    _load_index()
    sample_id = search_client._FAKE_INDEX[0]["nct_id"]
    response = client.post("/ask/", json={"query": "", "nct_id": sample_id})
    assert response.status_code == 400
