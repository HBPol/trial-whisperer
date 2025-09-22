import json

from fastapi.testclient import TestClient

from app.main import app
from app.retrieval import search_client, trial_store
from app.routers import qa
from eval.eval import answer_exact_match

client = TestClient(app)


def _load_index() -> None:
    path = trial_store.get_trials_data_path()
    with path.open("r", encoding="utf-8") as f:
        search_client.clear_fallback_index()
        search_client._FAKE_INDEX = [json.loads(line) for line in f]


def test_ask_returns_answer_and_citations():
    _load_index()
    sample_id = search_client._FAKE_INDEX[0]["nct_id"]
    response = client.post(
        "/ask/", json={"query": "What is this study?", "nct_id": sample_id}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data.get("answer"), str)
    assert data["answer"]
    assert not data["answer"].startswith("[LLM error]")
    assert "citations" in data and isinstance(data["citations"], list)
    assert len(data["citations"]) >= 1
    citation = data["citations"][0]
    assert {"nct_id", "section", "text_snippet"} <= citation.keys()


def test_ask_requires_query():
    _load_index()
    sample_id = search_client._FAKE_INDEX[0]["nct_id"]
    for payload in (
        {"query": "", "nct_id": sample_id},
        {"nct_id": sample_id},
    ):
        response = client.post("/ask/", json=payload)
        assert response.status_code == 400


def test_ask_strips_citation_markers(monkeypatch):
    _load_index()
    sample_id = search_client._FAKE_INDEX[0]["nct_id"]

    raw_answer = (
        "Answer: Based on the provided context, the study enrolls 120 participants. (1)"
    )
    fake_citations = [
        {"nct_id": sample_id, "section": "Overview", "text": "Enrollment is 120."}
    ]

    def _fake_call_llm(query, chunks):
        return raw_answer, fake_citations

    monkeypatch.setattr(qa, "call_llm_with_citations", _fake_call_llm)

    response = client.post(
        "/ask/",
        json={"query": "How many participants are in the study?", "nct_id": sample_id},
    )
    assert response.status_code == 200
    data = response.json()
    expected_answer = "The study enrolls 120 participants."
    assert data["answer"].lower() == expected_answer.lower()
    assert "(1)" not in data["answer"]
    assert data["answer"].lower().startswith("the study")
    assert answer_exact_match(data["answer"], [expected_answer])
