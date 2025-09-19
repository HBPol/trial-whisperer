import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.retrieval import search_client, trial_store

client = TestClient(app)


@pytest.fixture()
def offline_trials():
    nct_id = "NCTOFFLINE001"
    dataset = [
        {"nct_id": nct_id, "section": "title", "text": "Offline support trial"},
        {
            "nct_id": nct_id,
            "section": "summary",
            "text": "This offline retrieval trial explores hypertension therapy.",
        },
        {
            "nct_id": nct_id,
            "section": "eligibility.inclusion",
            "text": "Age 18 to 65",
        },
        {
            "nct_id": nct_id,
            "section": "eligibility.exclusion",
            "text": "History of stroke",
        },
    ]

    data_path = trial_store.get_trials_data_path()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    original_content = (
        data_path.read_text(encoding="utf-8") if data_path.exists() else None
    )
    data_path.write_text(
        "\n".join(json.dumps(item) for item in dataset), encoding="utf-8"
    )

    original_client = search_client._client
    original_index = search_client._FAKE_INDEX
    original_initialized = getattr(search_client, "_FALLBACK_INDEX_INITIALIZED", False)
    original_backend = search_client.settings.retrieval_backend
    original_url = search_client.settings.qdrant_url
    original_collection = search_client.settings.qdrant_collection

    search_client._client = None
    search_client.clear_fallback_index()
    search_client.settings.retrieval_backend = None
    search_client.settings.qdrant_url = None
    search_client.settings.qdrant_collection = None
    trial_store.clear_trials_cache()

    try:
        yield {"nct_id": nct_id, "summary": dataset[1]["text"]}
    finally:
        if original_content is None:
            data_path.unlink(missing_ok=True)
        else:
            data_path.write_text(original_content, encoding="utf-8")

        trial_store.clear_trials_cache()
        search_client._client = original_client
        search_client._FAKE_INDEX = original_index
        search_client._FALLBACK_INDEX_INITIALIZED = original_initialized
        search_client.settings.retrieval_backend = original_backend
        search_client.settings.qdrant_url = original_url
        search_client.settings.qdrant_collection = original_collection


def test_offline_ask_returns_passages(offline_trials):
    nct_id = offline_trials["nct_id"]
    response = client.post(
        "/ask/",
        json={"query": "hypertension therapy offline retrieval", "nct_id": nct_id},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"]
    assert all(citation["nct_id"] == nct_id for citation in payload["citations"])
    assert any(
        "offline retrieval" in citation["text_snippet"].lower()
        for citation in payload["citations"]
    )


def test_offline_trial_endpoint_uses_local_data(offline_trials):
    nct_id = offline_trials["nct_id"]
    response = client.get(f"/trial/{nct_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == nct_id
    assert payload["sections"]["summary"] == offline_trials["summary"]


def test_offline_check_eligibility(offline_trials):
    nct_id = offline_trials["nct_id"]
    response = client.post(
        "/check-eligibility/",
        json={"nct_id": nct_id, "patient": {"age": 30}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["eligible"] is True
    assert payload["reasons"] == []
