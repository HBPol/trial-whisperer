import json
from types import SimpleNamespace

from qdrant_client.http import exceptions as rest_exceptions

from app.retrieval import search_client, trial_store


def test_retrieve_chunks_falls_back_when_qdrant_empty(monkeypatch, tmp_path):
    """Fallback index should supply chunks when Qdrant has no matches."""

    search_client.clear_fallback_index()
    trial_store.clear_trials_cache()

    try:
        data_path = tmp_path / "trials.jsonl"
        payload = {
            "nct_id": "NCT04439149",
            "section": "eligibility.inclusion",
            "text": "Please sumarize inclusion criteria for participants.",
        }
        data_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        monkeypatch.setenv(trial_store.TRIALS_DATA_ENV_VAR, str(data_path))

        class EmptyQdrantClient:
            def search(self, *args, **kwargs):
                return []

        monkeypatch.setattr(
            search_client, "_client", EmptyQdrantClient(), raising=False
        )
        monkeypatch.setattr(
            search_client.settings, "qdrant_collection", "unit-test", raising=False
        )

        results = search_client.retrieve_chunks(
            "sumarize inclusion criteria", "NCT04439149"
        )

        assert results == [
            {
                "nct_id": "NCT04439149",
                "section": "eligibility.inclusion",
                "text": "Please sumarize inclusion criteria for participants.",
            }
        ]
    finally:
        search_client.clear_fallback_index()
        trial_store.clear_trials_cache()


def test_retrieve_chunks_recovers_when_payload_index_missing(monkeypatch):
    """Missing payload indexes should trigger an unfiltered retry."""

    search_client.clear_fallback_index()

    class PayloadIndexErrorClient:
        def __init__(self):
            self.calls: list[dict] = []

        def search(self, *args, **kwargs):
            self.calls.append({"args": args, "kwargs": dict(kwargs)})
            if kwargs.get("query_filter") is not None:
                raise rest_exceptions.UnexpectedResponse(
                    status_code=400,
                    reason_phrase="Bad Request",
                    content=b'{"status": {"error": "Payload index is missing"}}',
                    headers={},
                )

            return [
                SimpleNamespace(
                    payload={
                        "nct_id": "NCT01234567",
                        "section": "eligibility.inclusion",
                        "text": "matched",
                    }
                ),
                SimpleNamespace(
                    payload={
                        "nct_id": "NCT76543210",
                        "section": "eligibility.exclusion",
                        "text": "ignored",
                    }
                ),
            ]

    client = PayloadIndexErrorClient()
    monkeypatch.setattr(search_client, "_client", client, raising=False)
    monkeypatch.setattr(
        search_client.settings, "qdrant_collection", "unit-test", raising=False
    )

    results = search_client.retrieve_chunks("query", nct_id="NCT01234567", k=3)

    assert [result["nct_id"] for result in results] == ["NCT01234567"]
    assert len(client.calls) == 2
    assert client.calls[0]["kwargs"].get("query_filter") is not None
    assert client.calls[1]["kwargs"].get("query_filter") is None
