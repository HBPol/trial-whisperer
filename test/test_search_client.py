import json

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
