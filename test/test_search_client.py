import json
import logging
from types import SimpleNamespace

from qdrant_client.http import exceptions as rest_exceptions

from app.retrieval import search_client, trial_store


def _build_qdrant_error(status_code: int = 500, message: str = "boom"):
    return rest_exceptions.UnexpectedResponse(
        status_code=status_code,
        reason_phrase="Error",
        content=message.encode("utf-8"),
        headers={},
    )


def test_search_qdrant_with_vector_logs_error(monkeypatch, caplog):
    class ErrorClient:
        def search(self, *args, **kwargs):
            raise _build_qdrant_error()

    monkeypatch.setattr(search_client, "_client", ErrorClient(), raising=False)
    monkeypatch.setattr(
        search_client.settings, "qdrant_collection", "unit-test", raising=False
    )

    with caplog.at_level(logging.ERROR):
        results = search_client._search_qdrant_with_vector([0.1], nct_id=None, k=3)

    assert results == []
    assert any(
        "Qdrant vector search failed" in record.message for record in caplog.records
    )


def test_search_qdrant_with_text_logs_error(monkeypatch, caplog):
    class ErrorClient:
        def search(self, *args, **kwargs):
            raise _build_qdrant_error(status_code=429, message="rate limit")

    monkeypatch.setattr(search_client, "_client", ErrorClient(), raising=False)
    monkeypatch.setattr(
        search_client.settings, "qdrant_collection", "unit-test", raising=False
    )

    with caplog.at_level(logging.ERROR):
        results = search_client._search_qdrant_with_text("query", nct_id=None, k=5)

    assert results == []
    assert any(
        "Qdrant text search failed" in record.message for record in caplog.records
    )


def test_retrieve_chunks_returns_local_without_qdrant(monkeypatch, tmp_path):
    """Local chunks should satisfy the query without consulting Qdrant."""

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

        monkeypatch.setattr(search_client, "_client", SimpleNamespace(), raising=False)
        monkeypatch.setattr(
            search_client.settings, "qdrant_collection", "unit-test", raising=False
        )

        def _unexpected(*args, **kwargs):  # pragma: no cover - defensive
            raise AssertionError("Qdrant search should not be invoked")

        monkeypatch.setattr(search_client, "_search_qdrant_with_text", _unexpected)

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
    monkeypatch.setattr(search_client, "_embed_query", lambda _query: [0.1, 0.2, 0.3])

    results = search_client.retrieve_chunks("query", nct_id="NCT01234567", k=3)

    assert [result["nct_id"] for result in results] == ["NCT01234567"]
    assert len(client.calls) == 2
    assert client.calls[0]["kwargs"].get("query_filter") is not None
    assert client.calls[1]["kwargs"].get("query_filter") is None


def test_retrieve_chunks_queries_qdrant_when_local_missing(monkeypatch, tmp_path):
    """Qdrant should be consulted when the fallback index lacks the trial."""

    search_client.clear_fallback_index()
    trial_store.clear_trials_cache()

    try:
        data_path = tmp_path / "trials.jsonl"
        payload = {
            "nct_id": "NCT00000001",
            "section": "summary",
            "text": "Local trial unrelated to query.",
        }
        data_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        monkeypatch.setenv(trial_store.TRIALS_DATA_ENV_VAR, str(data_path))
        monkeypatch.setattr(search_client, "_client", SimpleNamespace(), raising=False)
        monkeypatch.setattr(
            search_client.settings, "qdrant_collection", "unit-test", raising=False
        )
        monkeypatch.setattr(
            search_client, "_embed_query", lambda _query: [0.4, 0.5, 0.6]
        )

        calls: list[dict] = []

        def _vector_miss(*args, **kwargs):
            return []

        def _fake_search(query, *, nct_id, k):
            calls.append({"query": query, "nct_id": nct_id, "k": k})
            return [
                SimpleNamespace(
                    payload={
                        "nct_id": "NCT99999999",
                        "section": "summary",
                        "text": "Remote hit",
                    }
                )
            ]

        monkeypatch.setattr(search_client, "_search_qdrant_with_vector", _vector_miss)
        monkeypatch.setattr(search_client, "_search_qdrant_with_text", _fake_search)

        results = search_client.retrieve_chunks("remote query", "NCT99999999", k=5)

        assert len(calls) == 1
        assert calls[0] == {"query": "remote query", "nct_id": "NCT99999999", "k": 5}
        assert results == [
            {
                "nct_id": "NCT99999999",
                "section": "summary",
                "text": "Remote hit",
            }
        ]
    finally:
        search_client.clear_fallback_index()
        trial_store.clear_trials_cache()


def test_retrieve_chunks_embeds_query_for_remote_search(monkeypatch):
    """When local data is absent the query should be embedded before searching."""

    search_client.clear_fallback_index()
    trial_store.clear_trials_cache()

    try:
        monkeypatch.delenv(trial_store.TRIALS_DATA_ENV_VAR, raising=False)
        monkeypatch.setattr(search_client, "_client", SimpleNamespace(), raising=False)
        monkeypatch.setattr(
            search_client.settings, "qdrant_collection", "unit-test", raising=False
        )

        embed_calls: list[str] = []

        def _fake_embed(query: str) -> list[float]:
            embed_calls.append(query)
            return [0.1, 0.2, 0.3]

        vector_calls: list[dict] = []

        def _fake_vector_search(vector, *, nct_id, k):
            vector_calls.append({"vector": vector, "nct_id": nct_id, "k": k})
            return [
                SimpleNamespace(
                    payload={
                        "nct_id": "NCT12345678",
                        "section": "summary",
                        "text": "Embedded hit",
                    }
                )
            ]

        def _unexpected(*args, **kwargs):  # pragma: no cover - defensive
            raise AssertionError(
                "Text search fallback should not be invoked when vector results exist"
            )

        monkeypatch.setattr(search_client, "_embed_query", _fake_embed)
        monkeypatch.setattr(
            search_client, "_search_qdrant_with_vector", _fake_vector_search
        )
        monkeypatch.setattr(search_client, "_search_qdrant_with_text", _unexpected)

        results = search_client.retrieve_chunks("explain outcomes", "NCT12345678", k=4)

        assert embed_calls == ["explain outcomes"]
        assert vector_calls == [
            {"vector": [0.1, 0.2, 0.3], "nct_id": "NCT12345678", "k": 4}
        ]
        assert results == [
            {
                "nct_id": "NCT12345678",
                "section": "summary",
                "text": "Embedded hit",
            }
        ]
    finally:
        search_client.clear_fallback_index()
        trial_store.clear_trials_cache()
