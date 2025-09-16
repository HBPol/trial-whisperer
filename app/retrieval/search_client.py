from typing import Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.deps import get_settings

from . import trial_store

settings = get_settings()

# Swap this placeholder with a real Qdrant/Vertex client
_client: QdrantClient | None = None
if settings.retrieval_backend == "qdrant" and settings.qdrant_url:
    _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

_FAKE_INDEX: List[dict] = []


def retrieve_chunks(query: str, nct_id: Optional[str] = None, k: int = 8) -> List[dict]:
    """Retrieve relevant text chunks for a query.

    When a Qdrant client is configured we issue a real search request.  During
    tests or local development the function falls back to an in-memory index
    populated via ``_FAKE_INDEX``.
    """

    if _client and settings.qdrant_collection:
        # Basic text search in Qdrant.  ``query_text`` performs on-the-fly
        # embedding using the configured text2vec model within Qdrant.
        query_filter = None
        if nct_id:
            query_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="nct_id", match=rest.MatchValue(value=nct_id)
                    )
                ]
            )

        try:
            results = _client.search(
                collection_name=settings.qdrant_collection,
                query_text=query,
                limit=k,
                query_filter=query_filter,
                with_payload=True,
            )
        except TypeError:
            # Some versions of the client may not support ``query_text``.
            # Try using ``text_search`` if available; otherwise fall back to
            # the in-memory index handled later.
            text_search = getattr(_client, "text_search", None)
            if text_search:
                results = text_search(
                    collection_name=settings.qdrant_collection,
                    query=query,
                    limit=k,
                    query_filter=query_filter,
                    with_payload=True,
                )
            else:
                results = []

        if results:
            return [
                {
                    "nct_id": r.payload.get("nct_id"),
                    "section": r.payload.get("section"),
                    "text": r.payload.get("text"),
                }
                for r in results
            ]

    results = [c for c in _FAKE_INDEX if (not nct_id or c.get("nct_id") == nct_id)]
    return results[:k]


def _collect_criteria(
    sections: Iterable[Tuple[Optional[str], Optional[str]]],
) -> dict | None:

    inclusion: List[str] = []
    exclusion: List[str] = []

    for section, text in sections:
        if not section or not isinstance(section, str):
            continue
        if text is None:
            continue
        if not isinstance(text, str):
            text = str(text)
        if not text:
            continue

        key = section.lower()
        if key.startswith("eligibility.inclusion"):
            inclusion.append(text)
        elif key.startswith("eligibility.exclusion"):
            exclusion.append(text)

    if not inclusion and not exclusion:
        return None

    return {"inclusion": inclusion, "exclusion": exclusion}


def retrieve_criteria_for_trial(nct_id: str) -> dict | None:
    trial = trial_store.get_trial_metadata(nct_id)
    if trial:
        criteria = _collect_criteria(trial.sections.items())
        if criteria:
            return criteria

    return _collect_criteria(
        (
            (chunk.get("section"), chunk.get("text"))
            for chunk in _FAKE_INDEX
            if chunk.get("nct_id") == nct_id
        )
    )
