from typing import List, Optional

from qdrant_client import QdrantClient

from app.deps import get_settings

settings = get_settings()

# Swap this placeholder with a real Qdrant/Vertex client
_client: QdrantClient | None = None
if settings.retrieval_backend == "qdrant" and settings.qdrant_url:
    _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

_FAKE_INDEX: List[dict] = []


def retrieve_chunks(query: str, nct_id: Optional[str] = None, k: int = 8) -> List[dict]:
    if _client:
        # TODO: vector + BM25 hybrid search via Qdrant
        pass

    results = [c for c in _FAKE_INDEX if (not nct_id or c.get("nct_id") == nct_id)]
    return results[:k]


def retrieve_criteria_for_trial(nct_id: str) -> dict:
    if _client:
        # TODO: fetch inclusion/exclusion split from vector DB
        pass
    return {"inclusion": [], "exclusion": []}