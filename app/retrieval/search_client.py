from typing import List, Optional

# Swap this placeholder with Qdrant/Vertex client

_FAKE_INDEX: List[dict] = []


def retrieve_chunks(query: str, nct_id: Optional[str] = None, k: int = 8) -> List[dict]:
    # TODO: vector + BM25 hybrid. For demo, return first k
    results = [c for c in _FAKE_INDEX if (not nct_id or c.get("nct_id") == nct_id)]
    return results[:k]


def retrieve_criteria_for_trial(nct_id: str) -> dict:
    # TODO: fetch inclusion/exclusion split
    return {"inclusion": [], "exclusion": []}