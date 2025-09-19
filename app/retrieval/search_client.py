import json
import re
from collections import Counter
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
_FALLBACK_INDEX_INITIALIZED = False
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _ensure_fake_index_loaded() -> None:
    """Populate ``_FAKE_INDEX`` from the processed trials dataset."""

    global _FALLBACK_INDEX_INITIALIZED

    if _client:
        return

    if _FAKE_INDEX:
        _FALLBACK_INDEX_INITIALIZED = True
        return

    if _FALLBACK_INDEX_INITIALIZED:
        return

    _FALLBACK_INDEX_INITIALIZED = True

    data_path = trial_store.get_trials_data_path()
    if not data_path.exists():
        return

    try:
        with data_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                nct_id = payload.get("nct_id")
                if not isinstance(nct_id, str):
                    continue

                normalized = trial_store.normalize_section_entry(
                    payload.get("section"), payload.get("text")
                )
                if normalized is None:
                    continue

                section, text = normalized
                _FAKE_INDEX.append({"nct_id": nct_id, "section": section, "text": text})
    except OSError:
        return


def clear_fallback_index() -> None:
    """Reset the in-memory fallback index used during tests/offline mode."""

    global _FAKE_INDEX, _FALLBACK_INDEX_INITIALIZED
    _FAKE_INDEX = []
    _FALLBACK_INDEX_INITIALIZED = False


def _tokenize(text: str | None) -> List[str]:
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


def _score_chunk(query_tokens: List[str], chunk: dict) -> float:
    if not query_tokens:
        return 0.0
    combined = " ".join(filter(None, [chunk.get("section"), chunk.get("text")]))
    tokens = _tokenize(combined)
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    return float(sum(counts[token] for token in query_tokens))


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

    _ensure_fake_index_loaded()

    candidates = [
        chunk for chunk in _FAKE_INDEX if (not nct_id or chunk.get("nct_id") == nct_id)
    ]

    if not candidates:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return candidates[:k]

    scored: List[Tuple[float, int, dict]] = []
    for idx, chunk in enumerate(candidates):
        score = _score_chunk(query_tokens, chunk)
        scored.append((score, idx, chunk))

    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    return [entry[2] for entry in scored[:k]]


def _collect_criteria(
    sections: Iterable[Tuple[Optional[str], Optional[str]]],
) -> dict | None:

    inclusion: List[str] = []
    exclusion: List[str] = []

    for section, text in sections:
        normalized = trial_store.normalize_section_entry(section, text)
        if normalized is None:
            continue
        section_key, text_value = normalized

        key = section.lower()
        if key.startswith("eligibility.inclusion"):
            inclusion.append(text_value)
        elif key.startswith("eligibility.exclusion"):
            exclusion.append(text_value)

    if not inclusion and not exclusion:
        return None

    return {"inclusion": inclusion, "exclusion": exclusion}


def retrieve_criteria_for_trial(nct_id: str) -> dict | None:
    trial = trial_store.get_trial_metadata(nct_id)
    if trial:
        criteria = _collect_criteria(trial.sections.items())
        if criteria:
            return criteria
    _ensure_fake_index_loaded()
    return _collect_criteria(
        (
            (chunk.get("section"), chunk.get("text"))
            for chunk in _FAKE_INDEX
            if chunk.get("nct_id") == nct_id
        )
    )
