import json
import logging
import re
from collections import Counter
from typing import Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import exceptions as rest_exceptions
from qdrant_client.http import models as rest

from app.deps import get_settings

logger = logging.getLogger(__name__)

from . import trial_store

settings = get_settings()

# Swap this placeholder with a real Qdrant/Vertex client
_client: QdrantClient | None = None
if settings.retrieval_backend == "qdrant" and settings.qdrant_url:
    _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

_FAKE_INDEX: List[dict] = []
_FALLBACK_INDEX_INITIALIZED = False
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


_QDRANT_FILTER_EXCEPTIONS: tuple[type[Exception], ...]
try:
    _QDRANT_FILTER_EXCEPTIONS = (
        rest_exceptions.UnexpectedResponse,
        rest_exceptions.ResponseHandlingException,
    )
except AttributeError:  # pragma: no cover - defensive for older clients
    _QDRANT_FILTER_EXCEPTIONS = (rest_exceptions.UnexpectedResponse,)

_QDRANT_SEARCH_EXCEPTIONS: tuple[type[Exception], ...]
try:
    _QDRANT_SEARCH_EXCEPTIONS = (
        rest_exceptions.ApiException,
        rest_exceptions.ResponseHandlingException,
    )
except AttributeError:  # pragma: no cover - defensive for older clients
    _QDRANT_SEARCH_EXCEPTIONS = (rest_exceptions.ResponseHandlingException,)


def _decode_bytes(value: bytes | None) -> str | None:
    if value is None:
        return None
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return value.decode("utf-8", errors="ignore")


def _log_qdrant_error(exc: Exception, *, search_kind: str) -> None:
    status_code = getattr(exc, "status_code", None)
    reason = getattr(exc, "reason_phrase", None)

    content = getattr(exc, "content", None)
    if isinstance(content, bytes):
        content = _decode_bytes(content)
    elif content is None:
        content = getattr(exc, "response_content", None)
        if isinstance(content, bytes):
            content = _decode_bytes(content)

    message = content or reason or str(exc)

    logger.error(
        "Qdrant %s search failed (status=%s, reason=%s): %s",
        search_kind,
        status_code if status_code is not None else "unknown",
        reason if reason else "unknown",
        message,
    )


def _payload_matches_nct_id(payload: dict | None, nct_id: str | None) -> bool:
    if not nct_id:
        return True
    if not isinstance(payload, dict):
        return False
    return payload.get("nct_id") == nct_id


def _is_payload_index_error(exc: Exception) -> bool:
    """Detect whether the raised exception is caused by a payload index error."""

    message_parts: List[str] = []

    content = getattr(exc, "content", None)
    if isinstance(content, bytes):
        try:
            message_parts.append(content.decode("utf-8"))
        except UnicodeDecodeError:
            message_parts.append(content.decode("utf-8", errors="ignore"))
    elif content:
        message_parts.append(str(content))

    message_parts.append(str(exc))
    combined = " ".join(part for part in message_parts if part)
    return "payload index" in combined.lower()


def _build_query_filter(nct_id: Optional[str]) -> rest.Filter | None:
    if not nct_id:
        return None
    return rest.Filter(
        must=[rest.FieldCondition(key="nct_id", match=rest.MatchValue(value=nct_id))]
    )


def _filter_points_by_nct_id(
    points: Iterable[rest.ScoredPoint],
    nct_id: Optional[str],
) -> List[rest.ScoredPoint]:
    if not nct_id:
        return list(points)
    return [point for point in points if _payload_matches_nct_id(point.payload, nct_id)]


def _execute_qdrant_search(
    search_callable,
    *,
    kwargs: dict,
    nct_id: Optional[str],
) -> List[rest.ScoredPoint]:
    query_filter = kwargs.get("query_filter")

    try:
        return list(search_callable(**kwargs))
    except _QDRANT_FILTER_EXCEPTIONS as exc:
        if not query_filter or not _is_payload_index_error(exc):
            raise

        retry_kwargs = dict(kwargs)
        retry_kwargs["query_filter"] = None
        results = list(search_callable(**retry_kwargs))
        return _filter_points_by_nct_id(results, nct_id)


def _search_qdrant_with_vector(
    vector: Iterable[float],
    *,
    nct_id: Optional[str],
    k: int,
) -> List[rest.ScoredPoint]:
    if not _client or not settings.qdrant_collection:
        return []

    kwargs = {
        "collection_name": settings.qdrant_collection,
        "query_vector": vector,
        "limit": k,
        "query_filter": _build_query_filter(nct_id),
        "with_payload": True,
    }

    try:
        return _execute_qdrant_search(_client.search, kwargs=kwargs, nct_id=nct_id)
    except _QDRANT_SEARCH_EXCEPTIONS as exc:
        _log_qdrant_error(exc, search_kind="vector")
        return []


def _search_qdrant_with_text(
    query: str,
    *,
    nct_id: Optional[str],
    k: int,
) -> List[rest.ScoredPoint]:
    if not _client or not settings.qdrant_collection:
        return []

    kwargs = {
        "collection_name": settings.qdrant_collection,
        "query_text": query,
        "limit": k,
        "query_filter": _build_query_filter(nct_id),
        "with_payload": True,
    }

    try:
        return _execute_qdrant_search(_client.search, kwargs=kwargs, nct_id=nct_id)
    except TypeError:
        text_search = getattr(_client, "text_search", None)
        if not text_search:
            return []

        text_kwargs = dict(kwargs)
        text_kwargs.pop("query_text")
        text_kwargs["query"] = query
        try:
            return _execute_qdrant_search(
                text_search, kwargs=text_kwargs, nct_id=nct_id
            )
        except _QDRANT_SEARCH_EXCEPTIONS as exc:
            _log_qdrant_error(exc, search_kind="text")
            return []
    except _QDRANT_SEARCH_EXCEPTIONS as exc:
        _log_qdrant_error(exc, search_kind="text")
        return []


def _ensure_fake_index_loaded(*, force: bool = False) -> None:
    """Populate ``_FAKE_INDEX`` from the processed trials dataset."""

    global _FALLBACK_INDEX_INITIALIZED

    if _client and not force:
        return

    if _FAKE_INDEX:
        _FALLBACK_INDEX_INITIALIZED = True
        return

    if _FALLBACK_INDEX_INITIALIZED and not force:
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


def _extract_scroll_points(response) -> List[rest.ScoredPoint]:
    """Return a list of points from the various scroll response formats."""

    if isinstance(response, tuple):
        points = response[0]
    elif hasattr(response, "points"):
        points = getattr(response, "points")
    else:
        points = response

    if points is None:
        return []
    return list(points)


def _scroll_qdrant_points(nct_id: str) -> List[rest.ScoredPoint]:
    if not _client or not settings.qdrant_collection or not nct_id:
        return []

    base_kwargs = {
        "collection_name": settings.qdrant_collection,
        "limit": 128,
        "with_payload": True,
        "with_vectors": False,
    }
    filter_obj = _build_query_filter(nct_id)
    scroll_kwargs = dict(base_kwargs)
    if filter_obj:
        scroll_kwargs["scroll_filter"] = filter_obj

    def _invoke_scroll(kwargs):
        try:
            return _client.scroll(**kwargs)
        except TypeError:
            adjusted = dict(kwargs)
            scroll_filter = adjusted.pop("scroll_filter", None)
            if scroll_filter is not None:
                adjusted["filter"] = scroll_filter
            return _client.scroll(**adjusted)

    try:
        response = _invoke_scroll(scroll_kwargs)
    except _QDRANT_FILTER_EXCEPTIONS as exc:
        if not filter_obj or not _is_payload_index_error(exc):
            _log_qdrant_error(exc, search_kind="scroll")
            return []
        retry_kwargs = dict(base_kwargs)
        try:
            response = _invoke_scroll(retry_kwargs)
        except _QDRANT_FILTER_EXCEPTIONS as retry_exc:
            _log_qdrant_error(retry_exc, search_kind="scroll")
            return []
        points = _extract_scroll_points(response)
        return _filter_points_by_nct_id(points, nct_id)
    except _QDRANT_SEARCH_EXCEPTIONS as exc:
        _log_qdrant_error(exc, search_kind="scroll")
        return []

    return _extract_scroll_points(response)


def _fetch_sections_from_remote(nct_id: str) -> List[Tuple[str, str]]:
    points = _scroll_qdrant_points(nct_id)
    sections: List[Tuple[str, str]] = []

    for point in points:
        payload = getattr(point, "payload", None)
        if not isinstance(payload, dict):
            continue
        if payload.get("nct_id") != nct_id:
            continue
        normalized = trial_store.normalize_section_entry(
            payload.get("section"), payload.get("text")
        )
        if normalized is None:
            continue
        sections.append(normalized)

    return sections


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

    _ensure_fake_index_loaded(force=_client is not None)

    candidates = [
        chunk for chunk in _FAKE_INDEX if (not nct_id or chunk.get("nct_id") == nct_id)
    ]

    if candidates:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return candidates[:k]

        scored: List[Tuple[float, int, dict]] = []
        for idx, chunk in enumerate(candidates):
            score = _score_chunk(query_tokens, chunk)
            scored.append((score, idx, chunk))

        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        return [entry[2] for entry in scored[:k]]

    if not _client or not settings.qdrant_collection:
        return []

    results = _search_qdrant_with_text(query, nct_id=nct_id, k=k)

    return [
        {
            "nct_id": r.payload.get("nct_id"),
            "section": r.payload.get("section"),
            "text": r.payload.get("text"),
        }
        for r in results
    ]


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
    sections = [
        (chunk.get("section"), chunk.get("text"))
        for chunk in _FAKE_INDEX
        if chunk.get("nct_id") == nct_id
    ]

    if not sections:
        _ensure_fake_index_loaded(force=True)
        sections = [
            (chunk.get("section"), chunk.get("text"))
            for chunk in _FAKE_INDEX
            if chunk.get("nct_id") == nct_id
        ]

    if not sections:
        remote_sections = _fetch_sections_from_remote(nct_id)
        if remote_sections:
            return _collect_criteria(remote_sections)
        return None

    return _collect_criteria(sections)
