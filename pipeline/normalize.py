"""Utilities for standardizing parsed clinical trial records.

This module exposes a :func:`normalize` function which reshapes dictionaries
produced by the XML parsing step into a consistent schema.  The normalizer is
purposefully small – it only knows about a few fields that the tests cover –
but it can easily be extended as new fields are required.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _normalize_title(value: Any) -> str | None:
    """Coerce the ``title`` field into a plain string.

    The parser may represent a title as a list (because many XML fields can
    occur multiple times).  When a list is encountered we keep the first
    non-empty value.  If the value is already a string it is returned as-is.
    """

    if isinstance(value, list):
        return value[0] if value else None
    if isinstance(value, str) or value is None:
        return value
    # Fallback: convert any other types to string for consistency
    return str(value)


def _normalize_condition(value: Any) -> List[str]:
    """Ensure the ``condition`` field is always a list of strings."""

    if value is None:
        return []
    if isinstance(value, list):
        # Filter out ``None`` values and cast everything to string
        return [str(v) for v in value if v is not None]
    # Single string (or other primitive) -> list with one string
    return [str(value)]


def normalize(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean fields and standardize schema of parsed records.

    Parameters
    ----------
    records:
        List of dictionaries produced by :mod:`pipeline.parse_xml`.

    Returns
    -------
    list[dict]
        A new list with normalized records.
    """

    normalized: List[Dict[str, Any]] = []
    for record in records:
        item = dict(record)  # shallow copy so we don't mutate the input

        # Coerce specific fields to their expected types
        if "title" in item:
            item["title"] = _normalize_title(item["title"])

        if "condition" in item:
            item["condition"] = _normalize_condition(item["condition"])

        normalized.append(item)

    return normalized
