"""Utilities for breaking trial records into smaller text chunks.

The real application indexes pieces of trial metadata – such as eligibility
criteria – into a vector database.  To keep the example project lightweight
we only implement a small portion of that behaviour here.  The
``chunk_sections`` function accepts a parsed trial record and emits a list of
documents ready for indexing.  Each document contains the ``nct_id`` of the
source trial, the name of the section from which the text originated and the
text itself.  Long pieces of text are split into fixed size chunks so that
each chunk contains at most ``target_tokens`` words.  A deliberately simple
tokenizer based on :py:meth:`str.split` is used to divide the text on
whitespace; more sophisticated approaches can be plugged in later if needed.
"""

from __future__ import annotations

from typing import Dict, Iterable, List


def _iter_sections(record: Dict[str, object]) -> Iterable[tuple[str, str]]:
    """Yield flattened section labels and their text.

    Only a very small subset of the ClinicalTrials.gov schema is supported
    which is sufficient for the unit tests.  Top level string fields, lists of
    strings and one level of nested dictionaries are handled.
    """

    for key, value in record.items():
        if key == "nct_id" or value in (None, ""):
            continue

        if isinstance(value, str):
            yield key, value

        elif isinstance(value, list):
            text = " ".join(str(v) for v in value if v)
            if text:
                yield key, text

        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, list):
                    text = " ".join(str(v) for v in sub_value if v)
                else:
                    text = str(sub_value) if sub_value else ""
                if text:
                    yield f"{key}.{sub_key}", text


def chunk_sections(
    record: Dict[str, object], target_tokens: int = 700
) -> List[Dict[str, str]]:
    """Split text fields of a trial record into smaller chunks.

    Parameters
    ----------
    record:
        Parsed trial record containing an ``nct_id`` and various text sections.
    target_tokens:
        Maximum number of whitespace separated tokens allowed in each chunk.

    Returns
    -------
    list[dict]
        Each dictionary contains ``text``, ``nct_id`` and ``section`` keys.
    """

    nct_id = record.get("nct_id")
    chunks: List[Dict[str, str]] = []

    for section, text in _iter_sections(record):
        # Simple whitespace tokenizer; sufficient for the unit tests
        tokens = text.split()
        for start in range(0, len(tokens), target_tokens):
            piece = " ".join(tokens[start : start + target_tokens])
            chunks.append({"text": piece, "nct_id": nct_id, "section": section})

    return chunks


__all__ = ["chunk_sections"]
