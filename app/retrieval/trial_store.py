"""Utility helpers for accessing processed trial metadata."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

from app.models.schemas import TrialMetadata

DEFAULT_TRIALS_PATH = Path(".data/processed/trials.jsonl")


def _normalise_path(data_path: Path | str) -> str:
    if isinstance(data_path, Path):
        return str(data_path)
    return data_path


def normalize_section_entry(section: object, text: object) -> Tuple[str, str] | None:
    """Return a ``(section, text)`` tuple when both values are usable."""

    if not section or not isinstance(section, str):
        return None
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return None
    return section, text


def _build_index(data_path: str) -> Dict[str, TrialMetadata]:
    path = Path(data_path)
    if not path.exists():
        return {}

    trials: Dict[str, TrialMetadata] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            nct_id = chunk.get("nct_id")
            if not isinstance(nct_id, str):
                continue
            normalized = normalize_section_entry(
                chunk.get("section"), chunk.get("text")
            )
            if normalized is None:
                continue
            section, text = normalized

            trial = trials.get(nct_id)
            if trial is None:
                trial = TrialMetadata(id=nct_id, title=None, sections={})
                trials[nct_id] = trial

            trial.sections[section] = text
            if section == "title":
                trial.title = text

    return trials


@lru_cache(maxsize=1)
def load_trials_index(
    data_path: str = str(DEFAULT_TRIALS_PATH),
) -> Dict[str, TrialMetadata]:
    """Load and cache trial metadata keyed by ``nct_id``."""

    return _build_index(data_path)


def get_trial_metadata(
    nct_id: str, *, data_path: Path | str = DEFAULT_TRIALS_PATH
) -> Optional[TrialMetadata]:
    """Return metadata for a specific trial if available."""

    index = load_trials_index(_normalise_path(data_path))
    trial = index.get(nct_id)
    if trial is None:
        return None
    return trial.model_copy(deep=True)


def clear_trials_cache() -> None:
    """Clear the cached trial metadata, forcing a reload on next access."""

    load_trials_index.cache_clear()
