"""Utilities for describing the ingested clinical trial dataset."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union

import tomli

from .retrieval.trial_store import TRIALS_DATA_ENV_VAR, get_trials_data_path

CONFIG_PATH = Path("config/appsettings.toml")
DEFAULT_CONFIG_FALLBACK = Path("config/appsettings.example.toml")
FilterValue = Union[str, List[str]]


def _load_config() -> Dict[str, Any]:
    """Return the application configuration from ``appsettings.toml`` if present."""

    for path in (CONFIG_PATH, DEFAULT_CONFIG_FALLBACK):
        if not path.exists():
            continue
        try:
            with path.open("rb") as handle:
                return tomli.load(handle)
        except (tomli.TOMLDecodeError, OSError):
            continue
    return {}


def _normalise_query_terms(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    text = str(raw_value).strip()
    if not text:
        return []
    return [text]


def _normalise_filter_value(raw_value: Any) -> FilterValue:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return str(raw_value)


def _parse_max_studies(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, (int, float)):
        return int(raw_value)
    if isinstance(raw_value, str):
        try:
            return int(raw_value.strip())
        except ValueError:
            return None
    return None


def _count_unique_trials(data_path: Path) -> int:
    if not data_path.exists():
        return 0

    unique_ids: set[str] = set()
    with data_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            nct_id = payload.get("nct_id")
            if isinstance(nct_id, str) and nct_id.strip():
                unique_ids.add(nct_id.strip().upper())
    return len(unique_ids)


def _resolve_trials_path() -> Path:
    override = os.getenv(TRIALS_DATA_ENV_VAR)
    if override:
        return Path(override)
    return get_trials_data_path()


def _get_last_updated_timestamp(data_path: Path) -> str | None:
    if not data_path.exists():
        return None
    try:
        mtime = data_path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def build_ingestion_summary() -> Dict[str, Any]:
    """Return a serialisable summary of the ingested trial corpus."""

    config = _load_config()
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    api_cfg = data_cfg.get("api", {}) if isinstance(data_cfg, dict) else {}
    params_cfg = api_cfg.get("params", {}) if isinstance(api_cfg, dict) else {}

    query_terms = _normalise_query_terms(params_cfg.get("query.term"))
    filters: Dict[str, FilterValue] = {}
    for key, value in params_cfg.items():
        if key == "query.term":
            continue
        filters[key] = _normalise_filter_value(value)

    trials_path = _resolve_trials_path()
    study_count = _count_unique_trials(trials_path)

    summary: Dict[str, Any] = {
        "study_count": study_count,
        "query_terms": query_terms,
        "filters": filters,
        "max_studies": _parse_max_studies(api_cfg.get("max_studies")),
        "last_updated": _get_last_updated_timestamp(trials_path),
    }
    return summary
