"""Download and prepare trial data from the ClinicalTrials.gov Data API."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .ctgov_api import CtGovApiClientProtocol, CtGovClient

__all__ = ["fetch_trial_records", "study_to_record"]


def _strip_bullet(line: str) -> str:
    return line.lstrip("-*•\u2022").strip()


def _split_eligibility(text: str | None) -> Dict[str, list[str]]:
    inclusion: list[str] = []
    exclusion: list[str] = []
    if not text:
        return {"inclusion": inclusion, "exclusion": exclusion}

    current: list[str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lowered = line.lower()
        if lowered.startswith("inclusion criteria"):
            current = inclusion
            continue
        if lowered.startswith("exclusion criteria"):
            current = exclusion
            continue

        if current is not None:
            cleaned = _strip_bullet(line)
            if cleaned:
                current.append(cleaned)

    return {"inclusion": inclusion, "exclusion": exclusion}


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def study_to_record(study: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a ClinicalTrials.gov study into the parser schema."""

    protocol = study.get("protocolSection", {}) or {}
    identification = protocol.get("identificationModule", {}) or {}
    conditions_module = protocol.get("conditionsModule", {}) or {}
    interventions_module = protocol.get("armsInterventionsModule", {}) or {}
    eligibility_module = protocol.get("eligibilityModule", {}) or {}
    outcomes_module = protocol.get("outcomesModule", {}) or {}

    nct_id = identification.get("nctId") or ""
    title = (
        identification.get("officialTitle") or identification.get("briefTitle") or ""
    )

    condition = _coerce_list(
        conditions_module.get("conditions")
        or conditions_module.get("conditionList", {}).get("conditions")
    )

    interventions: list[str] = []
    raw_interventions = interventions_module.get("interventions", [])
    if isinstance(raw_interventions, Iterable):
        for entry in raw_interventions:
            if not isinstance(entry, Mapping):
                continue
            i_type = entry.get("interventionType") or entry.get("type")
            name = entry.get("name") or entry.get("interventionName")
            if i_type and name:
                interventions.append(f"{i_type}: {name}")
            elif name:
                interventions.append(str(name))
            elif i_type:
                interventions.append(str(i_type))

    eligibility = _split_eligibility(eligibility_module.get("eligibilityCriteria"))

    outcomes: list[Dict[str, str]] = []
    raw_outcomes = outcomes_module.get("primaryOutcomes", [])
    if isinstance(raw_outcomes, Iterable):
        for entry in raw_outcomes:
            if not isinstance(entry, Mapping):
                continue
            outcomes.append(
                {
                    "measure": str(entry.get("measure") or ""),
                    "time_frame": str(
                        entry.get("timeFrame")
                        or entry.get("timeframe")
                        or entry.get("timeFrameDescription")
                        or ""
                    ),
                }
            )

    return {
        "nct_id": str(nct_id),
        "title": str(title) if title is not None else "",
        "condition": condition,
        "interventions": interventions,
        "eligibility": eligibility,
        "outcomes": outcomes,
    }


def fetch_trial_records(
    *,
    params: Mapping[str, Any] | None = None,
    page_size: int | None = 100,
    max_studies: int | None = None,
    client: CtGovApiClientProtocol | None = None,
    raw_dir: Path | None = None,
) -> list[Dict[str, Any]]:
    """Fetch and normalize study records from the Data API."""

    context = nullcontext(client) if client is not None else CtGovClient()
    with context as api_client:  # type: ignore[assignment]
        studies = api_client.fetch_studies(
            params=params or {},
            page_size=page_size,
            max_studies=max_studies,
        )
    if raw_dir is not None:
        _write_raw_studies(studies, raw_dir)

    return [study_to_record(study) for study in studies]


def _study_identifier(study: Mapping[str, Any], index: int) -> str:
    protocol = study.get("protocolSection", {}) or {}
    identification = protocol.get("identificationModule", {}) or {}
    candidate = identification.get("nctId")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return f"study_{index:05d}"


def _write_raw_studies(studies: Iterable[Mapping[str, Any]], raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    for index, study in enumerate(studies, start=1):
        identifier = _study_identifier(study, index)
        base_name = Path(identifier).name
        target = raw_dir / f"{base_name}.json"
        suffix = 1
        while target.exists():
            target = raw_dir / f"{base_name}_{suffix:02d}.json"
            suffix += 1
        with target.open("w", encoding="utf-8") as handle:
            json.dump(study, handle)
            handle.write("\n")
