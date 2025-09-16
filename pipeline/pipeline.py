from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import tomli

from .chunk import chunk_sections
from .normalize import normalize
from .parse_xml import parse_one


def process_trials(
    xml_dir: Path | None = None,
    *,
    records: Iterable[Mapping[str, Any]] | None = None,
    output_path: Path | None = None,
) -> Path:
    """Parse, normalize, chunk and write trials to JSONL."""

    if xml_dir is None and records is None:
        raise ValueError("Either xml_dir or records must be provided")

    if output_path is None:
        output_path = Path(".data/processed/trials.jsonl")
    if output_path is None:
        output_path = Path(".data/processed/trials.jsonl")

    if records is None:
        xml_dir = Path(xml_dir)  # type: ignore[arg-type]
        xml_files = sorted(xml_dir.glob("*.xml"))
        parsed = [parse_one(p) for p in xml_files]
    else:
        parsed = list(records)

    normalized = normalize(parsed)

    chunks: List[Dict[str, str]] = []
    for record in normalized:
        chunks.extend(chunk_sections(record))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk, f)
            f.write("\n")

    return output_path


def _load_config(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    return tomli.loads(content)


def _default_output_path(config: Mapping[str, Any]) -> Path:
    data_cfg = config.get("data", {}) or {}
    proc_dir = data_cfg.get("proc_dir", ".data/processed")
    return Path(proc_dir) / "trials.jsonl"


def _api_settings(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], int | None, int | None]:
    data_cfg = config.get("data", {}) or {}
    api_cfg = data_cfg.get("api", {}) or {}
    params = api_cfg.get("params", {}) or {}
    if not isinstance(params, Mapping):
        params = {}
    else:
        params = {str(k): v for k, v in params.items()}

    page_size = api_cfg.get("page_size")
    if isinstance(page_size, str) and page_size.isdigit():
        page_size = int(page_size)
    elif not isinstance(page_size, int):
        page_size = None

    max_studies = api_cfg.get("max_studies")
    if isinstance(max_studies, str) and max_studies.isdigit():
        max_studies = int(max_studies)
    elif not isinstance(max_studies, int):
        max_studies = None

    return dict(params), page_size, max_studies


def _append_param(params: dict[str, Any], key: str, value: str) -> None:
    existing = params.get(key)
    if existing is None:
        params[key] = value
        return
    if isinstance(existing, (list, tuple)):
        params[key] = [*list(existing), value]
        return
    params[key] = [existing, value]


def _parse_param(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Invalid parameter '{value}'. Expected key=value format.")
    key, val = value.split("=", 1)
    return key.strip(), val.strip()


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Trial ingestion pipeline")
    parser.add_argument("--config", type=Path, default=Path("config/appsettings.toml"))
    parser.add_argument(
        "--xml-dir", type=Path, help="Directory containing ClinicalTrials.gov XML files"
    )
    parser.add_argument(
        "--from-api",
        action="store_true",
        help="Fetch trials from the ClinicalTrials.gov Data API",
    )
    parser.add_argument("--output", type=Path, help="Destination JSONL path")
    parser.add_argument("--page-size", type=int, help="Number of studies per API page")
    parser.add_argument(
        "--max-studies", type=int, help="Maximum number of studies to download"
    )
    parser.add_argument(
        "--query-term", help="Shortcut for the 'query.term' API parameter"
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Additional API parameter in key=value form (can be repeated)",
    )

    args = parser.parse_args(argv)

    if args.from_api and args.xml_dir is not None:
        parser.error("--from-api cannot be combined with --xml-dir")

    config = _load_config(args.config)
    output_path = args.output or _default_output_path(config)

    if args.from_api:
        params, cfg_page_size, cfg_max_studies = _api_settings(config)

        if args.query_term:
            params["query.term"] = args.query_term

        for raw in args.param:
            key, value = _parse_param(raw)
            _append_param(params, key, value)

        page_size = args.page_size or cfg_page_size
        max_studies = args.max_studies or cfg_max_studies

        from .download import fetch_trial_records

        records = fetch_trial_records(
            params=params,
            page_size=page_size,
            max_studies=max_studies,
        )
        return process_trials(records=records, output_path=output_path)

    if args.xml_dir is None:
        parser.error("Specify --from-api or provide --xml-dir")

    return process_trials(args.xml_dir, output_path=output_path)


if __name__ == "__main__":
    main()


__all__ = ["process_trials", "main"]
