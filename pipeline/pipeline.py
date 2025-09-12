from __future__ import annotations

from pathlib import Path
import json
from typing import List, Dict

from .parse_xml import parse_one
from .normalize import normalize
from .chunk import chunk_sections


def process_trials(xml_dir: Path, output_path: Path | None = None) -> Path:
    """Parse, normalize, chunk and write trials to JSONL.

    Parameters
    ----------
    xml_dir: Path
        Directory containing ClinicalTrials.gov XML files.
    output_path: Path | None
        Where to write the resulting JSON Lines file. Defaults to
        ``.data/processed/trials.jsonl``.

    Returns
    -------
    Path
        Path to the JSON Lines file containing chunked trial data.
    """
    if output_path is None:
        output_path = Path(".data/processed/trials.jsonl")

    xml_files = sorted(xml_dir.glob("*.xml"))
    parsed = [parse_one(p) for p in xml_files]
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


__all__ = ["process_trials"]
