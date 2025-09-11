"""Utilities for parsing ClinicalTrials.gov XML files.

This module currently exposes :func:`parse_one` which consumes a path to a
ClinicalTrials.gov XML file and returns a lightweight dictionary with the
fields needed by the application.  Only a handful of fields are extracted
because this project only needs a small subset of the metadata, but the
parser is written so that missing pieces simply result in empty
strings/lists.  This keeps downstream code simple and mirrors how the real
API behaves.
"""

from pathlib import Path
from typing import List, Dict

from lxml import etree


def _text(element) -> str:
    """Return stripped text for an element, handling ``None`` gracefully."""

    if element is None or element.text is None:
        return ""
    return element.text.strip()


def parse_one(xml_path: Path) -> dict:
    """Parse a single ClinicalTrials.gov XML file.

        Parameters
        ----------
        xml_path:
            Path to the XML file on disk.

        Returns
        -------
        dict
            A dictionary with keys ``nct_id``, ``title``, ``condition``,
            ``interventions`` and ``eligibility``.  An ``outcomes`` field is also
            included for completeness even though it is not currently exercised by
            the tests.
        """

    tree = etree.parse(str(xml_path))

    # --- Basic identifiers -------------------------------------------------
    nct_id = _text(tree.find(".//nct_id"))

    title = _text(tree.find(".//official_title"))
    if not title:
        title = _text(tree.find(".//brief_title"))

    # --- Conditions --------------------------------------------------------
    condition = [_text(c) for c in tree.findall(".//condition") if _text(c)]

    # --- Interventions -----------------------------------------------------
    interventions: List[str] = []
    for iv in tree.findall(".//intervention"):
        i_type = _text(iv.find("intervention_type"))
        name = _text(iv.find("intervention_name"))
        if i_type and name:
            interventions.append(f"{i_type}: {name}")
        elif i_type or name:
            interventions.append(i_type or name)

    # --- Eligibility -------------------------------------------------------
    inclusion: List[str] = []
    exclusion: List[str] = []
    textblock = tree.find(".//eligibility//criteria//textblock")
    if textblock is not None and textblock.text:
        current: List[str] | None = None
        for line in textblock.text.splitlines():
            line = line.strip()
            if not line:
                continue
            lline = line.lower()
            if lline.startswith("inclusion criteria"):
                current = inclusion
                continue
            if lline.startswith("exclusion criteria"):
                current = exclusion
                continue
            if current is not None:
                current.append(line.lstrip("-").strip())

    eligibility: Dict[str, List[str]] = {
        "inclusion": inclusion,
        "exclusion": exclusion,
    }

    # --- Outcomes ----------------------------------------------------------
    outcomes: List[Dict[str, str]] = []
    for out in tree.findall(".//primary_outcome"):
        outcomes.append(
            {
                "measure": _text(out.find("measure")),
                "time_frame": _text(out.find("time_frame")),
            }
        )

    return {
        "nct_id": nct_id,
        "title": title,
        "condition": condition,
        "interventions": interventions,
        "eligibility": eligibility,
        "outcomes": outcomes,
    }


if __name__ == "__main__":
    pass
