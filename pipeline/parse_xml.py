"""Parse ClinicalTrials.gov XML → structured dicts.
Use lxml; extract NCT ID, title, conditions, interventions, eligibility, outcomes.
"""
from pathlib import Path


def parse_one(xml_path: Path) -> dict:
    # TODO: parse with lxml; return structured dict
    return {"nct_id": "NCT00000000", "eligibility": {"inclusion": [], "exclusion": []}}


if __name__ == "__main__":
    pass