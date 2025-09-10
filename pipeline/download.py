"""Download a curated subset of ClinicalTrials.gov XML.
For MVP, just create a tiny seed set in .data/raw/ to unblock development.
"""
from pathlib import Path


def main():
    raw = Path(".data/raw"); raw.mkdir(parents=True, exist_ok=True)
    # TODO: implement downloader. For now, drop placeholder files.
    (raw / "README.txt").write_text("Seed raw folder. Replace with real XML downloads.")


if __name__ == "__main__":
    main()