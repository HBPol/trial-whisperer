"""Populate the local ``.data/raw`` folder with example ClinicalTrials.gov XML."""

import shutil
from pathlib import Path


def main() -> None:
    """Copy test fixture XMLs into ``.data/raw``.

    The project currently relies on a small set of curated fixtures located
    under ``test/fixtures/xml``.  This function copies those files into the
    ``.data/raw`` directory so downstream steps in the pipeline have a
    predictable input location.
    """

    raw = Path(".data/raw")
    raw.mkdir(parents=True, exist_ok=True)

    fixture_dir = Path("test/fixtures/xml")
    if not fixture_dir.exists():
        raise FileNotFoundError(f"Fixture directory {fixture_dir} not found")

    for xml_file in fixture_dir.glob("*.xml"):
        shutil.copy2(xml_file, raw / xml_file.name)


if __name__ == "__main__":
    main()
