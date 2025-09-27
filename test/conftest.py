from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.retrieval import search_client, trial_store
from pipeline.pipeline import process_trials

_FIXTURE_XML_DIR = Path(__file__).parent / "fixtures" / "xml"
_TEST_DATA_PATH = Path(".data/test_processed/trials.jsonl")


def _reset_caches() -> None:
    trial_store.clear_trials_cache()
    search_client.clear_fallback_index()


@pytest.fixture(scope="session", autouse=True)
def prepare_trials_dataset() -> Path:
    """Populate a synthetic trials dataset for the test session."""

    original_env = os.environ.get("TRIALS_DATA_PATH")
    os.environ["TRIALS_DATA_PATH"] = str(_TEST_DATA_PATH)

    _reset_caches()
    process_trials(xml_dir=_FIXTURE_XML_DIR)
    _reset_caches()

    try:
        yield _TEST_DATA_PATH
    finally:
        if original_env is None:
            os.environ.pop("TRIALS_DATA_PATH", None)
        else:
            os.environ["TRIALS_DATA_PATH"] = original_env
        _reset_caches()
