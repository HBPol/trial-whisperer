import json
from pathlib import Path

from pipeline.pipeline import process_trials

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "xml"


def test_pipeline_creates_chunks_per_trial(monkeypatch, tmp_path):
    output_jsonl = tmp_path / "custom" / "trials.jsonl"
    monkeypatch.setenv("TRIALS_DATA_PATH", str(output_jsonl))

    process_trials(FIXTURE_DIR)

    assert output_jsonl.exists()

    with output_jsonl.open("r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    # Every chunk has essential metadata
    assert all("nct_id" in c and c["nct_id"] for c in chunks)
    assert all("section" in c and c["section"] for c in chunks)
    assert all("text" in c and c["text"] for c in chunks)

    # There is at least one chunk per trial
    expected_ids = {p.stem for p in FIXTURE_DIR.glob("NCT*.xml")}
    for nct_id in expected_ids:
        assert any(chunk["nct_id"] == nct_id for chunk in chunks)
