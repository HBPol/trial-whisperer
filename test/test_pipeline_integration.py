import json
from pathlib import Path

from pipeline import ctgov_api as ctgov_api_module
from pipeline import pipeline as pipeline_module
from pipeline.pipeline import process_trials

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "xml"


def test_pipeline_creates_chunks_per_trial(monkeypatch, tmp_path):
    output_jsonl = tmp_path / "custom" / "trials.jsonl"
    raw_dir = tmp_path / "raw"
    monkeypatch.setenv("TRIALS_DATA_PATH", str(output_jsonl))

    process_trials(FIXTURE_DIR, raw_dir=raw_dir)

    assert output_jsonl.exists()
    assert raw_dir.exists()

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

    archived = sorted(raw_dir.glob("*.xml.gz"))
    assert len(archived) == len(expected_ids)
    expected_raw_names = {f"{path.name}.gz" for path in FIXTURE_DIR.glob("*.xml")}
    assert {path.name for path in archived} == expected_raw_names


def test_pipeline_from_api_creates_raw_and_processed(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"

    monkeypatch.delenv("TRIALS_DATA_PATH", raising=False)

    config = {
        "data": {
            "raw_dir": str(raw_dir),
            "proc_dir": str(proc_dir),
            "api": {"backend": "httpx", "params": {}},
        }
    }

    monkeypatch.setattr(pipeline_module, "_load_config", lambda path: config)

    sample_study = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCTTEST0001",
                "officialTitle": "Integration Study",
                "briefTitle": "Integration Study",
            },
            "conditionsModule": {"conditions": ["Condition"]},
            "armsInterventionsModule": {
                "interventions": [{"interventionType": "Drug", "name": "Drug X"}]
            },
            "eligibilityModule": {
                "eligibilityCriteria": (
                    "Inclusion Criteria:\n"
                    "- Adults\n"
                    "Exclusion Criteria:\n"
                    "- Prior therapy"
                )
            },
            "outcomesModule": {
                "primaryOutcomes": [{"measure": "Outcome", "timeFrame": "12 months"}]
            },
        }
    }

    class DummyClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def fetch_studies(self, **_kwargs):
            return [sample_study]

    monkeypatch.setattr(ctgov_api_module, "CtGovClient", DummyClient)
    monkeypatch.setattr(ctgov_api_module, "CtGovRequestsClient", DummyClient)

    result_path = pipeline_module.main(
        ["--from-api", "--config", str(tmp_path / "config.toml")]
    )

    expected_output = proc_dir / "trials.jsonl"
    assert result_path == expected_output
    assert expected_output.exists()

    raw_files = list(raw_dir.glob("*.json"))
    assert len(raw_files) == 1
    with raw_files[0].open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored["protocolSection"]["identificationModule"]["nctId"] == "NCTTEST0001"
