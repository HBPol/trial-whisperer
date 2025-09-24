import json

from pipeline.download import fetch_trial_records, study_to_record


def _sample_study() -> dict:
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT12345678",
                "officialTitle": "Official Title",
                "briefTitle": "Fallback Title",
            },
            "conditionsModule": {
                "conditionList": {"conditions": ["Condition A", "Condition B"]}
            },
            "armsInterventionsModule": {
                "interventions": [
                    {"interventionType": "Drug", "name": "Drug A"},
                    {"interventionType": None, "name": "Placebo"},
                    {"interventionType": "Procedure", "name": None},
                ]
            },
            "eligibilityModule": {
                "eligibilityCriteria": (
                    "Inclusion Criteria:\n"
                    "- Adults aged 18 or older\n"
                    "• ECOG 0-1\n"
                    "Exclusion Criteria:\n"
                    "- Prior systemic therapy"
                )
            },
            "outcomesModule": {
                "primaryOutcomes": [
                    {"measure": "Progression-free survival", "timeFrame": "12 months"},
                    {"measure": "Overall survival"},
                ]
            },
        }
    }


def test_study_to_record_shapes_expected_schema():
    record = study_to_record(_sample_study())

    assert record["nct_id"] == "NCT12345678"
    assert record["title"] == "Official Title"
    assert record["condition"] == ["Condition A", "Condition B"]
    assert "Drug: Drug A" in record["interventions"]
    assert "Placebo" in record["interventions"]
    assert "Procedure" in record["interventions"]
    assert record["eligibility"]["inclusion"] == [
        "Adults aged 18 or older",
        "ECOG 0-1",
    ]
    assert record["eligibility"]["exclusion"] == ["Prior systemic therapy"]
    assert record["outcomes"][0] == {
        "measure": "Progression-free survival",
        "time_frame": "12 months",
    }
    assert record["outcomes"][1]["time_frame"] == ""


def test_fetch_trial_records_uses_provided_client():
    study = _sample_study()
    captured: dict | None = None

    class DummyClient:
        def fetch_studies(self, **kwargs):
            nonlocal captured
            captured = kwargs
            return [study]

    records = fetch_trial_records(
        params={"query.term": "glioblastoma"},
        page_size=25,
        max_studies=10,
        client=DummyClient(),
    )

    assert captured == {
        "params": {"query.term": "glioblastoma"},
        "page_size": 25,
        "max_studies": 10,
    }
    assert records[0]["nct_id"] == "NCT12345678"


def test_fetch_trial_records_persists_raw_payload(tmp_path):
    study = _sample_study()

    class DummyClient:
        def fetch_studies(self, **kwargs):
            return [study]

    raw_dir = tmp_path / "raw"
    records = fetch_trial_records(client=DummyClient(), raw_dir=raw_dir)

    assert records[0]["nct_id"] == "NCT12345678"

    raw_files = list(raw_dir.glob("*.json"))
    assert len(raw_files) == 1
    with raw_files[0].open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    assert stored == study
