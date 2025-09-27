from pathlib import Path

import pytest

from pipeline import download as download_module
from pipeline import pipeline as pipeline_module
from pipeline.ctgov_api import CtGovClient, CtGovRequestsClient
from pipeline.pipeline import _api_settings


def test_api_settings_includes_custom_base_url():
    config = {
        "data": {
            "api": {
                "base_url": " https://alt.example/v2 ",
                "params": {"query.term": "glioblastoma"},
            }
        }
    }

    _, _, _, client_settings, backend = _api_settings(config)

    assert client_settings["base_url"] == "https://alt.example/v2"
    assert backend == "httpx"


@pytest.mark.parametrize(
    ("backend_name", "expected_cls"),
    [("httpx", CtGovClient), ("requests", CtGovRequestsClient)],
)
def test_pipeline_selects_client_class_based_on_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_name: str,
    expected_cls: type,
):
    config = {"data": {"api": {"backend": backend_name}}}

    monkeypatch.setattr(pipeline_module, "_load_config", lambda path: config)

    captured: dict[str, type] = {}

    def fake_fetch_trial_records(*, client, **kwargs):
        captured["client_type"] = type(client)
        return []

    monkeypatch.setattr(
        download_module, "fetch_trial_records", fake_fetch_trial_records
    )

    def fake_process_trials(*, records=None, output_path=None, **kwargs):
        assert records == []
        assert output_path is not None
        return Path(output_path)

    monkeypatch.setattr(pipeline_module, "process_trials", fake_process_trials)

    output_path = tmp_path / "out.jsonl"
    result = pipeline_module.main(
        [
            "--from-api",
            "--config",
            str(tmp_path / "config.toml"),
            "--output",
            str(output_path),
        ]
    )

    assert result == output_path
    assert captured["client_type"] is expected_cls
