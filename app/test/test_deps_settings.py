import pytest

from ..deps import Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_environment_variable_overrides_toml(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "appsettings.toml").write_text(
        """
[llm]
model = "file-model"
        """.strip()
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_MODEL", "env-model")

    settings = Settings()

    assert settings.llm_model == "env-model"
    assert settings.model_dump().get("llm_model") == "env-model"

    # Ensure the TOML value is still available as a fallback when env is absent.
    monkeypatch.delenv("LLM_MODEL")
    settings_from_file = Settings()
    assert settings_from_file.llm_model == "file-model"
