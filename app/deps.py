from functools import lru_cache
from pathlib import Path

import tomli
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from file and environment."""
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    retrieval_backend: str | None = None
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str | None = None

    model_config = SettingsConfigDict(extra="ignore")

    @classmethod
    def settings_customise_sources(
            cls, settings_cls, init_settings, env_settings, file_secret_settings
    ):
        return (
            init_settings,
            cls._toml_config_settings_source,
            env_settings,
            file_secret_settings,
        )

    @classmethod
    def _toml_config_settings_source(cls, settings: BaseSettings):
        config_path = Path("config/appsettings.toml")
        if not config_path.exists():
            return {}
        data = tomli.loads(config_path.read_text())
        return {
            "llm_provider": data.get("llm", {}).get("provider"),
            "llm_model": data.get("llm", {}).get("model"),
            "llm_api_key": data.get("llm", {}).get("api_key"),
            "retrieval_backend": data.get("retrieval", {}).get("backend"),
            "qdrant_url": data.get("retrieval", {}).get("qdrant_url"),
            "qdrant_api_key": data.get("retrieval", {}).get("qdrant_api_key"),
            "qdrant_collection": data.get("retrieval", {}).get("collection"),
        }

@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
