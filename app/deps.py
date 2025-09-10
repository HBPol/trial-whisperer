from functools import lru_cache
import tomli
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    llm_provider: str
    llm_model: str
    llm_api_key: str
    retrieval_backend: str
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str | None = None


@lru_cache
def get_settings() -> Settings:
    cfg = tomli.loads(Path("config/appsettings.toml").read_text())
    return Settings(
        llm_provider=cfg["llm"]["provider"],
        llm_model=cfg["llm"]["model"],
        llm_api_key=cfg["llm"]["api_key"],
        retrieval_backend=cfg["retrieval"]["backend"],
        qdrant_url=cfg["retrieval"].get("qdrant_url"),
        qdrant_api_key=cfg["retrieval"].get("qdrant_api_key"),
        qdrant_collection=cfg["retrieval"].get("collection"),
    )
