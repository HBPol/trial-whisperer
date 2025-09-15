#!/usr/bin/env python
"""Entry point for indexing trial chunks into Qdrant."""

from __future__ import annotations

import os
from pathlib import Path

import tomli
from qdrant_client import QdrantClient

from pipeline.index_qdrant import index_chunks


def main() -> None:
    """Load configuration and index trial chunks."""
    config_path = Path("config/appsettings.toml")
    if not config_path.exists():
        msg = f"Missing configuration file: {config_path}"
        raise FileNotFoundError(msg)

    config = tomli.loads(config_path.read_text())
    proc_dir = config.get("data", {}).get("proc_dir", ".data/processed")
    data_file = Path(proc_dir) / "trials.jsonl"

    client: QdrantClient | None = None
    retrieval = config.get("retrieval", {})
    url = os.getenv("QDRANT_URL", retrieval.get("qdrant_url"))
    api_key = os.getenv("QDRANT_API_KEY", retrieval.get("qdrant_api_key"))
    if url or api_key:
        client = QdrantClient(url=url, api_key=api_key)

    index_chunks(client=client, data_path=data_file)


if __name__ == "__main__":
    main()
