#!/usr/bin/env python
"""Entry point for indexing trial chunks into Qdrant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tomli
from qdrant_client import QdrantClient

from pipeline.index_qdrant import index_chunks


def main() -> None:
    """Load configuration and index trial chunks."""
    config_path = Path("config/appsettings.toml")
    config: dict[str, Any] = {}
    if config_path.exists():
        config = tomli.loads(config_path.read_text())

    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        data_cfg = {}

    proc_dir = data_cfg.get("proc_dir")
    data_env = os.getenv("TRIALS_DATA_PATH")
    if proc_dir:
        data_file = Path(proc_dir) / "trials.jsonl"
    elif data_env:
        data_file = Path(data_env)
    else:
        data_file = Path(".data/processed") / "trials.jsonl"

    client: QdrantClient | None = None
    retrieval = config.get("retrieval", {})
    if not isinstance(retrieval, dict):
        retrieval = {}

    url = retrieval.get("qdrant_url") or os.getenv("QDRANT_URL")
    api_key = retrieval.get("qdrant_api_key") or os.getenv("QDRANT_API_KEY")
    if url or api_key:
        client = QdrantClient(url=url, api_key=api_key)

    index_chunks(client=client, data_path=data_file)


if __name__ == "__main__":
    main()
