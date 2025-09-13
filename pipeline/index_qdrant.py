"""Utilities for indexing trial chunks into Qdrant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

COLLECTION = "trialwhisperer"


def ensure_collection(client: QdrantClient, dim: int = 768) -> None:
    """Ensure the Qdrant collection exists with the desired vector size."""

    if COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def index_chunks(
    client: QdrantClient | None = None,
    embed_model: Optional[object] = None,
    chunks: Iterable[Dict[str, str]] | None = None,
    *,
    data_path: Path | str = Path(".data/processed/trials.jsonl"),
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    """Embed and upsert trial ``chunks`` into Qdrant.

    When ``client`` or ``embed_model`` are omitted the function will create
    sensible defaults using :class:`qdrant_client.QdrantClient` and a
    :class:`sentence_transformers.SentenceTransformer` model.  If ``chunks`` is
    ``None`` the data is loaded from ``data_path`` which is expected to be a
    JSON Lines file produced by :mod:`pipeline.pipeline`.
    """

    if chunks is None:
        with Path(data_path).open("r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]

    if embed_model is None:
        from sentence_transformers import SentenceTransformer

        embed_model = SentenceTransformer(model_name)

    if client is None:
        client = QdrantClient()

    dim = int(embed_model.get_sentence_embedding_dimension())
    ensure_collection(client, dim=dim)

    texts = [c["text"] for c in chunks]
    vectors = embed_model.encode(texts)
    points = [
        PointStruct(
            id=i,
            vector=vector,
            payload={k: c[k] for k in ("nct_id", "section", "text")},
        )
        for i, (c, vector) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=COLLECTION, points=points)


if __name__ == "__main__":
    index_chunks()
