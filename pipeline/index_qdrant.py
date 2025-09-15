"""Utilities for indexing trial chunks into Qdrant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException
from qdrant_client.http.models import Distance, PointStruct, VectorParams

COLLECTION = "trialwhisperer"


def ensure_collection(client: QdrantClient, dim: int = 768) -> None:
    """Ensure the Qdrant collection exists with the desired vector size."""

    vector_config = {"text": VectorParams(size=dim, distance=Distance.COSINE)}
    existing_collections = {c.name for c in client.get_collections().collections}

    needs_recreate = True
    if COLLECTION in existing_collections:
        try:
            collection_info = client.get_collection(COLLECTION)
        except ApiException:
            # If we fail to fetch collection info treat it as needing recreation.
            pass
        else:
            vectors = collection_info.config.params.vectors

            vector_mapping: Mapping[str, object] | None = None
            if isinstance(vectors, Mapping):
                vector_mapping = vectors
            else:
                to_dict = getattr(vectors, "to_dict", None)
                if callable(to_dict):
                    maybe_dict = to_dict()
                    if isinstance(maybe_dict, Mapping):
                        vector_mapping = maybe_dict

            if vector_mapping is not None:
                text_vector = vector_mapping.get("text")
                size: Optional[int] = None
                distance: Optional[object] = None

                if isinstance(text_vector, VectorParams):
                    size = text_vector.size
                    distance = text_vector.distance
                elif isinstance(text_vector, Mapping):
                    size = text_vector.get("size")
                    distance = text_vector.get("distance")

                if size == dim and (
                    distance == Distance.COSINE
                    or getattr(distance, "value", distance) == Distance.COSINE.value
                ):
                    needs_recreate = False

    if needs_recreate:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=vector_config,
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
    try:
        ensure_collection(client, dim=dim)
    except (httpx.ConnectError, ApiException) as exc:
        raise RuntimeError(
            "Failed to connect to Qdrant. Verify QDRANT_URL, QDRANT_API_KEY, or that a local Qdrant instance is running."
        ) from exc

    texts = [c["text"] for c in chunks]
    vectors = embed_model.encode(texts)
    points = [
        PointStruct(
            id=i,
            vector={"text": vector},
            payload={k: c[k] for k in ("nct_id", "section", "text")},
        )
        for i, (c, vector) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=COLLECTION, points=points)


if __name__ == "__main__":
    index_chunks()
