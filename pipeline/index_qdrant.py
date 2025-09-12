"""Index chunks into Qdrant (free tier)."""

from typing import Dict, Iterable

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

COLLECTION = "trialwhisperer"


def ensure_collection(client: QdrantClient, dim: int = 768):
    if COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def index_chunks(
    client: QdrantClient,
    embed_model,
    chunks: Iterable[Dict[str, str]],
) -> None:
    """Embed ``chunks`` and upsert into Qdrant ``client``."""

    texts = [c["text"] for c in chunks]
    vectors = embed_model.encode(texts)
    points = [
        PointStruct(id=i, vector=vector, payload=chunk)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=COLLECTION, points=points)


if __name__ == "__main__":
    # TODO: load chunks, embed (sentence-transformers), upsert
    pass
