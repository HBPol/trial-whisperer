"""Index chunks into Qdrant (free tier)."""

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

COLLECTION = "trialwhisperer"


def ensure_collection(client: QdrantClient, dim: int = 768):
    if COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


if __name__ == "__main__":
    # TODO: load chunks, embed (sentence-transformers), upsert
    pass