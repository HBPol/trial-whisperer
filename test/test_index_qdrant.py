from types import SimpleNamespace
from unittest.mock import MagicMock

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from pipeline.index_qdrant import COLLECTION, ensure_collection


def test_ensure_collection_creates_with_expected_dim():
    mock_client = MagicMock(spec=QdrantClient)
    mock_client.get_collections.return_value = SimpleNamespace(collections=[])

    ensure_collection(mock_client, dim=42)

    mock_client.create_collection.assert_called_once()
    kwargs = mock_client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == COLLECTION
    # Verify vector size passed to Qdrant
    assert kwargs["vectors_config"].size == 42


def test_upsert_payload_schema():
    mock_client = MagicMock(spec=QdrantClient)
    payload = {"nct_id": "NCT0", "section": "eligibility", "text": "info"}
    vector = [0.1, 0.2, 0.3]
    point = PointStruct(id=0, vector=vector, payload=payload)

    mock_client.upsert(collection_name=COLLECTION, points=[point])

    upserted_points = mock_client.upsert.call_args.kwargs["points"]
    assert len(upserted_points) == 1
    assert set(upserted_points[0].payload.keys()) == {"nct_id", "section", "text"}
