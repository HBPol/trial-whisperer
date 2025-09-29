import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException
from qdrant_client.http.models import PointStruct

from pipeline.index_qdrant import COLLECTION, ensure_collection, index_chunks


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


def test_index_chunks_upserts_embeddings_and_payloads():
    fixture_path = Path(__file__).parent / "fixtures" / "chunks.json"
    with fixture_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    mock_client = MagicMock(spec=QdrantClient)
    mock_embed = MagicMock()
    mock_embed.encode.return_value = [[0.1, 0.1], [0.2, 0.2]]

    index_chunks(mock_client, mock_embed, chunks)

    mock_embed.encode.assert_called_once_with([c["text"] for c in chunks])
    mock_client.upsert.assert_called_once()
    kwargs = mock_client.upsert.call_args.kwargs
    assert kwargs["collection_name"] == COLLECTION

    points = kwargs["points"]
    assert len(points) == len(chunks)
    for idx, point in enumerate(points):
        assert point.payload == chunks[idx]
        assert point.vector == mock_embed.encode.return_value[idx]


def test_index_chunks_reads_jsonl_and_invokes_dependencies(tmp_path, monkeypatch):
    data_file = tmp_path / "trials.jsonl"
    chunk = {"nct_id": "NCT0", "section": "title", "text": "hello world"}
    with data_file.open("w", encoding="utf-8") as f:
        json.dump(chunk, f)
        f.write("\n")

    mock_client = MagicMock(spec=QdrantClient)
    mock_embed = MagicMock()
    mock_embed.get_sentence_embedding_dimension.return_value = 3
    mock_embed.encode.return_value = [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(
        "pipeline.index_qdrant.QdrantClient", MagicMock(return_value=mock_client)
    )
    import types

    dummy_module = types.SimpleNamespace(SentenceTransformer=lambda name: mock_embed)
    monkeypatch.setitem(sys.modules, "sentence_transformers", dummy_module)

    index_chunks(data_path=data_file)

    mock_embed.encode.assert_called_once_with([chunk["text"]])
    mock_client.upsert.assert_called_once()
    # ensure collection created with vector size from model
    mock_client.create_collection.assert_called_once()
    size = mock_client.create_collection.call_args.kwargs["vectors_config"].size
    assert size == 3


def test_index_script_uses_qdrant_config(tmp_path, monkeypatch):
    monkeypatch.delenv("TRIALS_DATA_PATH", raising=False)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "appsettings.toml").write_text(
        """
[retrieval]
qdrant_url = "https://example"
qdrant_api_key = "secret"

[data]
proc_dir = "."
"""
    )
    data_file = tmp_path / "trials.jsonl"
    data_file.write_text("{}\n")

    from scripts import index as index_script

    mock_client_cls = MagicMock()
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    monkeypatch.setattr(index_script, "QdrantClient", mock_client_cls)
    mock_index_chunks = MagicMock()
    monkeypatch.setattr(index_script, "index_chunks", mock_index_chunks)

    monkeypatch.chdir(tmp_path)
    index_script.main()

    mock_client_cls.assert_called_once_with(url="https://example", api_key="secret")
    mock_index_chunks.assert_called_once()
    kwargs = mock_index_chunks.call_args.kwargs
    assert kwargs["client"] is mock_client
    assert Path(kwargs["data_path"]).resolve() == data_file


def test_index_script_uses_env_without_config(tmp_path, monkeypatch):
    monkeypatch.delenv("TRIALS_DATA_PATH", raising=False)
    data_file = tmp_path / "custom.jsonl"
    data_file.write_text("{}\n")

    monkeypatch.setenv("TRIALS_DATA_PATH", str(data_file))
    monkeypatch.setenv("QDRANT_URL", "https://env-url")
    monkeypatch.setenv("QDRANT_API_KEY", "env-key")

    from scripts import index as index_script

    mock_client_cls = MagicMock()
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    monkeypatch.setattr(index_script, "QdrantClient", mock_client_cls)
    mock_index_chunks = MagicMock()
    monkeypatch.setattr(index_script, "index_chunks", mock_index_chunks)

    monkeypatch.chdir(tmp_path)
    index_script.main()

    mock_client_cls.assert_called_once_with(url="https://env-url", api_key="env-key")
    mock_index_chunks.assert_called_once()
    kwargs = mock_index_chunks.call_args.kwargs
    assert kwargs["client"] is mock_client
    assert Path(kwargs["data_path"]).resolve() == data_file


@pytest.mark.parametrize("exc", [httpx.ConnectError("boom"), ApiException("boom")])
def test_index_chunks_raises_helpful_error_on_connection_failure(monkeypatch, exc):
    mock_client = MagicMock(spec=QdrantClient)
    mock_embed = MagicMock()
    mock_embed.get_sentence_embedding_dimension.return_value = 3
    chunk = {"nct_id": "NCT0", "section": "title", "text": "hi"}

    monkeypatch.setattr(
        "pipeline.index_qdrant.ensure_collection", MagicMock(side_effect=exc)
    )

    with pytest.raises(RuntimeError) as err:
        index_chunks(client=mock_client, embed_model=mock_embed, chunks=[chunk])

    message = str(err.value)
    assert "QDRANT_URL" in message
    assert "QDRANT_API_KEY" in message
    assert "local Qdrant instance" in message
