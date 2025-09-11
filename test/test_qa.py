from fastapi.testclient import TestClient
from app.main import app
from app.routers import qa

client = TestClient(app)

def test_ask_returns_answer():
    def mock_retrieve_chunks(query, nct_id):
        return [{"nct_id": "NCT1", "section": "intro", "text": "sample"}]
    def mock_call_llm_with_citations(query, chunks):
        return "mock answer", chunks
    qa.retrieve_chunks = mock_retrieve_chunks
    qa.call_llm_with_citations = mock_call_llm_with_citations
    response = client.post("/ask/", json={"query": "What?", "nct_id": "NCT1"})
    assert response.status_code == 200
    assert response.json()["answer"] == "mock answer"

def test_ask_requires_nonempty_query():
    qa.retrieve_chunks = lambda query, nct_id: []
    response = client.post("/ask/", json={"query": "", "nct_id": "NCT1"})
    assert response.status_code == 400
