import json

from fastapi.testclient import TestClient

from app.agents import tools
from app.main import app
from app.retrieval import search_client, trial_store
from app.routers import qa
from eval.eval import answer_exact_match, citations_match

client = TestClient(app)


def _load_index() -> None:
    path = trial_store.get_trials_data_path()
    with path.open("r", encoding="utf-8") as f:
        search_client.clear_fallback_index()
        search_client._FAKE_INDEX = [json.loads(line) for line in f]


def test_ask_returns_answer_and_citations():
    _load_index()
    sample_id = search_client._FAKE_INDEX[0]["nct_id"]
    response = client.post(
        "/ask/", json={"query": "What is this study?", "nct_id": sample_id}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data.get("answer"), str)
    assert data["answer"]
    assert not data["answer"].startswith("[LLM error]")
    assert "citations" in data and isinstance(data["citations"], list)
    assert len(data["citations"]) >= 1
    citation = data["citations"][0]
    assert {"nct_id", "section", "text_snippet"} <= citation.keys()


def test_ask_requires_query():
    _load_index()
    sample_id = search_client._FAKE_INDEX[0]["nct_id"]
    for payload in (
        {"query": "", "nct_id": sample_id},
        {"nct_id": sample_id},
    ):
        response = client.post("/ask/", json=payload)
        assert response.status_code == 400


def test_ask_strips_citation_markers(monkeypatch):
    _load_index()
    sample_id = search_client._FAKE_INDEX[0]["nct_id"]

    raw_answer = (
        "Answer: Based on the provided context, the study enrolls 120 participants. (1)"
    )
    fake_citations = [
        {"nct_id": sample_id, "section": "Overview", "text": "Enrollment is 120."}
    ]

    def _fake_call_llm(query, chunks):
        return raw_answer, fake_citations

    monkeypatch.setattr(qa, "call_llm_with_citations", _fake_call_llm)

    response = client.post(
        "/ask/",
        json={"query": "How many participants are in the study?", "nct_id": sample_id},
    )
    assert response.status_code == 200
    data = response.json()
    expected_answer = "The study enrolls 120 participants."
    assert data["answer"].lower() == expected_answer.lower()
    assert "(1)" not in data["answer"]
    assert data["answer"].lower().startswith("the study")
    assert answer_exact_match(data["answer"], [expected_answer])


def test_answer_alignment_restores_context_span(monkeypatch):
    sample_chunk = {
        "nct_id": "NCTALIGN01",
        "section": "Eligibility.Inclusion",
        "text": "Eligible patients must be at least 18 years of age.",
    }

    def _fake_retrieve_chunks(query, nct_id):
        return [sample_chunk]

    def _fake_call_llm(query, chunks):
        return "At least 18 years of age.", [sample_chunk]

    monkeypatch.setattr(qa, "retrieve_chunks", _fake_retrieve_chunks)
    monkeypatch.setattr(qa, "call_llm_with_citations", _fake_call_llm)

    response = client.post(
        "/ask/",
        json={"query": "What is the minimum age?", "nct_id": "NCTALIGN01"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == sample_chunk["text"]


def test_citation_selector_covers_late_sections():
    context_chunks = [
        {
            "nct_id": "NCT00000001",
            "section": "Overview",
            "text": "This phase 2 study evaluates response rates.",
        },
        {
            "nct_id": "NCT00000001",
            "section": "Background",
            "text": "Patients have relapsed disease.",
        },
        {
            "nct_id": "NCT00000001",
            "section": "Sponsor",
            "text": "Sponsored by Example Oncology Group.",
        },
        {
            "nct_id": "NCT00000001",
            "section": "Eligibility.Inclusion",
            "text": "Adults aged 18 years or older with ECOG performance status 0-1 are eligible.",
        },
        {
            "nct_id": "NCT00000001",
            "section": "Eligibility.Exclusion",
            "text": "Patients with prior systemic therapy are excluded from enrollment.",
        },
        {
            "nct_id": "NCT00000001",
            "section": "Outcome Measures.Primary",
            "text": "Primary outcome measure is progression-free survival at 12 months.",
        },
        {
            "nct_id": "NCT00000001",
            "section": "Arms",
            "text": "Participants receive ibrutinib monotherapy throughout the study.",
        },
    ]

    answer = (
        "Adults aged 18 years or older with ECOG 0-1 may enroll, patients with prior "
        "systemic therapy are excluded, the primary outcome measures progression-free "
        "survival, and participants receive ibrutinib monotherapy."
    )

    citations = tools._select_citations(answer, context_chunks)
    sections = [citation["section"] for citation in citations]
    expected_sections = [
        "Eligibility.Inclusion",
        "Eligibility.Exclusion",
        "Outcome Measures.Primary",
        "Arms",
    ]

    assert citations_match(citations, expected_sections, "NCT00000001")
    assert any(
        section not in {c["section"] for c in context_chunks[:3]}
        for section in sections
    )
    assert len(citations) >= len(expected_sections)


def test_select_chunks_context_includes_all_lines():
    chunks = [
        {
            "nct_id": "NCT12345678",
            "section": "Overview",
            "text": "First chunk of context.",
            "score": 0.9,
        },
        {
            "nct_id": "NCT12345678",
            "section": "Eligibility",
            "text": "Second chunk of context.",
            "score": 0.8,
        },
        {
            "nct_id": "NCT12345678",
            "section": "Arms",
            "text": "Third chunk of context.",
            "score": 0.7,
        },
    ]

    selected, context_text = tools._select_chunks_for_context(chunks, max_chars=1000)

    assert len(selected) == len(chunks)

    expected_lines = [
        tools._format_chunk_line(chunk, idx)
        for idx, chunk in enumerate(selected, start=1)
    ]

    assert context_text.splitlines() == expected_lines
    assert tools._format_context(selected).splitlines() == expected_lines
