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


def test_answer_alignment_preserves_exact_gold_answer(monkeypatch):
    sample_chunk = {
        "nct_id": "NCTALIGN02",
        "section": "Arms",
        "text": "The investigational drug is pembrolizumab.",
    }

    def _fake_retrieve_chunks(query, nct_id):
        return [sample_chunk]

    def _fake_call_llm(query, chunks):
        return "Pembrolizumab.", [sample_chunk]

    monkeypatch.setattr(qa, "retrieve_chunks", _fake_retrieve_chunks)
    monkeypatch.setattr(qa, "call_llm_with_citations", _fake_call_llm)

    response = client.post(
        "/ask/",
        json={"query": "What is the study drug?", "nct_id": "NCTALIGN02"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Pembrolizumab."


def test_answer_alignment_expands_truncated_sentence(monkeypatch):
    sample_chunk = {
        "nct_id": "NCTALIGN03",
        "section": "Eligibility.Inclusion",
        "text": (
            "Eligible patients must be at least 18 years of age. "
            "Participants also need adequate organ function."
        ),
    }

    def _fake_retrieve_chunks(query, nct_id):
        return [sample_chunk]

    def _fake_call_llm(query, chunks):
        return "At least 18 years of age.", [sample_chunk]

    monkeypatch.setattr(qa, "retrieve_chunks", _fake_retrieve_chunks)
    monkeypatch.setattr(qa, "call_llm_with_citations", _fake_call_llm)

    response = client.post(
        "/ask/",
        json={"query": "What is the minimum age?", "nct_id": "NCTALIGN03"},
    )

    assert response.status_code == 200
    data = response.json()
    expected = "Eligible patients must be at least 18 years of age."
    assert data["answer"] == expected


def test_alignment_expands_patient_label_clause(monkeypatch):
    sample_chunk = {
        "nct_id": "NCT06051214",
        "section": "Eligibility.Exclusion",
        "text": (
            "1. Patient who has had a VTE in the 12 months preceding the diagnosis of "
            "cancer, 2. Patient on low molecular weight heparins, standard "
            "unfractionated heparins and anti-vitamin K2, 3. Women who are pregnant, "
            "likely to become pregnant or who are breast-feeding, 4. Persons deprived "
            "of their liberty, under court protection, under curators or under the "
            "authority of a guardian, 5. Unable to undergo medical monitoring of the "
            "trial for geographical, social or psychological reasons."
        ),
    }

    def _fake_retrieve_chunks(query, nct_id):
        return [sample_chunk]

    def _fake_call_llm(query, chunks):
        answer = (
            "low molecular weight heparins, standard unfractionated heparins and "
            "anti-vitamin K2."
        )
        return answer, [sample_chunk]

    monkeypatch.setattr(qa, "retrieve_chunks", _fake_retrieve_chunks)
    monkeypatch.setattr(qa, "call_llm_with_citations", _fake_call_llm)

    response = client.post(
        "/ask/",
        json={
            "query": "Which anticoagulant therapies exclude someone from NCT06051214?",
            "nct_id": "NCT06051214",
        },
    )

    assert response.status_code == 200
    data = response.json()
    expected = (
        "Patient on low molecular weight heparins, standard unfractionated "
        "heparins and anti-vitamin K2"
    )
    assert data["answer"] == expected


def test_alignment_restores_numbered_biopsy_clause(monkeypatch):
    sample_chunk = {
        "nct_id": "NCT05386043",
        "section": "Eligibility.Inclusion",
        "text": (
            "1. Subject is between 18 and 89 years of age. 2. Subject has "
            "radiologically-diagnosed or suspected WHO Grade II-IV glioma based on "
            "physician review or conformance with published WHO criteria as evaluated "
            "by the PI*. 3. Subject is treatment-naïve with the exception of previous "
            "biopsy for the above condition. 4. Subject is planning to undergo surgical "
            "resection and biopsy of their brain tumor. 5. Subject has sufficient tissue "
            "so that the study team is able to acquire at least 2 biopsy samples during "
            "resection. 6. Subject is able to read and write in English."
        ),
    }

    def _fake_retrieve_chunks(query, nct_id):
        return [sample_chunk]

    def _fake_call_llm(query, chunks):
        return "Able to acquire at least 2 biopsy samples during resection.", [
            sample_chunk
        ]

    monkeypatch.setattr(qa, "retrieve_chunks", _fake_retrieve_chunks)
    monkeypatch.setattr(qa, "call_llm_with_citations", _fake_call_llm)

    response = client.post(
        "/ask/",
        json={
            "query": "How many biopsy samples must be obtainable during surgery?",
            "nct_id": "NCT05386043",
        },
    )

    assert response.status_code == 200
    data = response.json()
    expected = (
        "Subject has sufficient tissue so that the study team is able to acquire at "
        "least 2 biopsy samples during resection."
    )
    assert data["answer"] == expected


def test_alignment_includes_hypofractionated_label(monkeypatch):
    sample_chunk = {
        "nct_id": "NCT06740955",
        "section": "Interventions",
        "text": (
            "RADIATION: hypofractionated postoperative radiotherapy RADIATION: "
            "Conventionally fractionated postoperative radiotherapy"
        ),
    }

    def _fake_retrieve_chunks(query, nct_id):
        return [sample_chunk]

    def _fake_call_llm(query, chunks):
        return "hypofractionated postoperative radiotherapy", [sample_chunk]

    monkeypatch.setattr(qa, "retrieve_chunks", _fake_retrieve_chunks)
    monkeypatch.setattr(qa, "call_llm_with_citations", _fake_call_llm)

    response = client.post(
        "/ask/",
        json={
            "query": "What hypofractionated approach is being tested?",
            "nct_id": "NCT06740955",
        },
    )

    assert response.status_code == 200
    data = response.json()
    expected = "RADIATION: hypofractionated postoperative radiotherapy"
    assert data["answer"] == expected


def test_alignment_returns_pet_tracer_label(monkeypatch):
    sample_chunk = {
        "nct_id": "NCT06113705",
        "section": "Interventions",
        "text": (
            "DIAGNOSTIC_TEST: 18F-GE-180 PET DIAGNOSTIC_TEST: Advanced MRI OTHER: "
            "Collection of hematopoietic stem cells"
        ),
    }

    def _fake_retrieve_chunks(query, nct_id):
        return [sample_chunk]

    def _fake_call_llm(query, chunks):
        return "18F-GE-180 PET", [sample_chunk]

    monkeypatch.setattr(qa, "retrieve_chunks", _fake_retrieve_chunks)
    monkeypatch.setattr(qa, "call_llm_with_citations", _fake_call_llm)

    response = client.post(
        "/ask/",
        json={
            "query": "Which PET tracer is collected as part of the regimen?",
            "nct_id": "NCT06113705",
        },
    )

    assert response.status_code == 200
    data = response.json()
    expected = "DIAGNOSTIC_TEST: 18F-GE-180 PET"
    assert data["answer"] == expected


def test_alignment_captures_fgfr_requirement(monkeypatch):
    sample_chunk = {
        "nct_id": "NCT06308822",
        "section": "Eligibility.Inclusion",
        "text": (
            "Patients must meet the disease requirements as outlined in MATCH Master "
            "Protocol at the time of registration to treatment step (Step 1, 3, 5, 7). "
            "Patients must have FGFR Amplification as determined via the MATCH Master "
            "Protocol. Patients must have an electrocardiogram (ECG) within 8 weeks "
            "prior to treatment."
        ),
    }

    def _fake_retrieve_chunks(query, nct_id):
        return [sample_chunk]

    def _fake_call_llm(query, chunks):
        return "FGFR Amplification as determined via the MATCH Master Protocol.", [
            sample_chunk
        ]

    monkeypatch.setattr(qa, "retrieve_chunks", _fake_retrieve_chunks)
    monkeypatch.setattr(qa, "call_llm_with_citations", _fake_call_llm)

    response = client.post(
        "/ask/",
        json={
            "query": "What biomarker must patients carry?",
            "nct_id": "NCT06308822",
        },
    )

    assert response.status_code == 200
    data = response.json()
    expected = (
        "Patients must have FGFR Amplification as determined via the MATCH Master "
        "Protocol."
    )
    assert data["answer"] == expected


def test_alignment_prefers_caregiver_clause(monkeypatch):
    sample_chunk = {
        "nct_id": "NCT06989086",
        "section": "Eligibility.Inclusion",
        "text": (
            "Patients: Self-report a diagnosis of a primary malignant brain tumor "
            "(grade II-IV) >2 weeks post-cranial resection or biopsy Elevated Fear of "
            "Recurrence Distress Rating Primarily English speaking >= 18 years of age "
            "at the time of enrollment Caregivers: nonprofessional caregiver to a "
            "patient with a primary malignant brain tumor (grade II-IV) Elevated Fear "
            "of Recurrence Distress Rating Primarily English speaking >= 18 years of "
            "age at the time of enrollment"
        ),
    }

    def _fake_retrieve_chunks(query, nct_id):
        return [sample_chunk]

    def _fake_call_llm(query, chunks):
        answer = (
            "Caregivers can enroll if they are a nonprofessional caregiver to a patient "
            "with a primary malignant brain tumor (grade II-IV), have an Elevated Fear "
            "of Recurrence Distress Rating, are primarily English speaking, and are 18 "
            "years of age at the time of enrollment."
        )
        return answer, [sample_chunk]

    monkeypatch.setattr(qa, "retrieve_chunks", _fake_retrieve_chunks)
    monkeypatch.setattr(qa, "call_llm_with_citations", _fake_call_llm)

    response = client.post(
        "/ask/",
        json={
            "query": "In trial NCT06989086, who can enroll as caregivers?",
            "nct_id": "NCT06989086",
        },
    )

    assert response.status_code == 200
    data = response.json()
    expected = (
        "Caregivers: nonprofessional caregiver to a patient with a primary malignant "
        "brain tumor (grade II-IV) Elevated Fear of Recurrence Distress Rating "
        "Primarily English speaking >= 18 years of age at the time of enrollment"
    )
    assert data["answer"] == expected


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
