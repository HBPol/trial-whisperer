import sys
import types

from app.agents import tools
from app.agents.tools import call_llm_with_citations, check_eligibility


def _install_fake_genai(monkeypatch, client_cls):
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []  # mark as package
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = client_cls
    fake_errors = types.ModuleType("google.genai.errors")

    class ClientError(Exception):
        pass

    fake_errors.ClientError = ClientError
    fake_genai.errors = fake_errors
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.errors", fake_errors)


def test_check_eligibility_age_within_range():
    criteria = {"inclusion": ["Age 18 to 65"], "exclusion": []}
    result = check_eligibility(criteria, {"age": 35})

    assert result["eligible"] is True
    assert result["reasons"] == []


def test_check_eligibility_age_outside_range():
    criteria = {"inclusion": ["Age 18 to 65"], "exclusion": []}
    result = check_eligibility(criteria, {"age": 70})

    assert result["eligible"] is False
    assert any("outside required range" in reason for reason in result["reasons"])


def test_check_eligibility_inclusion_minimum_age_phrase_rejects_underage():
    criteria = {"inclusion": ["Be ≥ 18 years of age"], "exclusion": []}
    result = check_eligibility(criteria, {"age": 17})

    assert result["eligible"] is False
    assert any(
        "Be ≥ 18 years of age" in reason and "Age 17" in reason
        for reason in result["reasons"]
    )


def test_check_eligibility_inclusion_between_phrase_rejects_overage():
    criteria = {
        "inclusion": ["Subject is between 18 and 89 years of age"],
        "exclusion": [],
    }
    result = check_eligibility(criteria, {"age": 92})

    assert result["eligible"] is False
    assert any(
        "Subject is between 18 and 89 years of age" in reason and "Age 92" in reason
        for reason in result["reasons"]
    )


def test_check_eligibility_missing_age():
    criteria = {"inclusion": ["Age 18 to 65"], "exclusion": []}
    result = check_eligibility(criteria, {})

    assert result["eligible"] is False
    assert any("Missing age information" in reason for reason in result["reasons"])


def test_check_eligibility_sex_only_rule_allows_matching_patient():
    criteria = {"inclusion": ["Female participants only"], "exclusion": []}
    result = check_eligibility(criteria, {"sex": "Female"})

    assert result["eligible"] is True
    assert result["reasons"] == []


def test_check_eligibility_sex_only_rule_blocks_mismatch():
    criteria = {"inclusion": ["Female participants only"], "exclusion": []}
    result = check_eligibility(criteria, {"sex": "Male"})

    assert result["eligible"] is False
    assert any("not permitted" in reason for reason in result["reasons"])


def test_check_eligibility_exclusion_age_range_blocks_patient():
    criteria = {"inclusion": [], "exclusion": ["Age 30 to 40"]}
    result = check_eligibility(criteria, {"age": 35})

    assert result["eligible"] is False
    assert any("exclusion criterion" in reason for reason in result["reasons"])


def test_call_llm_with_citations_gemini_success(monkeypatch):
    class FakeClient:
        last_instance = None

        def __init__(self, api_key):
            assert api_key == "test-key"
            self.calls = []
            FakeClient.last_instance = self

            class _Models:
                def __init__(self, parent):
                    self._parent = parent

                def generate_content(self, **kwargs):
                    self._parent.calls.append(kwargs)

                    class _Candidate:
                        def __init__(self):
                            part = types.SimpleNamespace(text="Answer about the trial")
                            self.content = types.SimpleNamespace(parts=[part])

                    return types.SimpleNamespace(candidates=[_Candidate()])

            self.models = _Models(self)

    _install_fake_genai(monkeypatch, FakeClient)
    monkeypatch.setattr(tools.settings, "llm_provider", "gemini", raising=False)
    monkeypatch.setattr(tools.settings, "llm_api_key", "test-key", raising=False)
    monkeypatch.setattr(tools.settings, "llm_model", "gemini-1.5-flash", raising=False)

    chunks = [
        {"nct_id": "NCT0001", "section": "Summary", "text": "Study summary."},
        {"nct_id": "NCT0002", "section": "Details", "text": "More info."},
    ]

    answer, citations = call_llm_with_citations("What is studied?", chunks)

    assert answer == "Answer about the trial"
    assert citations == chunks[:3]
    assert FakeClient.last_instance.calls
    payload = FakeClient.last_instance.calls[0]
    assert payload["model"] == "gemini-1.5-flash"
    assert isinstance(payload["contents"], list)
    assert len(payload["contents"]) == 2
    instruction_item, user_message = payload["contents"]

    assert isinstance(instruction_item, str)
    expected_instruction = tools._get_qa_system_prompt()
    assert instruction_item == expected_instruction
    assert "direct answer text" in instruction_item.lower()

    assert user_message["role"] == "user"
    assert isinstance(user_message["parts"], list)
    user_text = user_message["parts"][0]["text"]
    assert user_text.startswith(
        "Context:\n(1) [Trial NCT0001] Summary: Study summary.\n(2) [Trial NCT0002]"
    )
    assert "Details: More info." in user_text
    assert "Question: What is studied?" in user_text


def test_call_llm_with_citations_gemini_error(monkeypatch):
    class ErrorClient:
        def __init__(self, api_key):
            from google.genai import errors

            raise errors.ClientError("boom")

    _install_fake_genai(monkeypatch, ErrorClient)
    monkeypatch.setattr(tools.settings, "llm_provider", "gemini", raising=False)
    monkeypatch.setattr(tools.settings, "llm_api_key", "test-key", raising=False)
    monkeypatch.setattr(tools.settings, "llm_model", None, raising=False)

    chunks = [
        {"nct_id": "NCT0001", "section": "Summary", "text": "Study summary."},
        {"nct_id": "NCT0002", "section": "Details", "text": "More info."},
        {"nct_id": "NCT0003", "section": "Extra", "text": "Even more."},
        {"nct_id": "NCT0004", "section": "Other", "text": "Other info."},
    ]

    expected_chunks, _ = tools._select_chunks_for_context(chunks)

    answer, citations = call_llm_with_citations("What is studied?", chunks)

    assert answer == tools._provider_unavailable_answer(len(expected_chunks))
    assert citations == expected_chunks[:3]


def test_call_llm_with_citations_truncates_context(monkeypatch):
    class RecordingClient:
        last_instance = None

        def __init__(self, api_key):
            assert api_key == "test-key"
            self.calls = []
            RecordingClient.last_instance = self

            class _Models:
                def __init__(self, parent):
                    self._parent = parent

                def generate_content(self, **kwargs):
                    self._parent.calls.append(kwargs)

                    part = types.SimpleNamespace(text="Trimmed answer")
                    candidate = types.SimpleNamespace(
                        content=types.SimpleNamespace(parts=[part])
                    )
                    return types.SimpleNamespace(candidates=[candidate])

            self.models = _Models(self)

    _install_fake_genai(monkeypatch, RecordingClient)
    monkeypatch.setattr(tools.settings, "llm_provider", "gemini", raising=False)
    monkeypatch.setattr(tools.settings, "llm_api_key", "test-key", raising=False)
    monkeypatch.setattr(tools.settings, "llm_model", "gemini-1.5-flash", raising=False)
    monkeypatch.setattr(tools, "DEFAULT_CONTEXT_CHAR_BUDGET", 120)

    chunks = [
        {
            "nct_id": f"NCT{i:04d}",
            "section": "Summary",
            "text": (chr(ord("A") + i) * (140 - i * 5)),
            "score": 1.0 - i * 0.05,
        }
        for i in range(6)
    ]

    expected_chunks, expected_context = tools._select_chunks_for_context(
        chunks, max_chars=120
    )

    answer, citations = call_llm_with_citations("Summarise the study", chunks)

    assert answer == "Trimmed answer"
    assert citations == expected_chunks[:3]
    assert RecordingClient.last_instance.calls

    payload = RecordingClient.last_instance.calls[0]
    user_text = payload["contents"][1]["parts"][0]["text"]
    context_section, _ = user_text.split("\n\nQuestion:", 1)
    assert context_section.startswith("Context:\n")
    context_body = context_section[len("Context:\n") :]

    assert context_body == expected_context
    assert len(context_body) <= 120
    if expected_context:
        assert any(
            part.endswith("…") or len(part) < len(chunks[0]["text"])
            for part in context_body.split("\n")
        )


def test_call_llm_with_citations_openai_provider_error(monkeypatch):
    class FakeRateLimitError(Exception):
        pass

    class FakeCompletions:
        def create(self, **kwargs):
            raise FakeRateLimitError("rate limited")

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self, api_key):
            assert api_key == "openai-key"
            self.chat = FakeChat()

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeClient
    fake_openai.RateLimitError = FakeRateLimitError

    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(
        tools,
        "_get_openai_error_types",
        lambda: (FakeRateLimitError,),
    )

    monkeypatch.setattr(tools.settings, "llm_provider", "openai", raising=False)
    monkeypatch.setattr(tools.settings, "llm_api_key", "openai-key", raising=False)
    monkeypatch.setattr(tools.settings, "llm_model", "gpt-3.5-turbo", raising=False)

    chunks = [
        {"nct_id": "NCT1000", "section": "Summary", "text": "Alpha details."},
        {"nct_id": "NCT1001", "section": "Details", "text": "Beta summary."},
        {"nct_id": "NCT1002", "section": "Extra", "text": "Gamma context."},
    ]

    expected_chunks, _ = tools._select_chunks_for_context(chunks)

    answer, citations = call_llm_with_citations("What is studied?", chunks)
    assert answer == tools._provider_unavailable_answer(len(expected_chunks))
    assert citations == expected_chunks[:3]
