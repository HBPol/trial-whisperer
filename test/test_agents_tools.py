import sys
import types

from app.agents.tools import call_llm_with_citations, check_eligibility


def _install_fake_genai(monkeypatch, client_cls):
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []  # mark as package
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = client_cls
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)


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
    from app.agents import tools

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
                    part = types.SimpleNamespace(text="Answer about the trial")
                    content = types.SimpleNamespace(parts=[part])
                    candidate = types.SimpleNamespace(content=content)
                    return types.SimpleNamespace(
                        candidates=[candidate], output_text="Answer about the trial"
                    )

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
    assert payload["contents"][0]["role"] == "system"
    assert payload["contents"][1]["role"] == "user"


def test_call_llm_with_citations_gemini_error(monkeypatch):
    from app.agents import tools

    class ErrorClient:
        def __init__(self, api_key):
            raise RuntimeError("boom")

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

    answer, citations = call_llm_with_citations("What is studied?", chunks)

    assert answer.startswith("[LLM error]")
    assert citations == chunks[:3]
