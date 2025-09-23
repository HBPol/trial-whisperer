from __future__ import annotations

from typing import Any, Dict, List

from eval.eval import MAX_REQUEST_ATTEMPTS, evaluate_examples


class DummyResponse:
    def __init__(
        self,
        status_code: int,
        json_payload: Dict[str, Any] | None = None,
        *,
        text: str = "",
        headers: Dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_payload = json_payload or {}
        self.text = text
        self.headers = headers or {}

    def json(self) -> Dict[str, Any]:
        return self._json_payload


class DummyClient:
    def __init__(self, responses: List[DummyResponse]) -> None:
        self._responses = responses
        self.calls = 0

    def post(
        self, url: str, json: Dict[str, Any]
    ) -> DummyResponse:  # pragma: no cover - url unused in tests
        if self.calls >= len(self._responses):
            raise AssertionError("No more responses configured for DummyClient")
        response = self._responses[self.calls]
        self.calls += 1
        return response


def make_example(**overrides: Any) -> Dict[str, Any]:
    example = {
        "query": "What is the intervention?",
        "nct_id": "NCT12345678",
        "answers": ["example answer"],
        "sections": [],
    }
    example.update(overrides)
    return example


def test_evaluate_examples_retries_and_succeeds_after_fallback() -> None:
    responses = [
        DummyResponse(
            200, {"answer": "[FALLBACK] Using local search", "citations": []}
        ),
        DummyResponse(
            200,
            {
                "answer": "Example Answer",
                "citations": [
                    {"section": "eligibility.inclusion", "nct_id": "NCT12345678"}
                ],
            },
        ),
    ]
    client = DummyClient(responses)

    examples = [make_example(answers=["example answer", "example answer".title()])]

    records = evaluate_examples(client, examples)

    assert client.calls == 2
    assert len(records) == 1
    record = records[0]
    assert record["answer"] == "Example Answer"
    assert not record.get("error")
    assert record["status_code"] == 200


def test_evaluate_examples_marks_error_if_all_attempts_fallback() -> None:
    responses = [
        DummyResponse(200, {"answer": "[FALLBACK] Using local search", "citations": []})
        for _ in range(MAX_REQUEST_ATTEMPTS)
    ]
    client = DummyClient(responses)

    examples = [make_example()]

    records = evaluate_examples(client, examples)

    assert client.calls == MAX_REQUEST_ATTEMPTS
    record = records[0]
    assert record["status_code"] == 200
    assert record["answer"] is None
    assert record["citations"] == []
    assert record["error"]
    assert "Fallback answer" in record["error"]
    assert not record["answer_exact_match"]
