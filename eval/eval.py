"""Evaluation utilities for the TrialWhisperer question answering API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.deps import get_settings
from app.main import app
from app.retrieval import search_client, trial_store

DEFAULT_TESTSET_PATH = Path("eval/testset.sample.jsonl")
DEFAULT_TRIALS_DATA_PATH = Path(".data/processed/trials.jsonl")

MAX_REQUEST_ATTEMPTS = 3


def enforce_min_request_interval(
    last_request_time: float | None, min_interval: float
) -> None:
    """Sleep until ``min_interval`` seconds have elapsed since the last call."""

    if min_interval <= 0 or last_request_time is None:
        return
    remaining = (last_request_time + min_interval) - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def parse_retry_after(value: str | None) -> float | None:
    """Return the retry delay encoded in a ``Retry-After`` header, if valid."""

    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            retry_time = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_time is None:
            return None
        if retry_time.tzinfo is None:
            retry_time = retry_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delay = (retry_time - now).total_seconds()
    if delay < 0:
        return 0.0
    return delay


def get_retry_after_delay(headers: Mapping[str, str]) -> float | None:
    """Look for a retry delay in the response headers."""

    header_value = headers.get("retry-after")
    if header_value is None:
        header_value = headers.get("Retry-After")
    return parse_retry_after(header_value)


def load_examples(path: Path) -> List[Dict[str, Any]]:
    """Load evaluation examples from a JSONL file."""

    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def normalize_answer(text: Any) -> str:
    """Return a normalized string representation suitable for comparison."""

    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.strip().lower().split())


def answer_exact_match(prediction: Any, gold_answers: Sequence[Any]) -> bool:
    """Check if ``prediction`` matches any gold answer after normalization."""

    if not gold_answers:
        return False
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return False
    return any(normalized_prediction == normalize_answer(ans) for ans in gold_answers)


def citations_match(
    citations: Sequence[Dict[str, Any]],
    expected_sections: Sequence[str],
    expected_nct_id: str | None,
) -> bool:
    """Verify returned citations cover the expected sections and trial."""

    if not expected_sections:
        return True
    if not citations:
        return False

    sections_seen = set()
    for citation in citations:
        section = citation.get("section")
        if section:
            sections_seen.add(section)
        if expected_nct_id and citation.get("nct_id") != expected_nct_id:
            return False
    return all(section in sections_seen for section in expected_sections)


def evaluate_examples(
    client: TestClient,
    examples: Sequence[Dict[str, Any]],
    *,
    min_request_interval: float = 0.0,
) -> List[Dict[str, Any]]:
    """Call the QA endpoint for each example and capture predictions."""

    records: List[Dict[str, Any]] = []
    last_request_time: float | None = None
    for example in examples:
        query = example.get("query")
        nct_id = example.get("nct_id")
        gold_answers = example.get("answers") or []
        expected_sections = example.get("sections") or []

        record: Dict[str, Any] = {
            "query": query,
            "nct_id": nct_id,
            "gold_answers": gold_answers,
            "expected_sections": expected_sections,
        }

        payload = {"query": query, "nct_id": nct_id}
        payload = {k: v for k, v in payload.items() if v}

        attempt = 0
        response = None
        data: Dict[str, Any] | None = None
        error_message: str | None = None

        while attempt < MAX_REQUEST_ATTEMPTS:
            attempt += 1
            if min_request_interval > 0:
                enforce_min_request_interval(last_request_time, min_request_interval)
            try:
                response = client.post("/ask/", json=payload)
            except Exception as exc:  # pragma: no cover - network/client failures
                last_request_time = time.monotonic()
                error_message = str(exc)
                response = None
                break

            last_request_time = time.monotonic()

            if response.status_code == 429 and attempt < MAX_REQUEST_ATTEMPTS:
                retry_delay = get_retry_after_delay(response.headers)
                if retry_delay and retry_delay > 0:
                    time.sleep(retry_delay)
                continue
            if response.status_code != 200:
                error_message = response.text
                break

            try:
                data = response.json()
            except ValueError as exc:  # pragma: no cover - unexpected response format
                error_message = str(exc)
                data = None
                break

            answer = data.get("answer", "")
            if isinstance(answer, str) and answer.startswith(("[FALLBACK]", "[DEMO]")):
                error_message = f"Fallback answer received: {answer}"
                data = None
                if attempt < MAX_REQUEST_ATTEMPTS:
                    continue
                break

            error_message = None
            break

        if response is None:
            record.update(
                {
                    "answer": None,
                    "citations": [],
                    "error": error_message,
                    "status_code": None,
                    "answer_exact_match": False,
                    "citation_match": False,
                }
            )
            records.append(record)
            continue

        record["status_code"] = response.status_code
        if response.status_code != 200 or data is None:
            record.update(
                {
                    "answer": None,
                    "citations": [],
                    "error": error_message or response.text,
                    "answer_exact_match": False,
                    "citation_match": False,
                }
            )
            records.append(record)
            continue

        answer = data.get("answer", "")
        citations = data.get("citations") or []
        if not isinstance(citations, list):
            citations = []

        record.update(
            {
                "answer": answer,
                "citations": citations,
                "answer_exact_match": answer_exact_match(answer, gold_answers),
                "citation_match": citations_match(citations, expected_sections, nct_id),
            }
        )
        records.append(record)

    return records


def compute_metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate accuracy metrics from evaluated examples."""

    total = len(records)
    answer_correct = sum(1 for record in records if record.get("answer_exact_match"))
    citation_applicable = sum(
        1 for record in records if record.get("expected_sections")
    )
    citation_correct = sum(
        1
        for record in records
        if record.get("expected_sections") and record.get("citation_match")
    )
    error_count = sum(1 for record in records if record.get("error"))

    metrics = {
        "total_examples": total,
        "answer_exact_match": {
            "correct": answer_correct,
            "total": total,
            "accuracy": (answer_correct / total) if total else None,
        },
        "citation_section_match": {
            "correct": citation_correct,
            "total": citation_applicable,
            "accuracy": (
                (citation_correct / citation_applicable)
                if citation_applicable
                else None
            ),
        },
        "error_count": error_count,
    }
    return metrics


def print_summary(metrics: Dict[str, Any]) -> None:
    """Emit a concise console summary of evaluation metrics."""

    total = metrics.get("total_examples", 0)
    answers = metrics.get("answer_exact_match", {})
    citations = metrics.get("citation_section_match", {})

    print(f"Evaluated {total} examples")
    answer_accuracy = answers.get("accuracy")
    if answer_accuracy is not None:
        print(
            "Answer exact match: "
            f"{answers.get('correct', 0)}/{answers.get('total', 0)} "
            f"({answer_accuracy:.1%})"
        )
    else:
        print("Answer exact match: n/a")

    citation_total = citations.get("total", 0)
    citation_accuracy = citations.get("accuracy")
    if citation_total and citation_accuracy is not None:
        print(
            "Citation coverage: "
            f"{citations.get('correct', 0)}/{citation_total} "
            f"({citation_accuracy:.1%})"
        )
    else:
        print("Citation coverage: n/a (no expected sections)")

    errors = metrics.get("error_count", 0)
    if errors:
        print(f"Errors encountered: {errors}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(DEFAULT_TESTSET_PATH),
        help="Path to the evaluation dataset (JSONL).",
    )
    parser.add_argument(
        "--json-report",
        dest="json_report",
        type=str,
        help="Optional file path to write the JSON report to.",
    )
    parser.add_argument(
        "--trials-data",
        dest="trials_data",
        type=str,
        default=None,
        help=(
            "Path to the processed trials dataset used by the fallback search. "
            "Defaults to '.data/processed/trials.jsonl' when running the sample "
            "evaluation."
        ),
    )
    parser.add_argument(
        "--min-request-interval",
        dest="min_request_interval",
        type=float,
        default=None,
        help=(
            "Minimum delay in seconds between /ask/ requests. "
            "Overrides provider-specific defaults."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = Path(args.dataset)

    trials_data_path: Path | None
    if args.trials_data:
        trials_data_path = Path(args.trials_data)
    elif dataset_path == DEFAULT_TESTSET_PATH:
        trials_data_path = DEFAULT_TRIALS_DATA_PATH
    else:
        trials_data_path = None

    if trials_data_path:
        os.environ[trial_store.TRIALS_DATA_ENV_VAR] = str(trials_data_path)

    trial_store.clear_trials_cache()
    search_client.clear_fallback_index()

    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    examples = load_examples(dataset_path)

    settings = get_settings()
    min_request_interval = 0.0
    if (settings.llm_provider or "").lower() == "gemini":
        min_request_interval = 6.0
    if args.min_request_interval is not None:
        min_request_interval = max(args.min_request_interval, 0.0)

    with TestClient(app) as client:
        records = evaluate_examples(
            client,
            examples,
            min_request_interval=min_request_interval,
        )

    metrics = compute_metrics(records)
    report = {
        "dataset": str(dataset_path),
        "metrics": metrics,
        "examples": records,
    }

    print_summary(metrics)

    report_json = json.dumps(report, indent=2, sort_keys=True)
    if args.json_report:
        report_path = Path(args.json_report)
        report_path.write_text(report_json + "\n", encoding="utf-8")
        print(f"Wrote JSON report to {report_path}")
    else:
        print(report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
