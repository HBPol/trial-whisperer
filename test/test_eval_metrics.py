from pathlib import Path

import pytest

from eval.eval import (
    answer_exact_match,
    citations_match,
    compute_metrics,
    load_examples,
    normalize_answer,
)


def test_normalize_answer_handles_non_strings():
    assert normalize_answer(123) == "123"
    assert normalize_answer("  MiXeD Case  ") == "mixed case"


def test_answer_exact_match_with_whitespace_and_case():
    gold = ["Age ≥ 18 years", "Adults aged 18 and above"]
    assert answer_exact_match(" age ≥ 18 YEARS ", gold)
    assert not answer_exact_match("Age 17", gold)


def test_citations_match_requires_all_sections_and_matching_trial():
    citations = [
        {
            "nct_id": "NCT00000000",
            "section": "eligibility.inclusion",
            "text_snippet": "Age ≥ 18 years",
        },
        {
            "nct_id": "NCT00000000",
            "section": "eligibility.exclusion",
            "text_snippet": "No prior therapy",
        },
    ]

    assert citations_match(citations, ["eligibility.inclusion"], "NCT00000000")
    assert not citations_match(citations, ["eligibility.summary"], "NCT00000000")
    assert not citations_match(
        [
            {
                "nct_id": "NCTX",
                "section": "eligibility.inclusion",
                "text_snippet": "Age ≥ 18 years",
            }
        ],
        ["eligibility.inclusion"],
        "NCT00000000",
    )
    assert citations_match(citations, [], "NCT00000000")


def test_compute_metrics_aggregates_counts_and_accuracy():
    records = [
        {
            "answer_exact_match": True,
            "citation_match": True,
            "expected_sections": ["eligibility.inclusion"],
        },
        {
            "answer_exact_match": False,
            "citation_match": False,
            "expected_sections": ["eligibility.exclusion"],
            "error": "404",
        },
        {
            "answer_exact_match": True,
            "citation_match": True,
            "expected_sections": [],
        },
    ]

    metrics = compute_metrics(records)

    assert metrics["total_examples"] == 3
    assert metrics["answer_exact_match"]["correct"] == 2
    assert metrics["answer_exact_match"]["total"] == 3
    assert metrics["answer_exact_match"]["accuracy"] == pytest.approx(2 / 3)

    citation_stats = metrics["citation_section_match"]
    assert citation_stats["total"] == 2
    assert citation_stats["correct"] == 1
    assert citation_stats["accuracy"] == pytest.approx(0.5)

    assert metrics["error_count"] == 1


def test_sample_dataset_has_expected_scope():
    examples = load_examples(Path("eval/testset.sample.jsonl"))

    assert len(examples) >= 20

    sections_seen = set()
    for example in examples:
        assert example.get("query")
        answers = example.get("answers")
        assert isinstance(answers, list) and answers
        sections = example.get("sections")
        assert isinstance(sections, list)
        sections_seen.update(sections)

    expected_sections = {
        "eligibility.inclusion",
        "eligibility.exclusion",
        "interventions",
        "outcomes",
        "condition",
        "title",
    }

    assert expected_sections.issubset(sections_seen)
