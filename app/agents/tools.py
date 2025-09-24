import importlib
import logging
import re
import time
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

from app.deps import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_CHAR_BUDGET = 24000
ELLIPSIS = "\u2026"
QA_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "qa_system.txt"
)
DEFAULT_QA_SYSTEM_PROMPT = (
    "You are TrialWhisperer, a clinical-trial protocol assistant. "
    "Answer only from the provided passages. If the passages do not contain the "
    "answer, say you don't know. Respond with only the direct answer text "
    "without inline citation markers or numbering; structured citations are "
    "handled separately."
)

_CITATION_MARKER_PATTERN = re.compile(r"\s*(?:\(\s*\d+\s*\)|\[\s*\d+\s*\])")
_KEYWORD_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_FRAGMENT_SPLIT_PATTERN = re.compile(r"[\n;]+|(?<!\d)\.(?!\d)")
_LABEL_SEGMENT_PATTERN = re.compile(r"([A-Z][A-Za-z0-9 _/\-]{1,30}:)")
_SECONDARY_REQUIREMENT_SPLIT_PATTERN = re.compile(
    r"\s+(?=(?:Able|Agreement|Completed|Currently|Documented|Elevated|Eligible|Exclusion|"
    r"Female|Have|History|Inclusion|Male|Must|Need|Needs|No|Not|Primarily|Provide|Requires?|"
    r"Should|Willing|Without|Women|Men|ECOG|Karnofsky|NYHA|BMI|ANC|Platelet|Hemoglobin|"
    r"Creatinine|AST|ALT|Bilirubin|INR|QTc|Blood|Systolic|Diastolic|Glucose|Pregnant|"
    r"Pregnancy|Contraception)\b)"
)
_ANSWER_LEADING_PATTERNS = [
    re.compile(r"^\s*(?:answer|final answer)\s*[:\-]\s*", re.IGNORECASE),
    re.compile(
        r"^\s*(?:based on|according to|from|using) (?:the )?(?:provided )?context"
        r"(?: (?:above|given))?\s*(?:[,:\-]|that)\s*",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*in summary\s*[:\-]\s*", re.IGNORECASE),
    re.compile(r"^\s*overall\s*[:\-]\s*", re.IGNORECASE),
    re.compile(r"^\s*this means\s*[:\-]\s*", re.IGNORECASE),
]

_ANSWER_WRAPPER_PATTERNS = [
    re.compile(
        r"^\s*(?:the\s+official\s+title\s*(?:of\s+[\w\-]+)?\s*(?:is|:))\s*",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(?:official\s+title\s*(?:is|:))\s*", re.IGNORECASE),
    re.compile(r"^\s*(?:the\s+answer\s*(?:is|:))\s*", re.IGNORECASE),
]

_LEADING_LIST_NUMERAL_PATTERN = re.compile(r"^\s*(?:\(\d+\)|\d+)[\.)]\s+")

_FALLBACK_ANSWER_MARKERS = (
    "i don't know",
    "i do not know",
    "don't know",
    "do not know",
    "not provided",
    "not specified",
    "not sure",
    "no information",
    "information not provided",
    "not available",
    "no answer",
    "unsure",
)


@lru_cache(maxsize=1)
def _get_qa_system_prompt() -> str:
    """Load the QA system prompt, ensuring direct-answer instructions."""

    prompt_text = DEFAULT_QA_SYSTEM_PROMPT
    try:
        file_prompt = QA_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        file_prompt = ""
    if file_prompt:
        prompt_text = file_prompt

    normalized_prompt = prompt_text.strip()
    directive = (
        "respond with only the direct answer text without inline citation markers"
    )
    if directive not in normalized_prompt.lower():
        normalized_prompt = (
            f"{normalized_prompt} Respond with only the direct answer text without "
            "inline citation markers or numbering; structured citations are "
            "handled separately."
        ).strip()
    return normalized_prompt


def _strip_leading_phrases(text: str) -> str:
    """Remove standard leading phrases used by LLM answers."""

    current = text
    while True:
        updated = current
        for pattern in _ANSWER_LEADING_PATTERNS:
            updated = pattern.sub("", updated)
        updated = updated.lstrip()
        if updated == current:
            return updated
        current = updated


def clean_answer_text(answer: Any) -> str:
    """Return ``answer`` with extraneous markers and boilerplate removed."""

    if answer is None:
        return ""

    original = str(answer).strip()
    if not original:
        return ""

    cleaned = _CITATION_MARKER_PATTERN.sub("", original)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = _strip_leading_phrases(cleaned)
    cleaned = cleaned.lstrip("-:;, ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned or original


def prepare_answer_text(answer: Any) -> str:
    """Normalize ``answer`` for downstream alignment and evaluation."""

    cleaned = clean_answer_text(answer)
    if not cleaned:
        return cleaned

    trimmed = cleaned
    for pattern in _ANSWER_WRAPPER_PATTERNS:
        trimmed = pattern.sub("", trimmed).strip()

    # Remove surrounding quotes introduced by explanatory phrasing.
    if (
        len(trimmed) >= 2
        and trimmed[0] in {'"', "'", "“", "”"}
        and trimmed[-1] in {'"', "'", "“", "”"}
    ):
        trimmed = trimmed[1:-1].strip()

    # Some wrappers append trailing punctuation after trimming, so clean again.
    trimmed = re.sub(r"\s+", " ", trimmed).strip()
    return trimmed or cleaned


def _format_chunk_prefix(chunk: dict, index: int) -> str:
    """Return the static prefix used when rendering a chunk in the context."""

    nct_id = chunk.get("nct_id", "unknown")
    section = chunk.get("section", "Context")
    return f"({index}) [Trial {nct_id}] {section}: "


def _format_chunk_line(chunk: dict, index: int) -> str:
    """Return the formatted line for ``chunk`` including the numbered prefix."""

    text = chunk.get("text", "")
    return f"{_format_chunk_prefix(chunk, index)}{text}"


def _format_context(chunks: List[dict]) -> str:
    """Create a numbered context block from retrieved chunks."""

    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        parts.append(_format_chunk_line(chunk, idx))
    return "\n".join(parts)


def _truncate_text(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` characters adding an ellipsis when truncated."""

    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    trimmed = text[: max(0, limit - len(ELLIPSIS))].rstrip()
    if not trimmed:
        trimmed = text[: max(0, limit - len(ELLIPSIS))]
    truncated = f"{trimmed}{ELLIPSIS}"
    return truncated[:limit]


def _score_key(item: Tuple[int, dict]) -> Tuple[float, int]:
    index, chunk = item
    score = chunk.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return (-float(score), index)
    return (0.0, index)


def _select_chunks_for_context(
    chunks: List[dict],
    *,
    max_chars: int = DEFAULT_CONTEXT_CHAR_BUDGET,
) -> Tuple[List[dict], str]:
    """Return the highest scoring chunks whose formatted text fits ``max_chars``."""

    if not chunks or max_chars is None or max_chars <= 0:
        return [], ""

    ordered = [chunk for _, chunk in sorted(enumerate(chunks), key=_score_key)]

    selected: List[dict] = []
    formatted_parts: List[str] = []
    current_length = 0

    for chunk in ordered:
        next_index = len(selected) + 1
        prefix = _format_chunk_prefix(chunk, next_index)
        text = str(chunk.get("text", ""))
        line = f"{prefix}{text}"
        newline_cost = 1 if formatted_parts else 0
        projected = current_length + newline_cost + len(line)

        if projected <= max_chars:
            selected.append(chunk)
            formatted_parts.append(line)
            current_length = projected
            continue

        # Attempt to include a truncated version of the chunk when possible.
        available = max_chars - current_length - newline_cost
        if available <= len(prefix):
            break
        text_budget = available - len(prefix)
        truncated_text = _truncate_text(text, text_budget)
        if not truncated_text:
            break
        truncated_line = f"{prefix}{truncated_text}"
        selected.append(chunk)
        formatted_parts.append(truncated_line)
        current_length = current_length + newline_cost + len(truncated_line)
        break

    context_text = "\n".join(formatted_parts)
    return selected, context_text


def _normalize_for_match(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\u00a0", " ")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


_FALLBACK_NORMALIZED_MARKERS = {
    _normalize_for_match(marker) for marker in _FALLBACK_ANSWER_MARKERS
}


def _extract_answer_fragments(answer: str) -> List[str]:
    fragments: List[str] = []
    if not answer:
        return fragments
    for part in _FRAGMENT_SPLIT_PATTERN.split(answer):
        normalized = _normalize_for_match(part)
        if normalized:
            fragments.append(normalized)
    return fragments


def align_answer_to_context(
    answer: str,
    context_chunks: List[dict],
    *,
    query: Optional[str] = None,
) -> str:
    """Return the closest matching chunk text for ``answer`` when available."""

    if not answer:
        return answer
    if not context_chunks:
        return answer

    stripped_answer = answer.strip()
    if not stripped_answer:
        return stripped_answer

    lowered = stripped_answer.lower()
    if lowered.startswith("[fallback]") or lowered.startswith("[demo]"):
        return stripped_answer

    cleaned_reference = clean_answer_text(stripped_answer) or stripped_answer
    reference_length = len(cleaned_reference)
    if reference_length <= 0:
        reference_length = len(stripped_answer)

    normalized_answer = _normalize_for_match(cleaned_reference)
    fragments = _extract_answer_fragments(cleaned_reference)
    answer_tokens = _chunk_keyword_tokens(cleaned_reference)
    answer_token_set = set(answer_tokens)
    query_tokens = _chunk_keyword_tokens(query) if query else Counter()
    query_token_set = set(query_tokens)
    if not (normalized_answer or fragments or answer_token_set or query_token_set):
        return stripped_answer

    fallback_answer = False
    lowered_clean = cleaned_reference.lower()
    for marker in _FALLBACK_ANSWER_MARKERS:
        if marker in lowered_clean:
            fallback_answer = True
            break
    if not fallback_answer and normalized_answer:
        if normalized_answer in _FALLBACK_NORMALIZED_MARKERS:
            fallback_answer = True

    reference_variants: List[str] = []
    seen_variants: set[str] = set()

    for candidate in (cleaned_reference, stripped_answer):
        if not candidate:
            continue
        base = candidate.strip()
        if not base:
            continue
        for option in (base, base.rstrip(" \t.;:,")):
            if not option:
                continue
            key = option.lower()
            if key in seen_variants:
                continue
            reference_variants.append(option)
            seen_variants.add(key)

    allowed_prefix_roots = {
        "patient",
        "patients",
        "subject",
        "subjects",
        "participant",
        "participants",
        "caregiver",
        "caregivers",
        "cohort",
        "cohorts",
        "arm",
        "arms",
        "eligible",
        "have",
        "has",
        "must",
        "be",
        "is",
        "are",
        "measurable",
    }

    qualifying_suffix_markers = (
        "as determined",
        "as defined",
        "as assessed",
        "as measured",
        "as documented",
        "as confirmed",
        "as outlined",
        "per ",
        "per the",
        "per protocol",
        "according to",
        "within ",
        "prior to",
        "no more than",
        "not more than",
        "no less than",
        "at least",
        "at most",
        "on or after",
        "on or before",
    )

    def _is_valid_prefix(prefix_text: str) -> bool:
        if not prefix_text:
            return False
        stripped = prefix_text.strip()
        if not stripped:
            return False
        if re.fullmatch(r"(?:\(\d+\)|\d+)[\.)]\s*", stripped):
            return True
        if re.fullmatch(r"[A-Z0-9 _/\-]{1,30}:", stripped):
            return True
        lower = stripped.lower()
        words = lower.split()
        if not words:
            return False
        first_word = words[0].rstrip(":'s")
        return first_word in allowed_prefix_roots

    total_token_count = sum(answer_tokens.values())
    total_query_token_count = sum(query_tokens.values())
    allow_query_only_matches = fallback_answer or total_token_count == 0
    max_token_overlap = 0

    def _split_label_segments(text: str) -> List[str]:
        if not text or ":" not in text:
            return []

        segments: List[str] = []
        seen_segments: set[str] = set()
        positions: List[int] = []

        for match in _LABEL_SEGMENT_PATTERN.finditer(text):
            prefix = match.group(1)
            if not prefix:
                continue
            if not _is_valid_prefix(prefix):
                continue
            start = match.start()
            if start > 0:
                preceding = text[start - 1]
                if preceding not in " \t\r\n;,-":
                    continue
            positions.append(start)

        if len(positions) <= 1:
            return []

        first_start = positions[0]
        if first_start > 0:
            leading_segment = text[:first_start].strip()
            if leading_segment:
                segments.append(leading_segment)

        for idx, start in enumerate(positions):
            end = positions[idx + 1] if idx + 1 < len(positions) else len(text)
            candidate = text[start:end].strip()
            if candidate:
                lowered_candidate = candidate.lower()
                if lowered_candidate not in seen_segments:
                    segments.append(candidate)
                    seen_segments.add(lowered_candidate)
                for sub_segment in _split_secondary_requirements(candidate):
                    lowered_sub = sub_segment.lower()
                    if lowered_sub in seen_segments:
                        continue
                    segments.append(sub_segment)
                    seen_segments.add(lowered_sub)

        return segments

    def _split_secondary_requirements(segment: str) -> List[str]:
        if not segment:
            return []

        match = _LABEL_SEGMENT_PATTERN.match(segment)
        if not match:
            return []

        prefix_end = match.end()
        remainder = segment[prefix_end:]
        if not remainder:
            return []

        stripped_remainder = remainder.strip()
        if not stripped_remainder:
            return []

        parts = [
            part.strip()
            for part in _SECONDARY_REQUIREMENT_SPLIT_PATTERN.split(stripped_remainder)
            if part.strip()
        ]
        if len(parts) <= 1:
            return []

        split_segments: List[str] = []
        first_part = parts[0]
        first_segment = f"{segment[:prefix_end]} {first_part}".strip()
        if first_segment:
            split_segments.append(first_segment)

        for body_part in parts[1:]:
            if body_part:
                split_segments.append(body_part)

        return split_segments

    def _evaluate_candidate(candidate_text: str):
        candidate_stripped = candidate_text.strip()
        if not candidate_stripped:
            return None

        normalized_candidate = _normalize_for_match(candidate_stripped)
        if not normalized_candidate:
            return None

        chunk_tokens = _chunk_keyword_tokens(candidate_stripped)
        if not chunk_tokens:
            return None

        chunk_token_set = set(chunk_tokens)

        token_overlap = sum(
            min(count, chunk_tokens.get(token, 0))
            for token, count in answer_tokens.items()
        )
        unique_overlap = len(answer_token_set & chunk_token_set)
        query_token_overlap = sum(
            min(count, chunk_tokens.get(token, 0))
            for token, count in query_tokens.items()
        )
        query_unique_overlap = len(query_token_set & chunk_token_set)
        fragment_matches = sum(
            1 for fragment in fragments if fragment and fragment in normalized_candidate
        )
        span_match = (
            bool(normalized_answer) and normalized_answer in normalized_candidate
        )

        if (
            not span_match
            and fragment_matches == 0
            and unique_overlap <= 1
            and query_unique_overlap == 0
        ):
            if not allow_query_only_matches or not query_tokens:
                return None

        coverage_ratio = 0.0
        if total_token_count:
            coverage_ratio = token_overlap / total_token_count

        query_coverage_ratio = 0.0
        if total_query_token_count:
            query_coverage_ratio = query_token_overlap / total_query_token_count

        if (
            not span_match
            and fragment_matches == 0
            and coverage_ratio < 0.4
            and query_coverage_ratio < 0.4
        ):
            if not allow_query_only_matches or not query_tokens:
                return None

        total_candidate_tokens = sum(chunk_tokens.values())
        candidate_fraction = (
            token_overlap / total_candidate_tokens if total_candidate_tokens else 0.0
        )
        additional_token_count = max(0, total_candidate_tokens - token_overlap)
        additional_unique_tokens = len(chunk_token_set - answer_token_set)

        candidate_length = len(candidate_stripped)

        majority_query_overlap = 0
        query_focus_ratio = 0.0
        if total_candidate_tokens:
            if query_token_overlap * 2 >= total_candidate_tokens:
                majority_query_overlap = 1
            query_focus_ratio = query_token_overlap / total_candidate_tokens

        label_bonus = 0
        if re.match(r"^[A-Z][A-Za-z0-9 _/\-]{1,30}:\s*", candidate_stripped):
            if answer_token_set & chunk_token_set:
                label_bonus = 1

        score = (
            int(span_match),
            fragment_matches,
            int(query_focus_ratio * 1000),
            majority_query_overlap,
            query_unique_overlap,
            query_token_overlap,
            unique_overlap,
            token_overlap,
            label_bonus,
            -abs(candidate_length - reference_length),
            -candidate_length,
        )

        return {
            "text": candidate_stripped,
            "score": score,
            "length": candidate_length,
            "coverage_ratio": coverage_ratio,
            "token_overlap": token_overlap,
            "unique_overlap": unique_overlap,
            "query_token_overlap": query_token_overlap,
            "query_unique_overlap": query_unique_overlap,
            "query_coverage_ratio": query_coverage_ratio,
            "majority_query_overlap": majority_query_overlap,
            "query_focus_ratio": query_focus_ratio,
            "candidate_fraction": candidate_fraction,
            "additional_token_count": additional_token_count,
            "additional_unique_tokens": additional_unique_tokens,
            "total_candidate_tokens": total_candidate_tokens,
        }

    def _evaluate_query_only_candidate(candidate_text: str):
        candidate_stripped = candidate_text.strip()
        if not candidate_stripped:
            return None

        normalized_candidate = _normalize_for_match(candidate_stripped)
        if not normalized_candidate:
            return None

        chunk_tokens = _chunk_keyword_tokens(candidate_stripped)
        if not chunk_tokens:
            return None

        chunk_token_set = set(chunk_tokens)
        query_token_overlap = sum(
            min(count, chunk_tokens.get(token, 0))
            for token, count in query_tokens.items()
        )
        query_unique_overlap = len(query_token_set & chunk_token_set)

        if not (query_token_overlap or query_unique_overlap):
            if not allow_query_only_matches:
                return None
            if not query_tokens:
                return None

        total_candidate_tokens = sum(chunk_tokens.values())
        query_coverage_ratio = 0.0
        if total_query_token_count:
            query_coverage_ratio = query_token_overlap / total_query_token_count

        majority_query_overlap = 0
        query_focus_ratio = 0.0
        if total_candidate_tokens:
            if query_token_overlap * 2 >= total_candidate_tokens:
                majority_query_overlap = 1
            query_focus_ratio = query_token_overlap / total_candidate_tokens

        return {
            "text": candidate_stripped,
            "length": len(candidate_stripped),
            "coverage_ratio": query_coverage_ratio,
            "token_overlap": 0,
            "unique_overlap": 0,
            "query_token_overlap": query_token_overlap,
            "query_unique_overlap": query_unique_overlap,
            "query_coverage_ratio": query_coverage_ratio,
            "majority_query_overlap": majority_query_overlap,
            "query_focus_ratio": query_focus_ratio,
            "candidate_fraction": 0.0,
            "additional_token_count": total_candidate_tokens,
            "additional_unique_tokens": len(chunk_token_set),
            "total_candidate_tokens": total_candidate_tokens,
        }

    def _restore_fragment_text(candidate_eval: dict) -> str:
        text = candidate_eval.get("text")
        source_chunk = candidate_eval.get("source_chunk")
        if not text or not source_chunk:
            return text
        start_index = source_chunk.find(text)
        if start_index == -1:
            return text
        end_index = start_index + len(text)
        suffix = source_chunk[end_index:]
        match = re.match(r"^(\s*[.;])", suffix)
        if match:
            delimiter = match.group(0).strip()
            if delimiter:
                return f"{text}{delimiter[0]}"
        return text

    def _prepare_aligned_text(candidate_text: str) -> str:
        if not candidate_text:
            return candidate_text
        cleaned_candidate = candidate_text.strip()
        cleaned_candidate = _LEADING_LIST_NUMERAL_PATTERN.sub("", cleaned_candidate)
        return cleaned_candidate.strip()

    baseline_eval = _evaluate_candidate(stripped_answer)
    baseline_score: Tuple[int, int, int, int, int, int, int, int, int, int, int]
    if baseline_eval:
        raw_baseline_score = tuple(baseline_eval["score"])
        baseline_score = (0,) + raw_baseline_score[1:]
    else:
        baseline_score = (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -abs(len(stripped_answer) - reference_length),
            -len(stripped_answer),
        )

    best_candidate = stripped_answer
    best_score = baseline_score
    best_eval = baseline_eval

    seen_candidates = {stripped_answer}
    fallback_fragment_eval = None
    fallback_eval = None
    best_query_eval: Optional[Tuple[Tuple[Any, ...], dict]] = None
    best_label_query_candidate: Optional[Tuple[Tuple[Any, ...], dict]] = None

    def _maybe_update_label_candidate(candidate_eval: dict) -> None:
        nonlocal best_label_query_candidate
        if not candidate_eval:
            return
        text = candidate_eval.get("text")
        if not text:
            return
        match = _LABEL_SEGMENT_PATTERN.match(text)
        if not match:
            return
        prefix = match.group(1)
        if not prefix:
            return
        prefix_root = prefix.rstrip(": ").split()
        if not prefix_root:
            return
        root_word = prefix_root[0].lower().rstrip("'s")
        if not root_word:
            return
        candidate_tokens = {root_word}
        if root_word.endswith("s"):
            candidate_tokens.add(root_word.rstrip("s"))
        else:
            candidate_tokens.add(f"{root_word}s")
        if candidate_tokens.isdisjoint(query_token_set):
            return
        query_overlap = candidate_eval.get("query_token_overlap", 0)
        query_unique = candidate_eval.get("query_unique_overlap", 0)
        if not (query_overlap or query_unique):
            return
        focus_ratio = candidate_eval.get("query_focus_ratio", 0.0)
        candidate_length = candidate_eval.get("length", 0)
        label_score = (
            query_unique,
            query_overlap,
            int(focus_ratio * 1000),
            -candidate_length,
            candidate_eval.get("token_overlap", 0),
        )
        if (
            best_label_query_candidate is None
            or label_score > best_label_query_candidate[0]
        ):
            best_label_query_candidate = (label_score, candidate_eval)

    def _consider_query_candidate(
        candidate_eval: dict, source_chunk: Optional[str] = None
    ) -> None:
        nonlocal best_query_eval
        if not candidate_eval:
            return
        query_unique = candidate_eval.get("query_unique_overlap", 0)
        query_tokens = candidate_eval.get("query_token_overlap", 0)
        if not (query_unique or query_tokens):
            if not allow_query_only_matches:
                return

        candidate_copy = dict(candidate_eval)
        if source_chunk and not candidate_copy.get("source_chunk"):
            candidate_copy["source_chunk"] = source_chunk

        query_score = (
            query_unique,
            query_tokens,
            -candidate_copy.get("length", 0),
            candidate_copy.get("coverage_ratio", 0.0),
            candidate_copy.get("token_overlap", 0),
        )

        if best_query_eval is None or query_score > best_query_eval[0]:
            best_query_eval = (query_score, candidate_copy)

    for chunk in context_chunks:
        text = chunk.get("text")
        if not text:
            continue
        chunk_text = str(text).strip()
        if not chunk_text:
            continue

        if chunk_text not in seen_candidates:
            seen_candidates.add(chunk_text)
            chunk_eval = _evaluate_candidate(chunk_text)
            if chunk_eval:
                max_token_overlap = max(
                    max_token_overlap, chunk_eval.get("token_overlap", 0)
                )
                _maybe_update_label_candidate(chunk_eval)
                _consider_query_candidate(chunk_eval, chunk_text)
                if chunk_eval["length"] <= reference_length:
                    if tuple(chunk_eval["score"]) > best_score:
                        best_candidate = chunk_eval["text"]
                        best_score = tuple(chunk_eval["score"])
                        best_eval = chunk_eval
                else:
                    if fallback_eval is None:
                        fallback_eval = chunk_eval
                    elif tuple(chunk_eval["score"]) > tuple(fallback_eval["score"]):
                        fallback_eval = chunk_eval
            elif query_tokens:
                query_only_eval = _evaluate_query_only_candidate(chunk_text)
                if query_only_eval:
                    _consider_query_candidate(query_only_eval, chunk_text)

        for fragment in _FRAGMENT_SPLIT_PATTERN.split(chunk_text):
            fragment_text = fragment.strip()
            if not fragment_text:
                continue
            candidate_texts = [fragment_text]
            label_segments = _split_label_segments(fragment_text)
            for segment in label_segments:
                if segment not in candidate_texts:
                    candidate_texts.append(segment)

            for candidate_text in candidate_texts:
                candidate_stripped = candidate_text.strip()
                if not candidate_stripped:
                    continue
                if candidate_stripped in seen_candidates:
                    continue
                seen_candidates.add(candidate_stripped)

                fragment_eval = _evaluate_candidate(candidate_stripped)
                if not fragment_eval:
                    if query_tokens:
                        query_only_eval = _evaluate_query_only_candidate(
                            candidate_stripped
                        )
                        if query_only_eval:
                            query_only_eval["source_chunk"] = chunk_text
                            _maybe_update_label_candidate(query_only_eval)
                            _consider_query_candidate(query_only_eval, chunk_text)
                    continue

                max_token_overlap = max(
                    max_token_overlap, fragment_eval.get("token_overlap", 0)
                )
                fragment_eval["source_chunk"] = chunk_text
                _maybe_update_label_candidate(fragment_eval)
                _consider_query_candidate(fragment_eval, chunk_text)

                if fragment_eval["length"] > reference_length:
                    if fallback_fragment_eval is None:
                        fallback_fragment_eval = fragment_eval
                    elif tuple(fragment_eval["score"]) > tuple(
                        fallback_fragment_eval["score"]
                    ):
                        fallback_fragment_eval = fragment_eval
                    continue

                if tuple(fragment_eval["score"]) > best_score:
                    best_candidate = fragment_eval["text"]
                    best_score = tuple(fragment_eval["score"])
                    best_eval = fragment_eval
    if (
        best_candidate == stripped_answer
        and best_label_query_candidate
        and best_label_query_candidate[1]
    ):
        label_eval = best_label_query_candidate[1]
        label_text = label_eval.get("text")
        label_length = label_eval.get("length", 0)
        if label_text:
            trimmed_segments = _split_secondary_requirements(label_text)
            if trimmed_segments:
                primary_segment = trimmed_segments[0]
                if primary_segment and primary_segment != label_text:
                    trimmed_eval = _evaluate_candidate(primary_segment)
                    if trimmed_eval:
                        trimmed_eval["source_chunk"] = label_eval.get("source_chunk")
                        label_eval = trimmed_eval
                        label_text = trimmed_eval.get("text", label_text)
                        label_length = trimmed_eval.get("length", label_length)
            if reference_length <= 0 or (
                label_length and label_length <= reference_length
            ):
                best_candidate = label_text
                best_eval = label_eval
                label_score = label_eval.get("score")
                if label_score:
                    best_score = tuple(label_score)

    if best_candidate != stripped_answer:
        if best_eval and best_eval.get("source_chunk"):
            return _prepare_aligned_text(_restore_fragment_text(best_eval))

        return _prepare_aligned_text(best_candidate)

    allow_query_replacement = (
        fallback_answer or total_token_count == 0 or max_token_overlap == 0
    )

    if best_query_eval and (allow_query_replacement or total_token_count >= 6):
        _, query_eval = best_query_eval
        coverage = query_eval.get("coverage_ratio", 0.0)
        token_overlap = query_eval.get("token_overlap", 0)
        length = query_eval.get("length", 0)
        if total_token_count:
            min_token_overlap = max(4, int(total_token_count * 0.4))
        else:
            min_token_overlap = 1
        length_limit = max(
            reference_length + 60,
            int(reference_length * 1.5) if reference_length > 0 else 0,
        )
        if reference_length <= 0:
            length_limit = length or 0

        query_tokens_matched = query_eval.get("query_token_overlap", 0)
        query_unique = query_eval.get("query_unique_overlap", 0)
        query_coverage = query_eval.get("query_coverage_ratio", 0.0)

        meets_threshold = coverage >= 0.5 or token_overlap >= min_token_overlap

        if allow_query_replacement and not meets_threshold:
            if fallback_answer and not (query_unique or query_tokens_matched):
                meets_threshold = True
            else:
                if total_query_token_count:
                    query_token_threshold = max(1, int(total_query_token_count * 0.3))
                else:
                    query_token_threshold = 1
                if total_query_token_count >= 4:
                    query_token_threshold = max(query_token_threshold, 2)

                meets_threshold = (
                    query_unique >= 2
                    or (
                        query_unique >= 1
                        and query_tokens_matched >= query_token_threshold
                    )
                    or query_coverage >= 0.5
                )

        if meets_threshold:
            if (
                allow_query_replacement
                or not length
                or reference_length <= 0
                or length <= length_limit
            ):
                selected_text = query_eval.get("text")
                if selected_text:
                    if query_eval.get("source_chunk"):
                        return _prepare_aligned_text(_restore_fragment_text(query_eval))
                    return _prepare_aligned_text(selected_text)

    def _should_expand(candidate_eval: dict) -> bool:
        total_candidate_tokens = candidate_eval.get("total_candidate_tokens", 0)
        candidate_fraction = candidate_eval.get("candidate_fraction", 0.0)
        additional_unique = candidate_eval.get("additional_unique_tokens", 0)
        additional_tokens = candidate_eval.get("additional_token_count", 0)
        candidate_length = candidate_eval.get("length", 0)
        candidate_text = candidate_eval.get("text", "")

        length_limit = max(reference_length * 2, reference_length + 120)
        if reference_length <= 0:
            length_limit = candidate_length

        if not candidate_text or not reference_variants:
            return False

        lowered_candidate = candidate_text.lower()
        short_answer = total_token_count < 4

        for variant in reference_variants:
            if not variant:
                continue
            lowered_variant = variant.lower()
            idx = lowered_candidate.find(lowered_variant)
            if idx == -1:
                continue
            prefix = candidate_text[:idx].strip()
            suffix = candidate_text[idx + len(variant) :].strip()
            if not prefix and not suffix:
                continue
            if candidate_length > length_limit:
                continue

            prefix_tokens_counter = (
                _chunk_keyword_tokens(prefix) if prefix else Counter()
            )
            prefix_tokens = sum(prefix_tokens_counter.values()) if prefix else 0

            suffix_tokens_counter = (
                _chunk_keyword_tokens(suffix) if suffix else Counter()
            )
            suffix_token_count = sum(suffix_tokens_counter.values()) if suffix else 0
            suffix_length = len(suffix)
            suffix_lower = suffix.lower()
            suffix_has_marker = any(
                marker in suffix_lower for marker in qualifying_suffix_markers
            )
            suffix_shares_answer = bool(answer_token_set & set(suffix_tokens_counter))
            suffix_allowed = False
            if not suffix:
                suffix_allowed = True
            elif suffix_token_count == 0 and suffix_length <= 5:
                suffix_allowed = True
            elif (
                suffix_token_count <= 12
                and suffix_length <= 80
                and (
                    suffix_has_marker
                    or (suffix_shares_answer and suffix_token_count <= 4)
                )
            ):
                suffix_allowed = True
            if not suffix_allowed:
                continue

            if not prefix:
                if suffix:
                    return True
                continue

            base_condition = (
                total_token_count >= 4
                and additional_unique >= 2
                and additional_tokens >= 3
                and candidate_fraction <= 0.7
                and candidate_length <= length_limit
            )

            prefix_valid = (
                _is_valid_prefix(prefix)
                and prefix_tokens <= 25
                and len(prefix) <= length_limit
            )
            if not prefix_valid:
                continue

            allow_short_prefix = (
                short_answer and prefix_tokens <= 12 and additional_unique >= 1
            )

            if base_condition or prefix_tokens <= 6 or allow_short_prefix:
                return True

        return False

    def _find_clause_start(text: str, index: int) -> int:
        start = index
        while start > 0:
            ch = text[start - 1]
            if ch in ".!?;\n":
                break
            start -= 1

        length = len(text)
        while start < length and text[start].isspace():
            start += 1

        prefix_segment = text[start:index]
        match = re.match(r"(?:\(\d+\)|\d+[\.)])\s+", prefix_segment)
        if match:
            start += match.end()

        return start

    def _find_clause_end(text: str, index: int) -> int:
        tail = text[index:]
        if not tail:
            return len(text)

        candidates: List[int] = []

        list_boundary = re.search(r",\s*(?:\d+[\.)])", tail)
        if list_boundary:
            candidates.append(index + list_boundary.start())

        label_boundary = re.search(r"\s+[A-Z][A-Z0-9 _/\-]{1,30}:", tail)
        if label_boundary:
            candidates.append(index + label_boundary.start())

        for punct in (".", ";", "!", "?"):
            pos = tail.find(punct)
            while pos != -1:
                # Skip decimal values like 1.5
                if punct == "." and pos + 1 < len(tail) and tail[pos + 1].isdigit():
                    pos = tail.find(punct, pos + 1)
                    continue
                candidates.append(index + pos + 1)
                break

        newline_pos = tail.find("\n")
        if newline_pos != -1:
            candidates.append(index + newline_pos)

        if not candidates:
            end = len(text)
        else:
            end = min(candidates)

        while end > index and text[end - 1].isspace():
            end -= 1

        return end

    def _expand_reference_from_chunk(chunk_text: str) -> Optional[str]:
        if not chunk_text or not reference_variants:
            return None

        lowered_chunk = chunk_text.lower()
        for variant in reference_variants:
            if not variant:
                continue
            lowered_variant = variant.lower()
            match_index = lowered_chunk.find(lowered_variant)
            if match_index == -1:
                continue
            start = _find_clause_start(chunk_text, match_index)
            end = _find_clause_end(chunk_text, match_index + len(variant))
            if end <= start:
                continue
            candidate = chunk_text[start:end].strip()
            if not candidate:
                continue

            candidate_lower = candidate.lower()
            variant_index = candidate_lower.find(lowered_variant)
            if (
                variant_index == -1
                and lowered_variant.rstrip(" .;:,") != lowered_variant
            ):
                trimmed_variant = lowered_variant.rstrip(" .;:,")
                variant_index = candidate_lower.find(trimmed_variant)
                if variant_index != -1:
                    variant_length = len(trimmed_variant)
                else:
                    variant_length = len(lowered_variant)
            else:
                variant_length = len(lowered_variant)

            if variant_index == -1:
                variant_index = 0
                variant_length = len(candidate)

            prefix_segment = candidate[:variant_index].strip()
            suffix_segment = candidate[variant_index + variant_length :].strip()

            if prefix_segment and not _is_valid_prefix(prefix_segment):
                continue
            if suffix_segment:
                suffix_tokens_counter = _chunk_keyword_tokens(suffix_segment)
                suffix_token_count = sum(suffix_tokens_counter.values())
                suffix_length = len(suffix_segment)
                suffix_lower = suffix_segment.lower()
                suffix_has_marker = any(
                    marker in suffix_lower for marker in qualifying_suffix_markers
                )
                suffix_shares_answer = bool(
                    answer_token_set & set(suffix_tokens_counter)
                )
                suffix_allowed = False
                if suffix_token_count == 0 and suffix_length <= 5:
                    suffix_allowed = True
                elif (
                    suffix_token_count <= 12
                    and suffix_length <= 80
                    and (
                        suffix_has_marker
                        or (suffix_shares_answer and suffix_token_count <= 4)
                    )
                ):
                    suffix_allowed = True
                if not suffix_allowed:
                    continue

            return candidate
        return None

    if fallback_fragment_eval and _should_expand(fallback_fragment_eval):
        restored = _restore_fragment_text(fallback_fragment_eval)
        return _prepare_aligned_text(restored)

    if fallback_eval and _should_expand(fallback_eval):
        return _prepare_aligned_text(fallback_eval["text"])

    expanded_candidates: List[dict] = []

    for chunk in context_chunks:
        text = chunk.get("text")
        if not text:
            continue
        chunk_text = str(text).strip()
        if not chunk_text:
            continue
        expanded = _expand_reference_from_chunk(chunk_text)
        if not expanded:
            continue
        prepared = _prepare_aligned_text(expanded)
        if not prepared:
            continue
        if prepared.strip().lower() == stripped_answer.lower():
            continue
        candidate_eval = _evaluate_candidate(prepared)
        if not candidate_eval:
            candidate_eval = {
                "text": prepared,
                "score": (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -abs(len(prepared) - reference_length),
                    -len(prepared),
                ),
            }
        else:
            candidate_eval["text"] = prepared
        expanded_candidates.append(candidate_eval)

    if expanded_candidates:
        best_expanded = max(expanded_candidates, key=lambda item: tuple(item["score"]))
        return best_expanded["text"]

    return stripped_answer


def _chunk_keyword_tokens(text: str) -> Counter[str]:
    if not text:
        return Counter()
    tokens = _KEYWORD_PATTERN.findall(text.lower())
    if not tokens:
        return Counter()
    return Counter(tokens)


def _select_citations(
    answer: Any,
    context_chunks: List[dict],
    *,
    max_default: int = 3,
) -> List[dict]:
    if not context_chunks:
        return []

    cleaned_answer = clean_answer_text(answer)
    if not cleaned_answer:
        return context_chunks[:max_default]

    normalized_answer = _normalize_for_match(cleaned_answer)
    fragments = _extract_answer_fragments(cleaned_answer)

    answer_tokens = _chunk_keyword_tokens(cleaned_answer)
    answer_token_set = set(answer_tokens)

    matches = []
    for index, chunk in enumerate(context_chunks):
        text = chunk.get("text") or ""
        normalized_chunk = _normalize_for_match(text)
        chunk_tokens = _chunk_keyword_tokens(text)
        token_overlap = sum(
            min(count, chunk_tokens.get(token, 0))
            for token, count in answer_tokens.items()
        )
        tokens_in_answer = set(chunk_tokens) & answer_token_set
        fragment_matches = {
            pos
            for pos, fragment in enumerate(fragments)
            if fragment and fragment in normalized_chunk
        }
        span_match = bool(normalized_answer) and normalized_answer in normalized_chunk
        matches.append(
            {
                "index": index,
                "span_match": span_match,
                "fragment_matches": fragment_matches,
                "tokens": tokens_in_answer,
                "token_overlap": token_overlap,
                "section": chunk.get("section"),
            }
        )

    matches.sort(
        key=lambda item: (
            -int(item["span_match"]),
            -len(item["fragment_matches"]),
            -item["token_overlap"],
            item["index"],
        )
    )

    selected_indices: List[int] = []
    covered_tokens: set[str] = set()
    covered_fragments: set[int] = set()
    covered_sections: set[str] = set()
    span_covered = False

    for match in matches:
        section = match["section"]
        tokens = match["tokens"]
        new_tokens = bool(answer_token_set and tokens - covered_tokens)
        new_fragments = bool(match["fragment_matches"] - covered_fragments)
        new_span = match["span_match"] and not span_covered
        contributes_section = bool(
            section
            and section not in covered_sections
            and (
                match["span_match"]
                or match["fragment_matches"]
                or match["token_overlap"]
            )
        )

        need_chunk = False
        if len(selected_indices) < max_default:
            need_chunk = True
        elif new_span or new_fragments or new_tokens or contributes_section:
            need_chunk = True

        if not need_chunk:
            continue

        selected_indices.append(match["index"])
        covered_fragments.update(match["fragment_matches"])
        covered_tokens.update(tokens)
        if section:
            covered_sections.add(section)
        if match["span_match"]:
            span_covered = True

    return [context_chunks[idx] for idx in selected_indices]


def _load_error_types(module_name: str, *class_names: str) -> Tuple[type, ...]:
    try:
        module = importlib.import_module(module_name)
    except Exception:  # pragma: no cover - import errors fall back to defaults
        return tuple()

    error_types: List[type] = []
    for name in class_names:
        candidate = getattr(module, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            error_types.append(candidate)
    return tuple(error_types)


def _get_openai_error_types() -> Tuple[type, ...]:
    return _load_error_types(
        "openai",
        "BadRequestError",
        "RateLimitError",
        "APIError",
        "APIStatusError",
        "OpenAIError",
    )


def _get_gemini_error_types() -> Tuple[type, ...]:
    provider_errors = list(
        _load_error_types(
            "google.api_core.exceptions",
            "GoogleAPIError",
            "InvalidArgument",
            "ResourceExhausted",
            "TooManyRequests",
        )
    )

    try:
        genai_errors = importlib.import_module("google.genai.errors")
    except Exception:  # pragma: no cover - optional dependency
        return tuple(provider_errors)

    client_error = getattr(genai_errors, "ClientError", None)
    if isinstance(client_error, type) and issubclass(client_error, BaseException):
        client_error_types: List[type] = [client_error]
        for name in dir(genai_errors):
            candidate = getattr(genai_errors, name)
            if (
                isinstance(candidate, type)
                and issubclass(candidate, client_error)
                and candidate not in client_error_types
            ):
                client_error_types.append(candidate)

        for error_type in client_error_types:
            if error_type not in provider_errors:
                provider_errors.append(error_type)

    return tuple(provider_errors)


def _is_provider_error(exc: Exception, candidates: Tuple[type, ...]) -> bool:
    return any(isinstance(exc, candidate) for candidate in candidates)


def _provider_unavailable_answer(num_chunks: int) -> str:
    return (
        "[FALLBACK] Unable to reach the language model. "
        f"Based on {num_chunks} retrieved passages, see citations."
    )


def _demo_answer(num_chunks: int) -> str:
    return f"[DEMO] Based on {num_chunks} retrieved passages, see citations."


def call_llm_with_citations(query: str, chunks: List[dict]) -> Tuple[str, List[dict]]:
    """Call the configured LLM asking a question grounded in ``chunks``.

    The function supports both OpenAI's chat completions API and Google's
    Gemini SDK.  When no provider/API key is configured a simple fallback answer
    is generated so tests can run without external dependencies.
    """

    max_chars = getattr(settings, "llm_context_char_budget", None)
    if not isinstance(max_chars, int) or max_chars <= 0:
        max_chars = DEFAULT_CONTEXT_CHAR_BUDGET

    bounded_chunks, context = _select_chunks_for_context(chunks, max_chars=max_chars)
    provider_fallback = _provider_unavailable_answer(len(bounded_chunks))
    demo_answer = _demo_answer(len(bounded_chunks))
    system_prompt = _get_qa_system_prompt()

    if settings.llm_provider == "openai" and settings.llm_api_key:
        provider_errors = _get_openai_error_types()
        max_attempts = 3
        base_delay = 1.0
        for attempt in range(1, max_attempts + 1):
            try:
                from openai import OpenAI

                client = OpenAI(api_key=settings.llm_api_key)
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}",
                    },
                ]
                response = client.chat.completions.create(
                    model=settings.llm_model or "gpt-3.5-turbo",
                    messages=messages,
                )
                answer = response.choices[0].message.content.strip()
            except Exception as exc:  # pragma: no cover - network failures hard to test
                is_provider_error = not provider_errors or _is_provider_error(
                    exc, provider_errors
                )
                if is_provider_error:
                    logger.exception(
                        "OpenAI call failed on attempt %s/%s",
                        attempt,
                        max_attempts,
                        exc_info=True,
                    )
                    if attempt == max_attempts:
                        logger.error(
                            "OpenAI provider unavailable after %s attempts; returning fallback response",
                            max_attempts,
                        )
                        return provider_fallback, _select_citations(
                            provider_fallback, bounded_chunks
                        )
                    sleep_seconds = base_delay * (2 ** (attempt - 1))
                    time.sleep(sleep_seconds)
                    continue
                logger.exception("LLM call failed", exc_info=True)
                raise HTTPException(
                    status_code=502, detail="LLM provider call failed"
                ) from exc
            else:
                return answer, _select_citations(answer, bounded_chunks)

    if settings.llm_provider == "gemini" and settings.llm_api_key:
        provider_errors = _get_gemini_error_types()
        max_attempts = 3
        base_delay = 1.0
        for attempt in range(1, max_attempts + 1):
            try:
                from google import genai

                client = genai.Client(api_key=settings.llm_api_key)
                instruction_text = system_prompt
                user_content = {
                    "role": "user",
                    "parts": [
                        {
                            "text": f"Context:\n{context}\n\nQuestion: {query}",
                        }
                    ],
                }
                response = client.models.generate_content(
                    model=settings.llm_model or "gemini-1.5-flash",
                    contents=[instruction_text, user_content],
                )
                answer = _extract_gemini_answer(response)
            except Exception as exc:  # pragma: no cover - network failures hard to test
                is_provider_error = not provider_errors or _is_provider_error(
                    exc, provider_errors
                )
                if is_provider_error:
                    logger.exception(
                        "Gemini call failed on attempt %s/%s",
                        attempt,
                        max_attempts,
                        exc_info=True,
                    )
                    if attempt == max_attempts:
                        logger.error(
                            "Gemini provider unavailable after %s attempts; returning fallback response",
                            max_attempts,
                        )
                        return provider_fallback, _select_citations(
                            provider_fallback, bounded_chunks
                        )
                    sleep_seconds = base_delay * (2 ** (attempt - 1))
                    time.sleep(sleep_seconds)
                    continue
                logger.exception("LLM call failed", exc_info=True)
                raise HTTPException(
                    status_code=502, detail="LLM provider call failed"
                ) from exc
            else:
                return answer, _select_citations(answer, bounded_chunks)

        # Fallback when no LLM provider is configured
    answer = demo_answer
    return answer, _select_citations(answer, bounded_chunks)


def _extract_gemini_answer(response: Any) -> str:
    """Return the primary answer text from a Gemini SDK response object."""

    def _iter_parts(parts: Any) -> Iterable[Any]:
        if parts is None:
            return []
        if isinstance(parts, dict):
            return parts.get("parts", [])
        if hasattr(parts, "parts"):
            return parts.parts
        if isinstance(parts, Iterable) and not isinstance(parts, (str, bytes)):
            return parts
        return []

    candidates = getattr(response, "candidates", None)
    if candidates is None and isinstance(response, dict):
        candidates = response.get("candidates")

    answer_chunks: List[str] = []
    for candidate in candidates or []:
        content = getattr(candidate, "content", None)
        if content is None and isinstance(candidate, dict):
            content = candidate.get("content")
        parts = _iter_parts(content)
        if not parts and isinstance(candidate, dict):
            parts = candidate.get("parts", [])
        for part in parts or []:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                answer_chunks.append(str(text).strip())

    if answer_chunks:
        return "\n".join(chunk for chunk in answer_chunks if chunk).strip()
    output_text = getattr(response, "output_text", None)
    if output_text is None and isinstance(response, dict):
        output_text = response.get("output_text")
    if output_text:
        return str(output_text).strip()
    fallback_text = getattr(response, "text", None)
    if fallback_text is None and isinstance(response, dict):
        fallback_text = response.get("text")
    return (fallback_text or "").strip()


def check_eligibility(criteria: dict, patient) -> dict:
    """Evaluate simple structured eligibility rules against ``patient``.

    The current implementation looks for age ranges (``"Age X to Y"``) and
    single-sex requirements (``"male only"`` / ``"female only"``) within the
    inclusion/exclusion criteria lists.  When a patient value is missing or a
    rule is violated we flag the criterion as a failure and mark the patient as
    ineligible.
    """

    patient_data = _patient_to_dict(patient)
    age = _parse_age_value(patient_data.get("age"))
    sex_value = patient_data.get("sex")
    sex_display = str(sex_value).strip() if sex_value is not None else "unspecified"
    sex = _normalize_sex(sex_value)
    sex_label = sex_display or "unspecified"

    rules = _extract_rules(criteria or {})

    eligible = True
    reasons: List[str] = []

    for rule in rules["age"]:
        lower = rule.get("min")
        upper = rule.get("max")
        text = rule["text"]
        source = rule["source"]

        if age is None:
            eligible = False
            reasons.append(f"Missing age information for {source} criterion ({text})")
            continue

        if lower is not None and upper is not None and lower > upper:
            lower, upper = upper, lower

        if source == "inclusion":
            if lower is not None and upper is not None:
                if not (lower <= age <= upper):
                    eligible = False
                    reasons.append(
                        f"Age {age} outside required range for inclusion criterion ({text})"
                    )
            else:
                if lower is not None and age < lower:
                    eligible = False
                    reasons.append(
                        f"Age {age} below minimum for inclusion criterion ({text})"
                    )
                if upper is not None and age > upper:
                    eligible = False
                    reasons.append(
                        f"Age {age} above maximum for inclusion criterion ({text})"
                    )
        else:  # exclusion
            match = False
            if lower is not None and upper is not None:
                match = lower <= age <= upper
            elif lower is not None:
                match = age >= lower
            elif upper is not None:
                match = age <= upper
            if match:
                eligible = False
                reasons.append(f"Age {age} triggers exclusion criterion ({text})")

    for rule in rules["sex"]:
        allowed = rule["allowed"]
        text = rule["text"]
        source = rule["source"]
        allowed_label = ", ".join(sorted(allowed))

        if sex is None:
            eligible = False
            reasons.append(
                f"Missing or unsupported sex value for {source} criterion ({text})"
            )
            continue

        if source == "inclusion":
            if sex not in allowed:
                eligible = False
                reasons.append(
                    (
                        f"Sex {sex_label} not permitted by inclusion criterion ({text}); "
                        f"allowed: {allowed_label}"
                    )
                )
        else:  # exclusion
            if sex in allowed:
                eligible = False
                reasons.append(f"Sex {sex_label} triggers exclusion criterion ({text})")

    return {"eligible": eligible, "reasons": reasons}


_AGE_RANGE_PATTERNS = [
    re.compile(
        r"between\s+(?P<min>\d{1,3})\s*(?:years?|yrs?)?\s*(?:and|to|through|up\s*to|upto|-)\s*(?P<max>\d{1,3})\s*(?:years?|yrs?)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:age[s]?|aged|subjects?|participants?)\s*(?:is|are|must\s+be|should\s+be|:|=)?\s*(?P<min>\d{1,3})\s*(?:years?|yrs?)?\s*(?:to|through|and|up\s*to|upto|-)\s*(?P<max>\d{1,3})\s*(?:years?|yrs?)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<min>\d{1,3})\s*(?:to|through|-)\s*(?P<max>\d{1,3})\s*(?:years?|yrs?)?\s*(?:of\s+age|years?\s+of\s+age|old)",
        re.IGNORECASE,
    ),
]

_AGE_MIN_PATTERNS = [
    re.compile(
        r"(?:>=|>\s*=|>\s*or\s*equal\s*to|greater\s+than\s+or\s+equal\s+to|at\s+least|no\s+less\s+than|minimum(?:\s+age)?(?:\s+of)?|not\s+younger\s+than)\s*(?P<value>\d{1,3})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<value>\d{1,3})\s*(?:\+|\s*or\s+older|\s*and\s+older)\s*(?:years?|yrs?)?(?:\s*of\s+age)?",
        re.IGNORECASE,
    ),
]

_AGE_MAX_PATTERNS = [
    re.compile(
        r"(?:<=|<\s*=|<\s*or\s*equal\s*to|less\s+than\s+or\s+equal\s+to|at\s+most|no\s+more\s+than|no\s+older\s+than|not\s+older\s+than|max(?:imum)?(?:\s+age)?(?:\s+of)?)\s*(?P<value>\d{1,3})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<value>\d{1,3})\s*(?:\s*or\s+younger|\s*and\s+younger)\s*(?:years?|yrs?)?(?:\s*of\s+age)?",
        re.IGNORECASE,
    ),
]
_SEX_SYNONYMS: Dict[str, str] = {
    "male": "male",
    "males": "male",
    "man": "male",
    "men": "male",
    "female": "female",
    "females": "female",
    "woman": "female",
    "women": "female",
}


def _patient_to_dict(patient: Any) -> Dict[str, Any]:
    if patient is None:
        return {}
    if hasattr(patient, "model_dump") and callable(patient.model_dump):
        try:
            data = patient.model_dump()
        except TypeError:  # pragma: no cover - defensive, should not happen in tests
            data = {}
        return data or {}
    if isinstance(patient, dict):
        return patient
    return {}


def _parse_age_value(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group())
    return None


def _normalize_sex(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return _SEX_SYNONYMS.get(text)


def _normalize_age_phrase(text: str) -> str:
    normalized = text.replace("\u00a0", " ")
    replacements = {
        "≥": ">=",
        "⩾": ">=",
        "≤": "<=",
        "⩽": "<=",
        "–": "-",
        "—": "-",
        "−": "-",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = normalized.lower()
    normalized = normalized.replace("upto", "up to")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _mentions_age(text: str) -> bool:
    return bool(re.search(r"\b(age|aged|ages|year|years|yrs|y/o|yo|old)\b", text))


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_age_rule(text: str) -> Dict[str, int | None] | None:
    normalized = _normalize_age_phrase(text)
    if not normalized or not _mentions_age(normalized):
        return None

    for pattern in _AGE_RANGE_PATTERNS:
        match = pattern.search(normalized)
        if match:
            min_value = _safe_int(match.group("min"))
            max_value = _safe_int(match.group("max"))
            if min_value is None and max_value is None:
                continue
            return {"min": min_value, "max": max_value}

    min_value: int | None = None
    max_value: int | None = None

    for pattern in _AGE_MIN_PATTERNS:
        match = pattern.search(normalized)
        if match:
            min_value = _safe_int(match.group("value"))
            if min_value is not None:
                break

    for pattern in _AGE_MAX_PATTERNS:
        match = pattern.search(normalized)
        if match:
            max_value = _safe_int(match.group("value"))
            if max_value is not None:
                break

    if min_value is None and max_value is None:
        return None

    return {"min": min_value, "max": max_value}


def _extract_rules(criteria: Dict[str, Any]) -> Dict[str, List[dict]]:
    rules = {"age": [], "sex": []}
    if not isinstance(criteria, dict):
        return rules

    for section in ("inclusion", "exclusion"):
        items = criteria.get(section, [])
        if items is None:
            continue
        if isinstance(items, (str, bytes)):
            items = [items]

        for raw in items:
            if not isinstance(raw, str):
                continue
            text = raw.strip()
            if not text:
                continue

            age_rule = _parse_age_rule(text)
            if age_rule:
                rules["age"].append(
                    {
                        "min": age_rule.get("min"),
                        "max": age_rule.get("max"),
                        "text": text,
                        "source": section,
                    }
                )

            sex_rule = _parse_sex_rule(text)
            if sex_rule:
                sex_rule.update({"text": text, "source": section})
                rules["sex"].append(sex_rule)

    return rules


def _parse_sex_rule(text: str) -> Dict[str, set[str]] | None:
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return None

    sexes_in_text = {
        canonical
        for token in tokens
        if (canonical := _SEX_SYNONYMS.get(token)) is not None
    }
    if len(sexes_in_text) != 1:
        return None

    target = next(iter(sexes_in_text))
    only_indices = [i for i, token in enumerate(tokens) if token == "only"]
    if not only_indices:
        return None

    sex_indices = [
        idx for idx, token in enumerate(tokens) if _SEX_SYNONYMS.get(token) == target
    ]

    if not any(abs(si - oi) <= 3 for si in sex_indices for oi in only_indices):
        return None

    return {"allowed": {target}}
