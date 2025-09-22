import importlib
import logging
import re
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def _extract_answer_fragments(answer: str) -> List[str]:
    fragments: List[str] = []
    if not answer:
        return fragments
    for part in _FRAGMENT_SPLIT_PATTERN.split(answer):
        normalized = _normalize_for_match(part)
        if normalized:
            fragments.append(normalized)
    return fragments


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
            if not provider_errors or _is_provider_error(exc, provider_errors):
                logger.exception("OpenAI call failed", exc_info=True)
                return provider_fallback, _select_citations(
                    provider_fallback, bounded_chunks
                )
            logger.exception("LLM call failed", exc_info=True)
            raise HTTPException(
                status_code=502, detail="LLM provider call failed"
            ) from exc
        return answer, _select_citations(answer, bounded_chunks)

    if settings.llm_provider == "gemini" and settings.llm_api_key:
        provider_errors = _get_gemini_error_types()
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
            if not provider_errors or _is_provider_error(exc, provider_errors):
                logger.exception("Gemini call failed", exc_info=True)
                return provider_fallback, _select_citations(
                    provider_fallback, bounded_chunks
                )
            logger.exception("LLM call failed", exc_info=True)
            raise HTTPException(
                status_code=502, detail="LLM provider call failed"
            ) from exc
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
