import re
from typing import Any, Dict, List, Tuple

from app.deps import get_settings

settings = get_settings()


def _format_context(chunks: List[dict]) -> str:
    """Create a numbered context block from retrieved chunks."""

    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        parts.append(
            f"({idx}) [Trial {chunk['nct_id']}] {chunk['section']}: {chunk['text']}"
        )
    return "\n".join(parts)


def call_llm_with_citations(query: str, chunks: List[dict]) -> Tuple[str, List[dict]]:
    """Call the configured LLM asking a question grounded in ``chunks``.

    The function supports both OpenAI's chat completions API and Google's
    Gemini SDK.  When no provider/API key is configured a simple fallback answer
    is generated so tests can run without external dependencies.
    """

    context = _format_context(chunks)
    citations = chunks[:3]

    if settings.llm_provider == "openai" and settings.llm_api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.llm_api_key)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You answer questions about clinical trials using the"
                        " provided context. Cite passages using (1), (2) etc."
                    ),
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
        except Exception:
            answer = (
                f"[LLM error] Based on {len(chunks)} retrieved passages, see citations."
            )
        return answer, citations

    if settings.llm_provider == "gemini" and settings.llm_api_key:
        try:
            import google.generativeai as genai

            genai.configure(api_key=settings.llm_api_key)
            model = genai.GenerativeModel(settings.llm_model or "gemini-1.5-flash")
            prompt = (
                "You answer questions about clinical trials using the provided"
                " context. Cite passages using (1), (2) etc.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}"
            )
            response = model.generate_content(prompt)
            answer = (response.text or "").strip()
        except Exception:
            answer = (
                f"[LLM error] Based on {len(chunks)} retrieved passages, see citations."
            )
        return answer, citations

        # Fallback when no LLM provider is configured
    answer = f"[DEMO] Based on {len(chunks)} retrieved passages, see citations."
    return answer, citations


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
        lower = rule["min"]
        upper = rule["max"]
        text = rule["text"]
        source = rule["source"]

        if age is None:
            eligible = False
            reasons.append(f"Missing age information for {source} criterion ({text})")
            continue

        if lower > upper:
            lower, upper = upper, lower

        if source == "inclusion":
            if not (lower <= age <= upper):
                eligible = False
                reasons.append(
                    f"Age {age} outside required range for inclusion criterion ({text})"
                )
        else:  # exclusion
            if lower <= age <= upper:
                eligible = False
                reasons.append(f"Age {age} falls within exclusion criterion ({text})")

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


_AGE_RANGE_PATTERN = re.compile(
    r"\bage[s]?\s*(?:is|between|from|:)?\s*(?P<min>\d{1,3})\s*(?:years?|yrs?)?\s*(?:to|-|–|—|through|and\s+up\s+to|and\s+under|and\s+over\s+to|up\s+to|upto)\s*(?P<max>\d{1,3})\s*(?:years?|yrs?)?",
    re.IGNORECASE,
)

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

            match = _AGE_RANGE_PATTERN.search(text)
            if match:
                try:
                    lower = int(match.group("min"))
                    upper = int(match.group("max"))
                except (TypeError, ValueError):
                    lower = upper = None
                if lower is not None and upper is not None:
                    rules["age"].append(
                        {"min": lower, "max": upper, "text": text, "source": section}
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
