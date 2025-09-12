from typing import List, Tuple

from app.deps import get_settings

settings = get_settings()

# Stub LLM caller — swap with Gemini/OpenAI SDK or plain HTTP


def call_llm_with_citations(query: str, chunks: List[dict]) -> Tuple[str, List[dict]]:
    # TODO: implement real prompt + call using settings.llm_api_key
    answer = f"[DEMO] Based on {len(chunks)} retrieved passages, see citations."
    citations = chunks[:3]
    return answer, citations


def check_eligibility(criteria: dict, patient) -> dict:
    # TODO: implement rules extraction + simple numeric checks
    # Demo outcome — always False with reasons placeholder
    return {
        "eligible": False,
        "reasons": ["Demo: rules engine not yet implemented"],
    }
