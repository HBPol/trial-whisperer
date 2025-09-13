from typing import List, Tuple

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
    # TODO: implement rules extraction + simple numeric checks
    # Demo outcome — always False with reasons placeholder
    return {
        "eligible": False,
        "reasons": ["Demo: rules engine not yet implemented"],
    }
