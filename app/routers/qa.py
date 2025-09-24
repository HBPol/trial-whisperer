import re
from typing import Optional

from fastapi import APIRouter, HTTPException

from ..agents.tools import (
    align_answer_to_context,
    call_llm_with_citations,
    clean_answer_text,
    refine_answer_with_context,
)
from ..models.schemas import AskRequest, AskResponse, Citation
from ..retrieval.search_client import retrieve_chunks

NCT_ID_PATTERN = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)


def _extract_nct_id_from_query(query: Optional[str]) -> Optional[str]:
    if not query:
        return None
    match = NCT_ID_PATTERN.search(query)
    if match:
        return match.group(0).upper()
    return None


router = APIRouter()


@router.post("/", response_model=AskResponse)
async def ask(body: AskRequest):
    # Ensure the request adheres to the AskRequest schema
    body = AskRequest.model_validate(body)

    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    nct_id = body.nct_id or _extract_nct_id_from_query(body.query)
    chunks = retrieve_chunks(query=body.query, nct_id=nct_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant passages found.")
    answer, cits = call_llm_with_citations(body.query, chunks)
    cleaned_answer = clean_answer_text(answer)
    alignment_context = cits or chunks
    final_answer = (
        align_answer_to_context(cleaned_answer, alignment_context, query=body.query)
        or cleaned_answer
    )
    final_answer = refine_answer_with_context(
        final_answer,
        alignment_context,
        query=body.query,
        original_answer=cleaned_answer,
    )
    citations = [
        Citation(nct_id=c["nct_id"], section=c["section"], text_snippet=c["text"])
        for c in cits
    ]
    return AskResponse(answer=final_answer, citations=citations)
