from fastapi import APIRouter, HTTPException

from ..agents.tools import (
    align_answer_to_context,
    call_llm_with_citations,
    clean_answer_text,
)
from ..models.schemas import AskRequest, AskResponse, Citation
from ..retrieval.search_client import retrieve_chunks

router = APIRouter()


@router.post("/", response_model=AskResponse)
async def ask(body: AskRequest):
    # Ensure the request adheres to the AskRequest schema
    body = AskRequest.model_validate(body)

    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    chunks = retrieve_chunks(query=body.query, nct_id=body.nct_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant passages found.")
    answer, cits = call_llm_with_citations(body.query, chunks)
    cleaned_answer = clean_answer_text(answer)
    alignment_context = cits or chunks
    final_answer = (
        align_answer_to_context(cleaned_answer, alignment_context, query=body.query)
        or cleaned_answer
    )
    citations = [
        Citation(nct_id=c["nct_id"], section=c["section"], text_snippet=c["text"])
        for c in cits
    ]
    return AskResponse(answer=final_answer, citations=citations)
