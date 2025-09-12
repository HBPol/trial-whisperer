from fastapi import APIRouter, HTTPException

from ..agents.tools import call_llm_with_citations
from ..models.schemas import AskRequest, AskResponse, Citation
from ..retrieval.search_client import retrieve_chunks



router = APIRouter()


@router.post("/", response_model=AskResponse)
async def ask(payload: AskRequest):
    if not payload.query:
        raise HTTPException(400, detail="Query cannot be empty")
    chunks = retrieve_chunks(query=payload.query, nct_id=payload.nct_id)
    if not chunks:
        raise HTTPException(404, detail="No relevant passages found.")
    answer, cits = call_llm_with_citations(payload.query, chunks)
    return AskResponse(
        answer=answer,
        citations=[
            Citation(nct_id=c["nct_id"], section=c["section"], text_snippet=c["text"])
            for c in cits
        ],
    )
