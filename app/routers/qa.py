from fastapi import APIRouter, HTTPException

from ..agents.tools import call_llm_with_citations
from ..models.schemas import AskRequest, AskResponse, Citation
from ..retrieval.search_client import retrieve_chunks

router = APIRouter()


@router.post("/", response_model=AskResponse)
async def ask(body: AskRequest):
    if not body.query:
        raise HTTPException(status_code=400, detail="query is required")
    chunks = retrieve_chunks(query=body.query, nct_id=body.nct_id)
    if not chunks:
        raise HTTPException(404, detail="No relevant passages found.")
    answer, cits = call_llm_with_citations(body.query, chunks)
    return AskResponse(
        answer=answer,
        citations=[
            Citation(nct_id=c["nct_id"], section=c["section"], text_snippet=c["text"])
            for c in cits
        ],
    )
