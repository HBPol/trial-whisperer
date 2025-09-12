from fastapi import APIRouter, HTTPException

from ..agents.tools import check_eligibility
from ..models.schemas import EligibilityRequest, EligibilityResponse
from ..retrieval.search_client import retrieve_criteria_for_trial

router = APIRouter()


@router.post("/", response_model=EligibilityResponse)
async def check(payload: EligibilityRequest):
    criteria = retrieve_criteria_for_trial(payload.nct_id)
    if not criteria:
        raise HTTPException(400, detail="Criteria not found for trial")
    result = check_eligibility(criteria, payload.patient)
    return result
