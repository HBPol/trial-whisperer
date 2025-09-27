from fastapi import APIRouter, HTTPException

from ..models.schemas import TrialMetadata
from ..retrieval.trial_store import get_trial_metadata

router = APIRouter()


@router.get("/{nct_id}", response_model=TrialMetadata)
async def get_trial(nct_id: str) -> TrialMetadata:
    trial = get_trial_metadata(nct_id)
    if trial is None:
        raise HTTPException(status_code=400, detail="Trial not found")
    return trial
