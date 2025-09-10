from fastapi import APIRouter, HTTPException

router = APIRouter()

# Placeholder store for demo; replace with proper datastore
TRIALS = {}

@router.get("/{nct_id}")
async def get_trial(nct_id: str):
    trial = TRIALS.get(nct_id)
    if not trial:
        raise HTTPException(404, detail="Trial not found")
    return trial