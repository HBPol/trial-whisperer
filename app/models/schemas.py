from typing import List, Optional

from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str
    nct_id: Optional[str] = None


class Citation(BaseModel):
    nct_id: str
    section: str
    text_snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]


class PatientProfile(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    labs: Optional[dict] = None  # {"ECOG": 2, "LVEF": 55}


class EligibilityRequest(BaseModel):
    nct_id: str
    patient: PatientProfile


class EligibilityResponse(BaseModel):
    eligible: bool
    reasons: List[str]
