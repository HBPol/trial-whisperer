"""Metadata endpoints for surface-level application information."""

from __future__ import annotations

from fastapi import APIRouter

from ..metadata import build_ingestion_summary
from ..models.schemas import IngestionSummary

router = APIRouter()


@router.get("/ingestion-summary", response_model=IngestionSummary)
async def read_ingestion_summary() -> IngestionSummary:
    """Return information about the currently indexed clinical trials."""

    summary = build_ingestion_summary()
    return IngestionSummary(**summary)
