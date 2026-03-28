"""
app/api/analyze.py

FastAPI router — POST /api/analyze
Triggers the Analyzer Agent for a stored dataset and returns the
structured AnalysisReport.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.dataset import UserDataset
from app.agents.datacleaning_agent import run_analyzer

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    dataset_id: str
    """UUID of the previously imported dataset to analyze."""


class AnalyzeResponse(BaseModel):
    dataset_id: str
    report: dict
    """Full AnalysisReport produced by the Analyzer Agent."""


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_dataset(
    body: AnalyzeRequest,
    db: AsyncSession = Depends(get_db),
) -> AnalyzeResponse:
    """
    Run the **Analyzer Agent** against a stored dataset.

    1. Fetches the raw data from the `user_datasets` table.
    2. Passes the records through the Analyzer Agent.
    3. Returns the structured `AnalysisReport`.
    """
    # ── 1. Fetch dataset from DB ──────────────────────────────────────────
    try:
        dataset_uuid = uuid.UUID(body.dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id format (must be UUID)")

    result = await db.execute(
        select(UserDataset).where(UserDataset.id == dataset_uuid)
    )
    dataset = result.scalar_one_or_none()

    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{body.dataset_id}' not found")

    # ── 2. Extract records ────────────────────────────────────────────────
    raw_data = dataset.raw_data  # stored as JSONB → list[dict]
    if not isinstance(raw_data, list) or not raw_data:
        raise HTTPException(status_code=422, detail="Dataset raw_data is empty or malformed")

    # ── 3. Run Analyzer Agent ─────────────────────────────────────────────
    try:
        report = await run_analyzer(
            dataset_id=body.dataset_id,
            records=raw_data,
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return AnalyzeResponse(dataset_id=body.dataset_id, report=report)
