"""
app/api/strategize.py

FastAPI router — POST /api/strategize

Two operational modes:
  A) Pass a dataset_id → pipeline runs Analyzer then Strategist automatically.
  B) Pass a pre-built analysis_report directly → Strategist only (faster for
     callers that already have the report from /api/analyze).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from database import get_db
from models.dataset import UserDataset
from app.agents.datacleaning_agent import run_analyzer, run_strategist

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class StrategizeRequest(BaseModel):
    """
    Provide EITHER dataset_id (auto-analyze first) OR analysis_report
    (skip the analysis step and go straight to strategy).
    """
    dataset_id: str | None = None
    analysis_report: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check_one_source(self) -> "StrategizeRequest":
        if not self.dataset_id and not self.analysis_report:
            raise ValueError("Provide either 'dataset_id' or 'analysis_report'.")
        return self


class StrategizeResponse(BaseModel):
    dataset_id: str
    analysis_report: dict[str, Any]
    cleaning_strategy: dict[str, Any]
    """
    Full CleaningStrategy produced by the Strategist Agent, including:
      - summary       : executive summary
      - cleaning_steps: ordered, priority-tagged list
      - null_strategy, duplicate_strategy, outlier_strategy, type_strategy
      - narrative     : numbered plain-English plan
    """


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/api/strategize", response_model=StrategizeResponse)
async def strategize_dataset(
    body: StrategizeRequest,
    db: AsyncSession = Depends(get_db),
) -> StrategizeResponse:
    """
    Generate a **CleaningStrategy** for a dataset.

    Flow A — dataset_id provided:
      1. Fetch raw records from DB
      2. Run Analyzer Agent  → AnalysisReport
      3. Run Strategist Agent → CleaningStrategy

    Flow B — analysis_report provided:
      1. Run Strategist Agent → CleaningStrategy  (Analyzer already done)
    """
    # ── Flow A: run full pipeline ─────────────────────────────────────────
    if body.dataset_id:
        try:
            dataset_uuid = uuid.UUID(body.dataset_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid dataset_id format (must be a UUID).",
            )

        result = await db.execute(
            select(UserDataset).where(UserDataset.id == dataset_uuid)
        )
        dataset = result.scalar_one_or_none()
        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{body.dataset_id}' not found.",
            )

        raw_data: list[dict] = dataset.raw_data
        if not isinstance(raw_data, list) or not raw_data:
            raise HTTPException(
                status_code=422,
                detail="Dataset raw_data is empty or malformed.",
            )

        # Step 1 — Analyze
        try:
            analysis_report = await run_analyzer(
                dataset_id=body.dataset_id,
                records=raw_data,
            )
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=f"Analyzer error: {exc}")

        dataset_id = body.dataset_id

    # ── Flow B: report already provided ──────────────────────────────────
    else:
        analysis_report = body.analysis_report  # type: ignore[assignment]
        dataset_id = analysis_report.get("dataset_id", "unknown")

    # ── Step 2 — Strategize ───────────────────────────────────────────────
    try:
        cleaning_strategy = await run_strategist(analysis_report=analysis_report)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Strategist error: {exc}")

    return StrategizeResponse(
        dataset_id=dataset_id,
        analysis_report=analysis_report,
        cleaning_strategy=cleaning_strategy,
    )
