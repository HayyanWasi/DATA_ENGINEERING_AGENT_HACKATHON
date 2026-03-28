"""
app/api/clean.py

FastAPI router — POST /api/clean

Single entry-point for the full data-cleaning pipeline.
All orchestration is delegated to the Orchestrator Agent (run_pipeline).

  Orchestrator
    ├── Analyzer Agent   → analysis report + stats
    ├── Strategist Agent → numbered cleaning strategy
    └── Executor Agent   → cleaned CSV + execution log

Query parameters:
  skip_executor=true  → stop after strategy (no code execution)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from database import get_db
from models.dataset import UserDataset
from app.agents import run_pipeline          # ← single entry point

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class CleanRequest(BaseModel):
    dataset_id: str
    """UUID of the previously imported dataset."""
    target_column: str
    """Name of the ML target / prediction column."""


class StageResult(BaseModel):
    dataset_id: str | None = None
    stats: dict[str, Any] | None = None
    analysis_report: str | None = None
    cleaning_strategy: str | None = None


class CleanResponse(BaseModel):
    dataset_id: str
    target_column: str
    stages_completed: int
    stages_attempted: int
    elapsed_seconds: float = 0.0
    # ── child-agent outputs ───────────────────────────────────────────────────
    analysis_report: str | None = None
    stats: dict[str, Any] | None = None
    cleaning_strategy: str | None = None
    execution_log: str | None = None
    output_file: str | None = None
    # ── pipeline health ───────────────────────────────────────────────────────
    errors: list[str] = []
    pipeline_log: str = ""


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/api/clean", response_model=CleanResponse)
async def clean_dataset(
    body: CleanRequest,
    db: AsyncSession = Depends(get_db),
    skip_executor: bool = Query(
        default=False,
        description="Set true to stop after strategy (skip code execution).",
    ),
) -> CleanResponse:
    """
    Run the full **3-agent data-cleaning pipeline**.

    The Orchestrator Agent manages:
      1. Input validation
      2. Analyzer Agent   → analysis report
      3. Strategist Agent → cleaning strategy
      4. Executor Agent   → cleaned CSV  (skipped if `skip_executor=true`)

    Returns all stage outputs plus a pipeline log.
    """
    # ── 1. Fetch dataset from DB ──────────────────────────────────────────────
    try:
        dataset_uuid = uuid.UUID(body.dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id (must be UUID).")

    result = await db.execute(
        select(UserDataset).where(UserDataset.id == dataset_uuid)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{body.dataset_id}' not found.")

    raw_data: list[dict] = dataset.raw_data
    if not isinstance(raw_data, list) or not raw_data:
        raise HTTPException(status_code=422, detail="Dataset raw_data is empty or malformed.")

    # ── 2. Hand off to Orchestrator ───────────────────────────────────────────
    result_dict = await run_pipeline(
        dataset_id=body.dataset_id,
        records=raw_data,
        target_column=body.target_column,
        skip_executor=skip_executor,
    )

    # ── 3. Unpack orchestrator result ─────────────────────────────────────────
    analysis: dict | None = result_dict.get("analysis")
    strategy: dict | None = result_dict.get("strategy")
    execution: dict | None = result_dict.get("execution")

    # Surface hard errors (no stages completed) as HTTP 500
    if result_dict.get("stages_completed", 0) == 0 and result_dict.get("errors"):
        raise HTTPException(
            status_code=500,
            detail={"errors": result_dict["errors"], "log": result_dict.get("pipeline_log", "")},
        )

    return CleanResponse(
        dataset_id=body.dataset_id,
        target_column=body.target_column,
        stages_completed=result_dict.get("stages_completed", 0),
        stages_attempted=result_dict.get("stages_attempted", 0),
        elapsed_seconds=result_dict.get("elapsed_seconds", 0.0),
        # Analyzer outputs
        analysis_report=analysis.get("analysis_report") if analysis else None,
        stats=analysis.get("stats") if analysis else None,
        # Strategist output
        cleaning_strategy=strategy.get("cleaning_strategy") if strategy else None,
        # Executor outputs
        execution_log=execution.get("execution_log") if execution else None,
        output_file=execution.get("output_file") if execution else None,
        # Pipeline health
        errors=result_dict.get("errors", []),
        pipeline_log=result_dict.get("pipeline_log", ""),
    )
