"""
app/api/clean_direct.py

FastAPI router — POST /api/clean/direct

**No database required.** Send raw records directly in the request body and
get back the full 3-agent pipeline result immediately.

This is designed for:
  - Quick testing / prototyping
  - Demos without needing an imported dataset
  - Verifying the orchestrator works end-to-end

Request body example:
{
  "target_column": "salary",
  "skip_executor": false,
  "records": [
    {"age": 25, "salary": 50000, "department": "HR", "name": "Alice"},
    {"age": null, "salary": 999999, "department": "IT", "name": "Bob"},
    ...
  ]
}
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, model_validator

from app.agents import run_pipeline

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class DirectCleanRequest(BaseModel):
    target_column: str
    """Name of the ML target / prediction column."""

    records: list[dict[str, Any]]
    """Raw rows to clean — list of dicts, each dict is one row."""

    skip_executor: bool = False
    """Set true to stop after strategy (no code execution)."""

    @model_validator(mode="after")
    def _check_records(self) -> "DirectCleanRequest":
        if not self.records:
            raise ValueError("records list cannot be empty.")
        if len(self.records) > 50_000:
            raise ValueError("records list exceeds 50,000 rows limit for direct mode.")
        return self


class DirectCleanResponse(BaseModel):
    dataset_id: str
    target_column: str
    row_count: int
    column_count: int
    stages_completed: int
    stages_attempted: int
    elapsed_seconds: float = 0.0
    # Stage outputs
    analysis_report: str | None = None
    stats: dict[str, Any] | None = None
    cleaning_strategy: str | None = None
    execution_log: str | None = None
    output_file: str | None = None
    # Health
    errors: list[str] = []
    pipeline_log: str = ""


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/api/clean/direct", response_model=DirectCleanResponse)
async def clean_direct(body: DirectCleanRequest) -> DirectCleanResponse:
    """
    Run the full **3-agent data cleaning pipeline** on raw records sent
    directly in the request body — **no database import needed**.

    Pipeline:
      1. **Analyzer Agent**   → profiles the raw data
      2. **Strategist Agent** → produces a numbered cleaning plan
      3. **Executor Agent**   → implements the plan, saves outputs/cleaned_data.csv

    Tip: use `?skip_executor=true` (or set `skip_executor: true` in the body)
    to stop after the strategy without executing any code.
    """
    # Use a synthetic dataset_id so the orchestrator has something to tag logs with
    dataset_id = f"direct-{len(body.records)}rows"

    result = await run_pipeline(
        dataset_id=dataset_id,
        records=body.records,
        target_column=body.target_column,
        skip_executor=body.skip_executor,
    )

    # Surface hard failures as HTTP 500
    if result.get("stages_completed", 0) == 0 and result.get("errors"):
        raise HTTPException(
            status_code=500,
            detail={
                "errors": result["errors"],
                "log": result.get("pipeline_log", ""),
            },
        )

    analysis  = result.get("analysis")
    strategy  = result.get("strategy")
    execution = result.get("execution")

    return DirectCleanResponse(
        dataset_id=dataset_id,
        target_column=body.target_column,
        row_count=len(body.records),
        column_count=len(body.records[0]) if body.records else 0,
        stages_completed=result.get("stages_completed", 0),
        stages_attempted=result.get("stages_attempted", 0),
        elapsed_seconds=result.get("elapsed_seconds", 0.0),
        analysis_report=analysis.get("analysis_report") if analysis else None,
        stats=analysis.get("stats") if analysis else None,
        cleaning_strategy=strategy.get("cleaning_strategy") if strategy else None,
        execution_log=execution.get("execution_log") if execution else None,
        output_file=execution.get("output_file") if execution else None,
        errors=result.get("errors", []),
        pipeline_log=result.get("pipeline_log", ""),
    )
