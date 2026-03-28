"""
app/api/pipeline.py

FastAPI router — full ML pipeline endpoints.

Endpoints:
    POST /api/pipeline/run          — trigger full 9-stage pipeline
    GET  /api/pipeline/status/{id}  — check pipeline progress
    GET  /api/results/{id}          — return evaluation results for UI
    GET  /api/download/{id}/{type}  — serve output file for download
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.dataset import UserDataset
from app.agents.master_orchestrator import PipelineStageError, run_full_pipeline

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    dataset_id: str
    """UUID of the previously imported dataset."""
    target_col: str
    """Name of the ML target / prediction column."""


class PipelineSummary(BaseModel):
    model_used: str = ""
    performance_rating: str = ""
    accuracy: float = 0.0
    f1: float = 0.0
    smote_applied: bool = False
    tuning_applied: bool = False
    eda_charts: list[str] = []


class PipelineRunResponse(BaseModel):
    status: str
    dataset_id: str
    target_col: str
    stages_completed: int = 0
    pipeline_summary: PipelineSummary = PipelineSummary()
    output_files: dict[str, str] = {}
    errors: list[str] = []
    elapsed_seconds: float = 0.0


class PipelineStatusResponse(BaseModel):
    dataset_id: str
    status: str           # pending / running / complete / failed
    stages_completed: int = 0
    last_stage: str = ""


class PipelineResultsResponse(BaseModel):
    dataset_id: str
    evaluation_summary: dict[str, Any] = {}
    available_downloads: list[dict[str, str]] = []


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

_DOWNLOAD_MAP: dict[str, tuple[str, str]] = {
    "model":        ("outputs/final_model.joblib", "application/octet-stream"),
    "report":       ("outputs/final_report.pdf",   "application/pdf"),
    "cleaned_data": ("outputs/cleaned_data.csv",   "text/csv"),
    "evaluation":   ("outputs/evaluation_summary.json", "application/json"),
}


def _load_results_manifest() -> dict[str, Any]:
    """Read outputs/results_manifest.json — the single source of truth for the UI."""
    path = "outputs/results_manifest.json"
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}



def _infer_pipeline_status() -> tuple[str, int, str]:
    """
    Infer pipeline status from files on disk.

    Returns:
        (status, stages_completed, last_stage)
    """
    stages = [
        ("cleaned_data.csv",          "Data Cleaning",         1),
        ("eda_stats.json",             "EDA",                   2),
        ("engineered_data.csv",        "Feature Engineering",   3),
        ("scaling_summary.json",       "Feature Scaling",       4),
        ("balance_report.json",        "Class Imbalance",       5),
        ("training_results.json",      "Model Training",        6),
        ("model_selection_summary.json","Model Selection",      7),
        ("tuning_results.json",        "Hyperparameter Tuning", 8),
        ("evaluation_summary.json",    "Model Evaluation",      9),
        ("results_manifest.json",      "Final Output",         10),
    ]
    last_stage = ""
    stages_completed = 0
    for filename, stage_name, stage_num in stages:
        if os.path.isfile(f"outputs/{filename}") or os.path.isfile(f"charts/{filename}"):
            last_stage = stage_name
            stages_completed = stage_num

    if stages_completed == 0:
        return "pending", 0, ""
    if stages_completed == 10:
        return "complete", 10, last_stage
    return "running", stages_completed, last_stage


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — POST /api/pipeline/run
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/api/pipeline/run", response_model=PipelineRunResponse)
async def run_pipeline_endpoint(
    body: PipelineRunRequest,
    db: AsyncSession = Depends(get_db),
) -> PipelineRunResponse:
    """
    Trigger the full 9-stage ML pipeline for a dataset.

    Fetches dataset from PostgreSQL, validates the target column,
    then runs all pipeline stages via run_full_pipeline().
    """
    # 1. Validate and parse dataset_id
    try:
        dataset_uuid = uuid.UUID(body.dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id (must be a UUID).")

    # 2. Fetch dataset from DB
    result = await db.execute(
        select(UserDataset).where(UserDataset.id == dataset_uuid)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{body.dataset_id}' not found.",
        )

    # 3. Validate raw_data and target_col
    raw_data: list[dict] = dataset.raw_data
    if not isinstance(raw_data, list) or not raw_data:
        raise HTTPException(
            status_code=422, detail="Dataset raw_data is empty or malformed."
        )

    col_names = list(raw_data[0].keys())
    if body.target_col not in col_names:
        raise HTTPException(
            status_code=400,
            detail=(
                f"target_col '{body.target_col}' not found in dataset columns: {col_names}."
            ),
        )

    # 4. Run full pipeline
    try:
        pipeline_result = await run_full_pipeline(
            dataset_id=body.dataset_id,
            records=raw_data,
            target_col=body.target_col,
        )
    except PipelineStageError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "stage_name": exc.stage_name,
                "original_error": str(exc.original_error),
            },
        )

    # 5. Return structured response
    summary_raw = pipeline_result.get("pipeline_summary", {})
    return PipelineRunResponse(
        status=pipeline_result.get("status", "unknown"),
        dataset_id=body.dataset_id,
        target_col=body.target_col,
        stages_completed=int(pipeline_result.get("stages_completed", 0)),
        pipeline_summary=PipelineSummary(
            model_used=str(summary_raw.get("model_used", "")),
            performance_rating=str(summary_raw.get("performance_rating", "")),
            accuracy=float(summary_raw.get("accuracy", 0.0)),
            f1=float(summary_raw.get("f1", 0.0)),
            smote_applied=bool(summary_raw.get("smote_applied", False)),
            tuning_applied=bool(summary_raw.get("tuning_applied", False)),
            eda_charts=list(summary_raw.get("eda_charts", [])),
        ),
        output_files=dict(pipeline_result.get("output_files", {})),
        errors=list(pipeline_result.get("errors", [])),
        elapsed_seconds=float(pipeline_result.get("elapsed_seconds", 0.0)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — GET /api/pipeline/status/{dataset_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/api/pipeline/status/{dataset_id}",
    response_model=PipelineStatusResponse,
)
async def pipeline_status(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
) -> PipelineStatusResponse:
    """
    Check if and how far the pipeline has run for this dataset.

    Returns status: pending / running / complete / failed
    based on the presence of expected output files on disk.
    """
    # Validate dataset exists
    try:
        dataset_uuid = uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id (must be a UUID).")

    result = await db.execute(
        select(UserDataset).where(UserDataset.id == dataset_uuid)
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_id}' not found."
        )

    status, stages_completed, last_stage = _infer_pipeline_status()
    return PipelineStatusResponse(
        dataset_id=dataset_id,
        status=status,
        stages_completed=stages_completed,
        last_stage=last_stage,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — GET /api/results/{dataset_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/api/results/{dataset_id}")
async def get_results(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """
    Return full pipeline results for the Next.js frontend.

    Loads outputs/results_manifest.json (written by Stage 10 — Final Output).
    Returns 404 if the manifest does not exist yet.
    """
    try:
        dataset_uuid = uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id (must be a UUID).")

    result = await db.execute(
        select(UserDataset).where(UserDataset.id == dataset_uuid)
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_id}' not found."
        )

    manifest = _load_results_manifest()
    if not manifest:
        raise HTTPException(
            status_code=404,
            detail=(
                "Pipeline results not found. "
                "Run POST /api/pipeline/run first, or wait for Stage 10 to complete."
            ),
        )

    return JSONResponse(content=manifest)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4 — GET /api/download/{dataset_id}/{file_type}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/api/download/{dataset_id}/{file_type}")
async def download_file(
    dataset_id: str,
    file_type: str,
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """
    Serve a pipeline output file for download.

    file_type options:
        model        → outputs/final_model.joblib
        report       → outputs/final_report.pdf
        cleaned_data → outputs/cleaned_data.csv
        evaluation   → outputs/evaluation_summary.json

    Returns 404 if the file does not exist yet.
    """
    try:
        dataset_uuid = uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset_id (must be a UUID).")

    result = await db.execute(
        select(UserDataset).where(UserDataset.id == dataset_uuid)
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_id}' not found."
        )

    if file_type not in _DOWNLOAD_MAP:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown file_type '{file_type}'. "
                f"Valid options: {list(_DOWNLOAD_MAP.keys())}."
            ),
        )

    file_path, media_type = _DOWNLOAD_MAP[file_type]
    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"File '{file_path}' does not exist yet. "
                "Run POST /api/pipeline/run first."
            ),
        )

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=os.path.basename(file_path),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5 — GET /api/charts/{dataset_id}/{filename}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/api/charts/{dataset_id}/{filename}")
async def get_chart(
    dataset_id: str,
    filename: str,
) -> FileResponse:
    """
    Serve an EDA or evaluation chart image.

    Searches charts/ directory for the requested filename.
    Returns 404 if not found.
    """
    # Sanitise filename — no path traversal
    safe_name = os.path.basename(filename)
    candidates = [
        f"charts/{safe_name}",
        f"outputs/charts/{safe_name}",
        f"outputs/{safe_name}",
    ]
    for path in candidates:
        if os.path.isfile(path):
            ext = safe_name.rsplit(".", 1)[-1].lower()
            media_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "svg": "image/svg+xml"}
            return FileResponse(
                path=path,
                media_type=media_map.get(ext, "application/octet-stream"),
                filename=safe_name,
            )
    raise HTTPException(
        status_code=404,
        detail=f"Chart '{safe_name}' not found.",
    )
