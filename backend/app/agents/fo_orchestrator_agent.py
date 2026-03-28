"""
app/agents/fo_orchestrator_agent.py

Final Output Orchestrator — Stage 10 of the ML pipeline.

Single-phase pipeline:
    Executor — verify files → save joblib → build results_manifest.json → verify

Inputs come from stages 6 (MT), 8 (HT), and 9 (ME).
Additional context (eda_charts, dataset_stats) is resolved from disk.

Usage:
    from app.agents.fo_orchestrator_agent import run_fo_orchestrator
    result = await run_fo_orchestrator(
        me_result, mt_result, ht_result, dataset_id, target_col
    )
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any

from app.tools.fo_tools import load_json_safe

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_eda_charts() -> list[str]:
    """
    Gather EDA chart file paths from the charts/ directory.
    Returns relative paths suitable for GET /api/charts/{dataset_id}/{filename}.
    """
    charts: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
        charts.extend(glob.glob(f"charts/{ext}"))
    charts.sort()
    return charts


def _compute_dataset_stats() -> dict[str, Any]:
    """
    Estimate row/feature counts from disk files.

    Priority: training_results.json → engineered_data.csv (line count) → cleaned_data.csv
    """
    # Try training_results.json first (most reliable)
    tr = load_json_safe("outputs/training_results.json")
    if tr:
        rows = int(tr.get("train_samples", 0)) + int(tr.get("test_samples", 0))
        features = int(tr.get("feature_count", tr.get("n_features", 0)))
        if rows > 0 or features > 0:
            return {"rows": rows, "features": features}

    # Fall back to counting lines in engineered_data.csv
    for csv_path in ("outputs/engineered_data.csv", "outputs/cleaned_data.csv"):
        if os.path.isfile(csv_path):
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    lines = sum(1 for _ in f)
                rows = max(0, lines - 1)  # subtract header
                return {"rows": rows, "features": 0}
            except Exception:
                pass

    return {"rows": 0, "features": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Public orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_fo_orchestrator(
    me_result: dict[str, Any],
    mt_result: dict[str, Any],
    ht_result: dict[str, Any],
    dataset_id: str,
    target_col: str,
) -> dict[str, Any]:
    """
    Run Stage 10 — Final Output — for one dataset.

    Collects EDA charts, computes dataset stats, invokes the FO executor
    agent to save final_model.joblib and build results_manifest.json.

    Args:
        me_result:   Output from run_me_orchestrator()
        mt_result:   Output from run_mt_orchestrator()
        ht_result:   Output from run_ht_orchestrator()
        dataset_id:  UUID string of the dataset
        target_col:  ML target column name

    Returns:
        {
            "status":        "success" | "error",
            "manifest":      dict (contents of results_manifest.json),
            "execution_log": str,
            "dataset_id":    str,
            "target_col":    str,
        }
    """
    from app.agents.final_output_agent.executor_agent import run_fo_executor

    logger.info("[FO] Final Output Orchestrator starting.")

    # Resolve context from disk
    eda_charts = _collect_eda_charts()
    dataset_stats = _compute_dataset_stats()
    elapsed_seconds = float(me_result.get("elapsed_seconds", 0.0))

    logger.info(
        f"[FO] EDA charts found: {len(eda_charts)}, "
        f"dataset rows: {dataset_stats.get('rows', 0)}"
    )

    orchestrator_input: dict[str, Any] = {
        "me_result":       me_result,
        "mt_result":       mt_result,
        "ht_result":       ht_result,
        "dataset_id":      dataset_id,
        "target_col":      target_col,
        "eda_charts":      eda_charts,
        "elapsed_seconds": elapsed_seconds,
        "dataset_stats":   dataset_stats,
    }

    try:
        fo_result = await run_fo_executor(orchestrator_input)
    except Exception as exc:
        logger.error(f"[FO] Executor failed: {exc}")
        return {
            "status":        "error",
            "manifest":      {},
            "execution_log": str(exc),
            "dataset_id":    dataset_id,
            "target_col":    target_col,
        }

    # Load the manifest written by the executor
    manifest = load_json_safe("outputs/results_manifest.json")
    if not manifest:
        logger.warning("[FO] results_manifest.json not found after executor run.")

    logger.info(
        f"[FO] Final Output complete. "
        f"Model: {manifest.get('pipeline_summary', {}).get('model_used', '?')}. "
        f"Downloads: {len(manifest.get('downloads', []))}."
    )

    return {
        "status":        "success" if manifest else "partial",
        "manifest":      manifest,
        "execution_log": fo_result.get("execution_log", ""),
        "verify_manifest": fo_result.get("verify_manifest", ""),
        "dataset_id":    dataset_id,
        "target_col":    target_col,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Thin ADK agent shim — kept for __init__.py consistency with other stages
# ─────────────────────────────────────────────────────────────----------─────

from google.adk.agents import Agent

fo_orchestrator_agent = Agent(
    name="fo_orchestrator_agent",
    model="gemini-2.0-flash",
    description=(
        "Orchestrates the Final Output stage: saves final_model.joblib "
        "and builds results_manifest.json for the Next.js frontend."
    ),
    instruction=(
        "You coordinate the Final Output stage. "
        "Delegate all work to run_fo_executor via run_fo_orchestrator()."
    ),
    tools=[],
)
