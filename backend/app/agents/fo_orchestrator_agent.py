"""
app/agents/fo_orchestrator_agent.py

Final Output Orchestrator — Stage 10 of the ML pipeline.

Pipeline (single phase):
    STEP 1 — Build pipeline summary from sandbox variables + JSON files on disk
    STEP 2 — Save final model as joblib (PipelineStageError on failure)
    STEP 3 — Run FO executor agent (dataset_id + pipeline_summary)
    STEP 4 — Load and return results_manifest.json

Usage:
    from app.agents.fo_orchestrator_agent import run_fo_orchestrator
    result = await run_fo_orchestrator(
        me_result, mt_result, ht_result, dataset_id, target_col
    )
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.tools.executor_tools import _SANDBOX
from app.tools.fo_tools import (
    collect_output_files,
    get_pipeline_summary,
    load_json_safe,
    save_model_as_joblib,
)

logger = logging.getLogger(__name__)


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
    Run Stage 10 — Final Output.

    STEP 1 — Build pipeline summary using get_pipeline_summary(_SANDBOX) and
             load_json_safe() for stage JSON files on disk.
    STEP 2 — Save final_model to outputs/final_model.joblib via
             save_model_as_joblib().  Raises PipelineStageError if the model
             is None (i.e. training never completed).
    STEP 3 — Run FO executor agent, passing dataset_id and pipeline_summary.
    STEP 4 — Load outputs/results_manifest.json and build the return dict.

    Args:
        me_result:   Output from run_me_orchestrator() — kept for backward compat.
        mt_result:   Output from run_mt_orchestrator() — kept for backward compat.
        ht_result:   Output from run_ht_orchestrator() — kept for backward compat.
        dataset_id:  UUID string of the dataset.
        target_col:  ML target column name.

    Returns:
        {
            "status":                   "success" | "partial",
            "dataset_id":               str,
            "output_files":             dict,
            "pipeline_summary":         dict,
            "eda_section":              dict,
            "model_evaluation_section": dict,
            "downloads":                list,
            "next_phase":               "complete",
        }

    Raises:
        PipelineStageError: if save_model_as_joblib() raises (model is None).
    """
    # Lazy import to avoid circular dependency with master_orchestrator
    from app.agents.master_orchestrator import PipelineStageError
    from app.agents.final_output_agent.executor_agent import run_fo_executor

    # ── STEP 1: Build pipeline summary ───────────────────────────────────────
    logger.info("[FO] STEP 1 — Building pipeline summary.")
    pipeline_summary = get_pipeline_summary(_SANDBOX)
    model_name = pipeline_summary.get("model_info", {}).get("final_model_name", "?")
    accuracy   = pipeline_summary.get("metrics", {}).get("accuracy", 0.0)
    logger.info(f"[FO] Pipeline summary built — model={model_name}, accuracy={accuracy:.4f}.")

    # ── STEP 2: Save model as joblib ──────────────────────────────────────────
    logger.info("[FO] STEP 2 — Saving final_model to outputs/final_model.joblib.")
    try:
        model_path = save_model_as_joblib()
        model_size = os.path.getsize(model_path)
        logger.info(f"[FO] Model saved: {model_path} ({model_size:,} bytes).")
    except Exception as exc:
        logger.error(f"[FO] save_model_as_joblib failed — {exc}")
        raise PipelineStageError("Final Output", exc)

    # ── STEP 3: Run executor agent ────────────────────────────────────────────
    logger.info("[FO] STEP 3 — Running FO executor agent.")
    fo_result: dict[str, Any] = {}
    try:
        fo_result = await run_fo_executor(
            {
                "dataset_id":       dataset_id,
                "target_col":       target_col,
                "pipeline_summary": pipeline_summary,
                # Kept for backward compat with executor's user_message builder
                "me_result":        me_result,
                "mt_result":        mt_result,
                "ht_result":        ht_result,
            }
        )
        logger.info(f"[FO] Executor complete. Status: {fo_result.get('status', '?')}.")
    except Exception as exc:
        logger.warning(f"[FO] Executor agent failed: {exc} — continuing to manifest load.")

    # ── STEP 4: Load manifest from disk ──────────────────────────────────────
    logger.info("[FO] STEP 4 — Loading results_manifest.json.")
    manifest = load_json_safe("outputs/results_manifest.json")
    if not manifest:
        logger.warning("[FO] results_manifest.json not found or empty after executor run.")

    # Collect all output files that are present on disk
    scanned = collect_output_files()

    # Extract manifest sections (empty dicts/lists if executor failed)
    eda_section:        dict[str, Any]        = dict(manifest.get("eda_section", {}))
    model_eval_section: dict[str, Any]        = dict(manifest.get("model_evaluation_section", {}))
    downloads:          list[dict[str, Any]]  = list(manifest.get("downloads", []))

    logger.info(
        f"[FO] Final Output complete — model={model_name}, "
        f"downloads={len(downloads)}, "
        f"eda_charts={len(eda_section.get('charts', []))}."
    )

    return {
        "status":                   "success" if manifest else "partial",
        "dataset_id":               dataset_id,
        "output_files":             {
            "model_files":       scanned["model_files"],
            "data_files":        scanned["data_files"],
            "evaluation_charts": scanned["evaluation_charts"],
            "eda_charts":        scanned["eda_charts"],
            "json_reports":      scanned["json_reports"],
        },
        "pipeline_summary":         pipeline_summary,
        "eda_section":              eda_section,
        "model_evaluation_section": model_eval_section,
        "downloads":                downloads,
        "next_phase":               "complete",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Thin ADK agent shim — kept for __init__.py consistency with other stages
# ─────────────────────────────────────────────────────────────────────────────

from google.adk.agents import Agent  # noqa: E402

fo_orchestrator_agent = Agent(
    name="fo_orchestrator_agent",
    model="gemini-2.0-flash",
    description=(
        "Orchestrates the Final Output stage: builds pipeline summary from the "
        "sandbox, saves final_model.joblib, and builds results_manifest.json "
        "for the Next.js frontend."
    ),
    instruction=(
        "You coordinate the Final Output stage. "
        "Delegate all work to run_fo_orchestrator()."
    ),
    tools=[],
)
