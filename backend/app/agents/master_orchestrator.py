"""
app/agents/master_orchestrator.py

Master Orchestrator — single entry-point for the full 9-stage ML pipeline.

Architecture:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                       Master Orchestrator                               │
  │  Calls 9 stage orchestrators in strict sequential order.               │
  │  Blocking stages:  DC → FE → Scaling → MT → ME (raise PipelineStageError)│
  │  Non-blocking:     EDA → CI → MS → HT  (log warning, continue)         │
  └─────────────────────────────────────────────────────────────────────────┘

Usage:
    from app.agents.master_orchestrator import run_full_pipeline, PipelineStageError
    result = await run_full_pipeline(dataset_id, records, target_col)
"""

from __future__ import annotations

import csv
import logging
import os
import time
from typing import Any

from app.tools.executor_tools import _SANDBOX
from app.tools.content_guardrail import check_content_guardrail
from app.agents.data_cleaning_orchestrator_agent import run_pipeline as _run_dc
from app.agents.eda_orchestrator_agent import run_eda_pipeline as _run_eda
from app.agents.fe_orchestrator_agent import run_fe_orchestrator
from app.agents.scaling_orchestrator_agent import run_scaling_orchestrator
from app.agents.ci_orchestrator_agent import run_ci_orchestrator
from app.agents.mt_orchestrator_agent import run_mt_orchestrator
from app.agents.ms_orchestrator_agent import run_ms_orchestrator
from app.agents.ht_orchestrator_agent import run_ht_orchestrator
from app.agents.me_orchestrator_agent import run_me_orchestrator
from app.agents.fo_orchestrator_agent import run_fo_orchestrator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cancellation registry
# ─────────────────────────────────────────────────────────────────────────────

_cancelled_pipelines: set[str] = set()


def cancel_pipeline(dataset_id: str) -> None:
    """Signal that the pipeline for this dataset should stop."""
    _cancelled_pipelines.add(dataset_id)


def _check_cancelled(dataset_id: str) -> None:
    """Raise PipelineCancelledError if a stop was requested."""
    if dataset_id in _cancelled_pipelines:
        _cancelled_pipelines.discard(dataset_id)
        raise PipelineCancelledError(dataset_id)


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class PipelineCancelledError(Exception):
    """Raised when the user requests pipeline cancellation."""
    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        super().__init__(f"Pipeline cancelled by user for dataset: {dataset_id}")


class PipelineStageError(Exception):
    """Raised when a blocking pipeline stage fails."""

    def __init__(self, stage_name: str, original_error: Exception) -> None:
        self.stage_name = stage_name
        self.original_error = original_error
        super().__init__(
            f"Pipeline failed at stage: {stage_name} — {original_error}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> list[dict[str, Any]]:
    """Load a CSV file as a list of dicts.  Returns [] if file missing."""
    if not os.path.isfile(path):
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except Exception:
        pass
    return rows


def _build_df_stats(
    records: list[dict[str, Any]],
    target_col: str = "",
) -> tuple[dict[str, Any], list[str], list[str]]:
    """
    Build minimal df_stats dict, numeric_cols, categorical_cols from records.

    The df_stats dict embeds `records` so FE / Scaling orchestrators can
    fall back to it when `records` is not passed separately.
    """
    if not records:
        return (
            {"shape": [0, 0], "records": [], "numeric_cols": [], "categorical_cols": []},
            [],
            [],
        )

    import pandas as pd

    df = pd.DataFrame(records)
    num_cols: list[str] = [
        c
        for c in df.select_dtypes(include=["number"]).columns
        if c != target_col
    ]
    cat_cols: list[str] = [
        c for c in df.columns if c not in num_cols and c != target_col
    ]
    df_stats: dict[str, Any] = {
        "records": records,
        "shape": list(df.shape),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "rows": df.shape[0],
        "columns": df.shape[1],
    }
    return df_stats, num_cols, cat_cols


def _infer_task_type(y_values: Any) -> str:
    """Infer CLASSIFICATION vs REGRESSION from label values."""
    if y_values is None:
        return "CLASSIFICATION"
    try:
        vals = list(y_values) if hasattr(y_values, "__iter__") else [y_values]
    except Exception:
        return "CLASSIFICATION"
    if not vals:
        return "CLASSIFICATION"
    first = vals[0]
    if isinstance(first, str):
        return "CLASSIFICATION"
    unique_count = len(set(str(v) for v in vals[:300]))
    return "CLASSIFICATION" if unique_count <= 20 else "REGRESSION"


def _infer_target_dtype(y_values: Any) -> str:
    """Infer dtype string from label values."""
    try:
        vals = list(y_values) if hasattr(y_values, "__iter__") else []
    except Exception:
        return "object"
    if not vals:
        return "object"
    first = vals[0]
    if isinstance(first, float):
        return "float64"
    if isinstance(first, int):
        return "int64"
    return "object"


def _collect_eda_charts(eda_result: dict[str, Any]) -> list[str]:
    """Flatten all chart paths from EDA Phase-4 result into a single list."""
    charts_dict = eda_result.get("charts", {})
    paths: list[str] = []
    if isinstance(charts_dict, dict):
        for value in charts_dict.values():
            if isinstance(value, list):
                paths.extend(str(p) for p in value if p)
            elif isinstance(value, str) and value:
                paths.append(value)
    elif isinstance(charts_dict, list):
        paths = [str(p) for p in charts_dict if p]
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_full_pipeline(
    dataset_id: str,
    records: list[dict[str, Any]],
    target_col: str,
) -> dict[str, Any]:
    """
    Run all 9 pipeline stages end-to-end for one dataset.

    Blocking stages (raise PipelineStageError on failure):
        1 Data Cleaning  3 Feature Engineering  4 Feature Scaling
        6 Model Training  9 Model Evaluation

    Non-blocking stages (log warning, continue on failure):
        2 EDA  5 Class Imbalance  7 Model Selection  8 Hyperparameter Tuning

    Args:
        dataset_id:  UUID string of the dataset.
        records:     Raw rows as a list of plain dicts.
        target_col:  Name of the ML target column.

    Returns:
        Structured dict with status, pipeline_summary, stage_results,
        output_files, errors, pipeline_log, elapsed_seconds.

    Raises:
        PipelineStageError: if a blocking stage fails.
    """
    pipeline_start = time.time()
    errors: list[str] = []
    log_lines: list[str] = []
    stages_completed = 0

    # Result placeholders — kept so return dict is always fully populated
    dc_result: dict[str, Any] = {}
    eda_result: dict[str, Any] = {"status": "skipped", "charts": {}}
    fe_result: dict[str, Any] = {}
    scaling_result: dict[str, Any] = {}
    ci_result: dict[str, Any] = {"status": "skipped", "smote_applied": False}
    mt_result: dict[str, Any] = {}
    ms_result: dict[str, Any] = {"status": "skipped"}
    ht_result: dict[str, Any] = {
        "status": "skipped",
        "best_params": {},
        "best_model_name": "",
        "tuned_score": 0.0,
        "improvement_delta": 0.0,
    }
    me_result: dict[str, Any] = {}

    smote_applied = False
    tuning_applied = False
    best_model_name = ""
    final_model_name = ""
    performance_rating = "Unknown"
    final_metrics: dict[str, Any] = {}
    chart_paths: list[str] = []

    def _log(msg: str) -> None:
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # ─────────────────────────────────────────────────────────────────────────
    # PRE-FLIGHT — Content Safety Guardrail  (blocking)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[GUARDRAIL] Content safety check starting...")
    _log("[GUARDRAIL] Content safety check starting.")
    column_names = list(records[0].keys()) if records else []
    guardrail = check_content_guardrail(
        records=records,
        target_column=target_col,
        column_names=column_names,
    )
    if not guardrail["allowed"]:
        logger.warning(
            f"[GUARDRAIL] Dataset REJECTED — {len(guardrail['violations'])} violation(s)."
        )
        _log("[GUARDRAIL] Dataset REJECTED.")
        raise PipelineStageError(
            "Content Safety Guardrail",
            ValueError(guardrail["message"]),
        )
    logger.info("[GUARDRAIL] Content safety check passed.")
    _log("[GUARDRAIL] Content safety check passed.")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1 — Data Cleaning  (blocking)
    # ─────────────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info("[STAGE 1/9] Data Cleaning starting...")
    _log("[STAGE 1/9] Data Cleaning starting.")
    try:
        dc_result = await _run_dc(
            dataset_id=dataset_id,
            records=records,
            target_column=target_col,
        )
        stages_completed += 1
        elapsed = round(time.time() - stage_start, 2)
        logger.info(f"[STAGE 1/9] Data Cleaning complete — {elapsed}s")
        _log(f"[STAGE 1/9] Data Cleaning complete ({elapsed}s).")
    except PipelineStageError:
        raise
    except Exception as exc:
        raise PipelineStageError("Data Cleaning", exc)

    cleaned_records = _load_csv("outputs/cleaned_data.csv") or records

    _check_cancelled(dataset_id)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2 — EDA  (non-blocking)
    # ─────────────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info("[STAGE 2/9] EDA starting...")
    _log("[STAGE 2/9] EDA starting.")
    try:
        eda_result = await _run_eda(
            dataset_id=dataset_id,
            records=cleaned_records,
            target_col=target_col,
            raw_records=records,
        )
        stages_completed += 1
        elapsed = round(time.time() - stage_start, 2)
        logger.info(f"[STAGE 2/9] EDA complete — {elapsed}s")
        _log(f"[STAGE 2/9] EDA complete ({elapsed}s).")
    except Exception as exc:
        elapsed = round(time.time() - stage_start, 2)
        msg = f"EDA failed: {exc}"
        errors.append(msg)
        logger.warning(f"[STAGE 2/9] EDA failed — continuing. {msg}")
        _log(f"[STAGE 2/9] EDA failed — continuing ({elapsed}s).")
        eda_result = {"status": "skipped", "charts": {}, "errors": [msg]}

    chart_paths = _collect_eda_charts(eda_result)

    _check_cancelled(dataset_id)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3 — Feature Engineering  (blocking)
    # ─────────────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info("[STAGE 3/9] Feature Engineering starting...")
    _log("[STAGE 3/9] Feature Engineering starting.")
    try:
        fe_df_stats, fe_numeric_cols, fe_categorical_cols = _build_df_stats(
            cleaned_records, target_col
        )
        fe_result = await run_fe_orchestrator(
            df_stats=fe_df_stats,
            target_col=target_col,
            numeric_cols=fe_numeric_cols,
            categorical_cols=fe_categorical_cols,
            records=cleaned_records,
        )
        if fe_result.get("status") == "error":
            raise RuntimeError(
                f"FE stage returned error: {fe_result.get('errors', [])}"
            )
        stages_completed += 1
        elapsed = round(time.time() - stage_start, 2)
        logger.info(f"[STAGE 3/9] Feature Engineering complete — {elapsed}s")
        _log(f"[STAGE 3/9] Feature Engineering complete ({elapsed}s).")
    except PipelineStageError:
        raise
    except Exception as exc:
        raise PipelineStageError("Feature Engineering", exc)

    engineered_records = _load_csv("outputs/engineered_data.csv") or cleaned_records
    encoded_cols: list[str] = fe_result.get("features_encoded", [])
    transformed_cols: list[str] = fe_result.get("features_transformed", [])

    _check_cancelled(dataset_id)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4 — Feature Scaling  (blocking)
    # ─────────────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info("[STAGE 4/9] Feature Scaling starting...")
    _log("[STAGE 4/9] Feature Scaling starting.")
    try:
        sc_df_stats, sc_numeric_cols, sc_categorical_cols = _build_df_stats(
            engineered_records, target_col
        )
        scaling_result = await run_scaling_orchestrator(
            df_stats=sc_df_stats,
            target_col=target_col,
            numeric_cols=sc_numeric_cols,
            categorical_cols=sc_categorical_cols,
            encoded_cols=encoded_cols,
            transformed_cols=transformed_cols,
        )
        if scaling_result.get("status") == "error":
            raise RuntimeError(
                f"Scaling stage returned error: {scaling_result.get('errors', [])}"
            )
        # Verify sandbox has been populated by scaling executor
        if _SANDBOX.get("X_train") is None or _SANDBOX.get("X_test") is None:
            raise RuntimeError(
                "X_train / X_test not found in sandbox after Feature Scaling."
            )
        stages_completed += 1
        elapsed = round(time.time() - stage_start, 2)
        logger.info(f"[STAGE 4/9] Feature Scaling complete — {elapsed}s")
        _log(f"[STAGE 4/9] Feature Scaling complete ({elapsed}s).")
    except PipelineStageError:
        raise
    except Exception as exc:
        raise PipelineStageError("Feature Scaling", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 5 — Class Imbalance  (non-blocking)
    # ─────────────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info("[STAGE 5/9] Class Imbalance starting...")
    _log("[STAGE 5/9] Class Imbalance starting.")

    _check_cancelled(dataset_id)

    y_train_raw = _SANDBOX.get("y_train")
    y_test_raw = _SANDBOX.get("y_test")
    task_type = _infer_task_type(y_train_raw)
    target_dtype = _infer_target_dtype(y_train_raw)

    try:
        ci_result = await run_ci_orchestrator(
            y_train=y_train_raw,
            y_test=y_test_raw,
            target_col=target_col,
            target_dtype=target_dtype,
            task_type=task_type,
        )
        smote_applied = bool(ci_result.get("smote_applied", False))
        stages_completed += 1
        elapsed = round(time.time() - stage_start, 2)
        logger.info(f"[STAGE 5/9] Class Imbalance complete — {elapsed}s")
        _log(f"[STAGE 5/9] Class Imbalance complete ({elapsed}s).")
    except Exception as exc:
        elapsed = round(time.time() - stage_start, 2)
        msg = f"Class Imbalance failed: {exc}"
        errors.append(msg)
        smote_applied = False
        logger.warning(f"[STAGE 5/9] Class Imbalance failed — continuing. {msg}")
        _log(f"[STAGE 5/9] Class Imbalance failed — continuing ({elapsed}s).")
        ci_result = {
            "status": "skipped",
            "smote_applied": False,
            "technique_applied": "none",
            "errors": [msg],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 6 — Model Training  (blocking)
    # ─────────────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info("[STAGE 6/9] Model Training starting...")
    _log("[STAGE 6/9] Model Training starting.")
    try:
        # Prefer balanced data from CI; fall back to original splits
        X_train_val = _SANDBOX.get("X_train_bal") or _SANDBOX.get("X_train")
        X_test_val = _SANDBOX.get("X_test")
        y_train_val = _SANDBOX.get("y_train_bal") or _SANDBOX.get("y_train")
        y_test_val = _SANDBOX.get("y_test")

        if X_train_val is None or X_test_val is None:
            raise RuntimeError(
                "X_train / X_test not found in sandbox — scaling stage may have failed."
            )

        feature_names: list[str] = []
        if hasattr(X_train_val, "columns"):
            feature_names = list(X_train_val.columns)

        mt_result = await run_mt_orchestrator(
            X_train=X_train_val,
            X_test=X_test_val,
            y_train=y_train_val,
            y_test=y_test_val,
            target_col=target_col,
            feature_names=feature_names,
            scaling_summary=scaling_result,
        )
        if mt_result.get("status") == "error":
            raise RuntimeError(
                f"MT stage returned error: {mt_result.get('errors', [])}"
            )
        stages_completed += 1
        elapsed = round(time.time() - stage_start, 2)
        logger.info(f"[STAGE 6/9] Model Training complete — {elapsed}s")
        _log(f"[STAGE 6/9] Model Training complete ({elapsed}s).")
    except PipelineStageError:
        raise
    except Exception as exc:
        raise PipelineStageError("Model Training", exc)

    best_model_name = str(mt_result.get("best_model", ""))

    _check_cancelled(dataset_id)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 7 — Model Selection  (non-blocking)
    # ─────────────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info("[STAGE 7/9] Model Selection starting...")
    _log("[STAGE 7/9] Model Selection starting.")
    try:
        ms_result = await run_ms_orchestrator(mt_result)
        stages_completed += 1
        elapsed = round(time.time() - stage_start, 2)
        logger.info(f"[STAGE 7/9] Model Selection complete — {elapsed}s")
        _log(f"[STAGE 7/9] Model Selection complete ({elapsed}s).")
    except Exception as exc:
        elapsed = round(time.time() - stage_start, 2)
        msg = f"Model Selection failed: {exc}"
        errors.append(msg)
        logger.warning(f"[STAGE 7/9] Model Selection failed — continuing. {msg}")
        _log(f"[STAGE 7/9] Model Selection failed — continuing ({elapsed}s).")
        ms_result = {
            "status": "skipped",
            "final_model_name": best_model_name,
            "errors": [msg],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 8 — Hyperparameter Tuning  (non-blocking)
    # ─────────────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info("[STAGE 8/9] Hyperparameter Tuning starting...")
    _log("[STAGE 8/9] Hyperparameter Tuning starting.")
    try:
        ht_result = await run_ht_orchestrator(mt_result)
        tuning_applied = bool(ht_result.get("best_params"))
        stages_completed += 1
        elapsed = round(time.time() - stage_start, 2)
        logger.info(f"[STAGE 8/9] Hyperparameter Tuning complete — {elapsed}s")
        _log(f"[STAGE 8/9] Hyperparameter Tuning complete ({elapsed}s).")
    except Exception as exc:
        elapsed = round(time.time() - stage_start, 2)
        msg = f"Hyperparameter Tuning failed: {exc}"
        errors.append(msg)
        tuning_applied = False
        logger.warning(f"[STAGE 8/9] Hyperparameter Tuning failed — continuing. {msg}")
        _log(f"[STAGE 8/9] Hyperparameter Tuning failed — continuing ({elapsed}s).")
        ht_result = {
            "status": "skipped",
            "best_params": {},
            "best_model_name": best_model_name,
            "tuned_score": 0.0,
            "improvement_delta": 0.0,
            "errors": [msg],
        }

    final_model_name = str(ht_result.get("best_model_name") or best_model_name)

    _check_cancelled(dataset_id)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 9 — Model Evaluation  (blocking)
    # ─────────────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info("[STAGE 9/9] Model Evaluation starting...")
    _log("[STAGE 9/9] Model Evaluation starting.")
    try:
        me_result = await run_me_orchestrator(
            ht_results=ht_result,
            mt_results=mt_result,
        )
        if me_result.get("status") == "error":
            raise RuntimeError(
                f"ME stage returned error: {me_result.get('errors', [])}"
            )
        stages_completed += 1
        elapsed = round(time.time() - stage_start, 2)
        logger.info(f"[STAGE 9/9] Model Evaluation complete — {elapsed}s")
        _log(f"[STAGE 9/9] Model Evaluation complete ({elapsed}s).")
    except PipelineStageError:
        raise
    except Exception as exc:
        raise PipelineStageError("Model Evaluation", exc)

    performance_rating = str(me_result.get("performance_rating", "Unknown"))
    final_metrics = dict(me_result.get("final_metrics", {}))
    final_model_name = str(me_result.get("final_model_name", final_model_name))

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 10 — Final Output  (non-blocking)
    # ─────────────────────────────────────────────────────────────────────────
    stage_start = time.time()
    logger.info("[STAGE 10/10] Final Output starting...")
    _log("[STAGE 10/10] Final Output starting.")
    fo_result: dict[str, Any] = {"status": "skipped"}
    try:
        fo_result = await run_fo_orchestrator(
            me_result=me_result,
            mt_result=mt_result,
            ht_result=ht_result,
            dataset_id=dataset_id,
            target_col=target_col,
        )
        stages_completed += 1
        elapsed = round(time.time() - stage_start, 2)
        logger.info(f"[STAGE 10/10] Final Output complete — {elapsed}s")
        _log(f"[STAGE 10/10] Final Output complete ({elapsed}s).")
    except Exception as exc:
        elapsed = round(time.time() - stage_start, 2)
        msg = f"Final Output failed: {exc}"
        errors.append(msg)
        logger.warning(f"[STAGE 10/10] Final Output failed — continuing. {msg}")
        _log(f"[STAGE 10/10] Final Output failed — continuing ({elapsed}s).")
        fo_result = {"status": "skipped", "errors": [msg]}

    # ─────────────────────────────────────────────────────────────────────────
    # Final structured result
    # ─────────────────────────────────────────────────────────────────────────
    elapsed_total = round(time.time() - pipeline_start, 2)
    logger.info(
        f"[PIPELINE] Complete — {stages_completed}/10 stages in {elapsed_total}s."
    )
    _log(f"Pipeline complete — {stages_completed}/10 stages in {elapsed_total}s.")

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "target_col": target_col,
        "stages_completed": stages_completed,
        "pipeline_summary": {
            "model_used": final_model_name,
            "performance_rating": performance_rating,
            "accuracy": float(final_metrics.get("accuracy", 0.0)),
            "f1": float(final_metrics.get("f1", 0.0)),
            "smote_applied": smote_applied,
            "tuning_applied": tuning_applied,
            "eda_charts": chart_paths,
        },
        "stage_results": {
            "data_cleaning": dc_result,
            "eda": eda_result,
            "feature_engineering": fe_result,
            "feature_scaling": scaling_result,
            "class_imbalance": ci_result,
            "model_training": mt_result,
            "model_selection": ms_result,
            "hyperparameter_tuning": ht_result,
            "model_evaluation": me_result,
        },
        "output_files": {
            "model": "outputs/final_model.joblib",
            "report": "outputs/final_report.pdf",
            "cleaned_data": "outputs/cleaned_data.csv",
            "evaluation_summary": "outputs/evaluation_summary.json",
            "results_manifest": "outputs/results_manifest.json",
        },
        "final_output": fo_result,
        "errors": errors,
        "pipeline_log": "\n".join(log_lines),
        "elapsed_seconds": elapsed_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public aliases — mirror the names used in __init__.py
# ─────────────────────────────────────────────────────────────────────────────

run_dc_orchestrator = _run_dc
run_eda_orchestrator = _run_eda
