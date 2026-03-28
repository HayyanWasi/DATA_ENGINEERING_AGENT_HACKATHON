"""
app/tools/fo_tools.py

Final Output helper functions — plain Python, NOT FunctionTools.

Called by fo_executor_agent via execute_python to collect pipeline
outputs and build results_manifest.json (the single source of truth
for the Next.js frontend).

Functions:
    collect_output_files()       — scan outputs/ and return presence dict
    load_json_safe(path)         — load JSON, return {} on any error
    get_pipeline_summary(...)    — assemble summary from stage results
    save_model_as_joblib()       — joblib-dump _SANDBOX["final_model"]
    build_results_manifest(...)  — write outputs/results_manifest.json
"""

from __future__ import annotations

import json
import os
from typing import Any

from app.tools.executor_tools import _SANDBOX

# ─────────────────────────────────────────────────────────────────────────────
# Expected output filenames — one per pipeline stage
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_FILES: dict[str, str] = {
    "cleaned_data":           "outputs/cleaned_data.csv",
    "eda_stats":              "outputs/eda_stats.json",
    "engineered_data":        "outputs/engineered_data.csv",
    "scaling_summary":        "outputs/scaling_summary.json",
    "balance_report":         "outputs/balance_report.json",
    "training_results":       "outputs/training_results.json",
    "model_selection_summary":"outputs/model_selection_summary.json",
    "tuning_results":         "outputs/tuning_results.json",
    "evaluation_summary":     "outputs/evaluation_summary.json",
    "final_model":            "outputs/final_model.joblib",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. collect_output_files
# ─────────────────────────────────────────────────────────────────────────────

def collect_output_files() -> dict[str, Any]:
    """
    Scan the outputs/ directory and report which expected files are present.

    Returns:
        {
            "present": ["cleaned_data.csv", ...],
            "missing": ["final_model.joblib", ...],
            "paths": {"cleaned_data": "outputs/cleaned_data.csv", ...},
            "stages_complete": int  (0-9, counts stage output files present),
        }
    """
    os.makedirs("outputs", exist_ok=True)
    present: list[str] = []
    missing: list[str] = []
    paths: dict[str, str] = {}

    for key, path in _STAGE_FILES.items():
        if os.path.isfile(path):
            present.append(os.path.basename(path))
            paths[key] = path
        else:
            missing.append(os.path.basename(path))

    # Stage count excludes the model file (not a stage output per se)
    stage_keys = [
        "cleaned_data", "eda_stats", "engineered_data", "scaling_summary",
        "balance_report", "training_results", "model_selection_summary",
        "tuning_results", "evaluation_summary",
    ]
    stages_complete = sum(1 for k in stage_keys if k in paths)

    return {
        "present": present,
        "missing": missing,
        "paths": paths,
        "stages_complete": stages_complete,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. load_json_safe
# ─────────────────────────────────────────────────────────────────────────────

def load_json_safe(path: str) -> dict[str, Any]:
    """Load a JSON file; return {} on any error (missing file, bad JSON, etc.)."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. get_pipeline_summary
# ─────────────────────────────────────────────────────────────────────────────

def get_pipeline_summary(
    me_result: dict[str, Any],
    mt_result: dict[str, Any],
    ht_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Assemble a flat pipeline summary dict from the three final stage results.

    Returns:
        {
            "model_used":         str,
            "performance_rating": str,
            "accuracy":           float,
            "f1":                 float,
            "precision":          float,
            "recall":             float,
            "roc_auc":            float,
            "smote_applied":      bool,
            "tuning_applied":     bool,
            "best_params":        dict,
            "model_comparison":   list[dict],
        }
    """
    final_metrics: dict[str, Any] = dict(me_result.get("final_metrics", {}))

    model_used = str(
        me_result.get("final_model_name")
        or ht_result.get("best_model_name")
        or mt_result.get("best_model", "")
    )

    # Build model comparison table from mt_result
    raw_models = mt_result.get("all_models", []) or mt_result.get("model_comparison", [])
    model_comparison: list[dict[str, Any]] = []
    for m in raw_models:
        if not isinstance(m, dict):
            continue
        name = str(m.get("model", m.get("name", "")))
        acc = float(m.get("accuracy", m.get("test_accuracy", 0.0)))
        f1 = float(m.get("f1", m.get("f1_score", m.get("f1_weighted", 0.0))))
        if name == model_used:
            status = "Best"
        elif acc >= 0.85:
            status = "Good"
        else:
            status = "Fair"
        model_comparison.append({"model": name, "accuracy": acc, "f1": f1, "status": status})

    # Sort best first
    model_comparison.sort(key=lambda x: x["accuracy"], reverse=True)

    # smote_applied from balance_report.json on disk (more reliable than result dict)
    balance_report = load_json_safe("outputs/balance_report.json")
    smote_applied = bool(balance_report.get("smote_applied", False))

    return {
        "model_used":         model_used,
        "performance_rating": str(me_result.get("performance_rating", "Unknown")),
        "accuracy":           float(final_metrics.get("accuracy", 0.0)),
        "f1":                 float(final_metrics.get("f1", final_metrics.get("f1_score", final_metrics.get("f1_weighted", 0.0)))),
        "precision":          float(final_metrics.get("precision", 0.0)),
        "recall":             float(final_metrics.get("recall", 0.0)),
        "roc_auc":            float(final_metrics.get("roc_auc", final_metrics.get("auc", 0.0))),
        "smote_applied":      smote_applied,
        "tuning_applied":     bool(ht_result.get("best_params")),
        "best_params":        dict(ht_result.get("best_params", {})),
        "model_comparison":   model_comparison,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. save_model_as_joblib
# ─────────────────────────────────────────────────────────────────────────────

def save_model_as_joblib() -> dict[str, Any]:
    """
    Dump _SANDBOX["final_model"] to outputs/final_model.joblib using joblib.

    Returns:
        {"success": True,  "path": "outputs/final_model.joblib"}
        {"success": False, "error": "reason string"}
    """
    os.makedirs("outputs", exist_ok=True)
    model = _SANDBOX.get("final_model")
    if model is None:
        return {"success": False, "error": "final_model not found in sandbox."}
    try:
        import joblib
        path = "outputs/final_model.joblib"
        joblib.dump(model, path)
        return {"success": True, "path": path}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# 5. build_results_manifest
# ─────────────────────────────────────────────────────────────────────────────

def build_results_manifest(
    dataset_id: str,
    target_col: str,
    pipeline_summary: dict[str, Any],
    eda_charts: list[str],
    elapsed_seconds: float,
    dataset_stats: dict[str, Any],
) -> dict[str, Any]:
    """
    Build and persist outputs/results_manifest.json.

    This is the single source of truth read by GET /api/results/{dataset_id}
    and rendered by the Next.js results page.

    Returns the manifest dict (also written to disk).
    """
    os.makedirs("outputs", exist_ok=True)

    # Collect download links for files currently on disk
    download_candidates = {
        "model":        "outputs/final_model.joblib",
        "cleaned_data": "outputs/cleaned_data.csv",
        "evaluation":   "outputs/evaluation_summary.json",
        "report":       "outputs/final_report.pdf",
    }
    downloads: list[dict[str, str]] = []
    for file_type, path in download_candidates.items():
        if os.path.isfile(path):
            downloads.append({
                "file_type": file_type,
                "filename":  os.path.basename(path),
                "url":       f"/api/download/{dataset_id}/{file_type}",
            })

    manifest: dict[str, Any] = {
        "dataset_id":      dataset_id,
        "target_col":      target_col,
        "elapsed_seconds": elapsed_seconds,
        "pipeline_summary": pipeline_summary,
        "eda_charts":      eda_charts,
        "dataset_stats":   dataset_stats,
        "downloads":       downloads,
    }

    try:
        with open("outputs/results_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
    except Exception:
        pass  # best-effort write; caller can verify separately

    return manifest
