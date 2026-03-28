"""
app/tools/fo_tools.py

Final Output helper functions — plain Python, NOT FunctionTools.

Called by the Final Output stage orchestrator to collect pipeline
outputs, assemble summary data, and build the results manifest for
downstream consumption.

Functions:
    collect_output_files()         — scan outputs/ and charts/, return categorised file lists
    load_json_safe(filepath)       — load JSON, return {} on any error
    get_pipeline_summary(sandbox)  — assemble summary from sandbox variables and JSON files
    save_model_as_joblib()         — joblib-dump _SANDBOX["final_model"] to outputs/
    build_results_manifest(...)    — build results manifest dict with dynamic chart discovery
"""

from __future__ import annotations

import json
import os
from typing import Any

from app.tools.executor_tools import _SANDBOX


# ─────────────────────────────────────────────────────────────────────────────
# 1. collect_output_files
# ─────────────────────────────────────────────────────────────────────────────

def collect_output_files() -> dict[str, list[str]]:
    """
    Scan outputs/ and charts/ directories and return categorised file lists.

    Returns:
        {
            "model_files":        [...],  # .pkl, .joblib files from outputs/
            "data_files":         [...],  # .csv files from outputs/
            "evaluation_charts":  [...],  # .png files from outputs/
            "eda_charts":         [...],  # .png files from charts/
            "json_reports":       [...],  # .json files from outputs/ and charts/
        }
    Missing directories are handled safely — their categories return empty lists.
    """
    model_files: list[str] = []
    data_files: list[str] = []
    evaluation_charts: list[str] = []
    eda_charts: list[str] = []
    json_reports: list[str] = []

    # Scan outputs/
    if os.path.isdir("outputs"):
        for fname in sorted(os.listdir("outputs")):
            lower = fname.lower()
            fpath = os.path.join("outputs", fname)
            if not os.path.isfile(fpath):
                continue
            if lower.endswith(".pkl") or lower.endswith(".joblib"):
                model_files.append(fpath)
            elif lower.endswith(".csv"):
                data_files.append(fpath)
            elif lower.endswith(".png"):
                evaluation_charts.append(fpath)
            elif lower.endswith(".json"):
                json_reports.append(fpath)

    # Scan charts/
    if os.path.isdir("charts"):
        for fname in sorted(os.listdir("charts")):
            lower = fname.lower()
            fpath = os.path.join("charts", fname)
            if not os.path.isfile(fpath):
                continue
            if lower.endswith(".png"):
                eda_charts.append(fpath)
            elif lower.endswith(".json"):
                json_reports.append(fpath)

    return {
        "model_files":       model_files,
        "data_files":        data_files,
        "evaluation_charts": evaluation_charts,
        "eda_charts":        eda_charts,
        "json_reports":      json_reports,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. load_json_safe
# ─────────────────────────────────────────────────────────────────────────────

def load_json_safe(filepath: str) -> dict[str, Any]:
    """Load a JSON file; return {} on any error (missing file, bad JSON, etc.)."""
    if not os.path.isfile(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. get_pipeline_summary
# ─────────────────────────────────────────────────────────────────────────────

def get_pipeline_summary(sandbox: dict[str, Any]) -> dict[str, Any]:
    """
    Assemble a structured pipeline summary from sandbox variables and JSON reports.

    Reads sandbox variables (target_col, df/df_raw shapes, engineered_features)
    and the following JSON files via load_json_safe:
      - outputs/evaluation_summary.json
      - outputs/training_results.json
      - outputs/model_selection_summary.json
      - outputs/tuning_results.json
      - outputs/balance_report.json

    Args:
        sandbox: The shared execution sandbox dict (e.g. _SANDBOX from executor_tools).

    Returns:
        {
            "dataset_info": {
                "target_col":           str,
                "original_shape":       [rows, cols],
                "cleaned_shape":        [rows, cols],
                "engineered_features":  int,
            },
            "model_info": {
                "final_model_name":   str,
                "tuning_applied":     bool,
                "smote_applied":      bool,
                "performance_rating": str,
            },
            "metrics":          dict,   # accuracy, f1, precision, recall, roc_auc
            "model_comparison": list,   # per-model rows sorted by accuracy desc
            "balance_info":     dict,   # class balance / SMOTE report
            "tuning_info":      dict,   # best params and tuning metadata
        }
    """
    # ── Load JSON stage reports ───────────────────────────────────────────────
    eval_summary  = load_json_safe("outputs/evaluation_summary.json")
    train_results = load_json_safe("outputs/training_results.json")
    model_sel     = load_json_safe("outputs/model_selection_summary.json")
    tuning        = load_json_safe("outputs/tuning_results.json")
    balance       = load_json_safe("outputs/balance_report.json")

    # ── Dataset info from sandbox ─────────────────────────────────────────────
    target_col = str(sandbox.get("target_col", ""))

    df_raw = sandbox.get("df_raw")
    original_shape: list[int] = list(df_raw.shape) if df_raw is not None else []

    df = sandbox.get("df")
    cleaned_shape: list[int] = list(df.shape) if df is not None else []

    # engineered_features: prefer explicit sandbox value, fall back to column delta
    eng_features = sandbox.get("engineered_features")
    if eng_features is None:
        if original_shape and cleaned_shape:
            eng_features = max(0, cleaned_shape[1] - original_shape[1])
        else:
            eng_features = 0

    # ── Model info ────────────────────────────────────────────────────────────
    final_model_name = str(
        eval_summary.get("final_model_name")
        or model_sel.get("best_model")
        or train_results.get("best_model")
        or sandbox.get("final_model_name", "")
    )

    tuning_applied = bool(
        tuning.get("best_params") or tuning.get("tuning_applied", False)
    )

    smote_applied = bool(balance.get("smote_applied", False))

    performance_rating = str(eval_summary.get("performance_rating", "Unknown"))

    # ── Metrics ───────────────────────────────────────────────────────────────
    # Prefer the "final_metrics" sub-dict; fall back to top-level eval_summary
    # keys (some ME implementations write metrics at the top level).
    _fm = eval_summary.get("final_metrics")
    raw_metrics: dict[str, Any] = dict(_fm) if isinstance(_fm, dict) else dict(eval_summary)
    metrics: dict[str, Any] = {
        "accuracy":  float(raw_metrics.get("accuracy", 0.0)),
        "f1":        float(raw_metrics.get("f1", raw_metrics.get("f1_score", raw_metrics.get("f1_weighted", 0.0)))),
        "precision": float(raw_metrics.get("precision", 0.0)),
        "recall":    float(raw_metrics.get("recall", 0.0)),
        "roc_auc":   float(raw_metrics.get("roc_auc", raw_metrics.get("auc", 0.0))),
    }

    # ── Model comparison ──────────────────────────────────────────────────────
    raw_models = (
        train_results.get("all_models")
        or train_results.get("model_comparison")
        or model_sel.get("model_comparison")
        or []
    )
    model_comparison: list[dict[str, Any]] = []
    for m in raw_models:
        if not isinstance(m, dict):
            continue
        name = str(m.get("model", m.get("name", "")))
        acc  = float(m.get("accuracy", m.get("test_accuracy", 0.0)))
        f1   = float(m.get("f1", m.get("f1_score", m.get("f1_weighted", 0.0))))
        if name == final_model_name:
            status = "Best"
        elif acc >= 0.85:
            status = "Good"
        else:
            status = "Fair"
        model_comparison.append({"model": name, "accuracy": acc, "f1": f1, "status": status})
    model_comparison.sort(key=lambda x: x["accuracy"], reverse=True)

    return {
        "dataset_info": {
            "target_col":          target_col,
            "original_shape":      original_shape,
            "cleaned_shape":       cleaned_shape,
            "engineered_features": int(eng_features),
        },
        "model_info": {
            "final_model_name":   final_model_name,
            "tuning_applied":     tuning_applied,
            "smote_applied":      smote_applied,
            "performance_rating": performance_rating,
        },
        "metrics":          metrics,
        "model_comparison": model_comparison,
        "balance_info":     balance,
        "tuning_info":      tuning,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. save_model_as_joblib
# ─────────────────────────────────────────────────────────────────────────────

def save_model_as_joblib() -> str:
    """
    Dump _SANDBOX["final_model"] to outputs/final_model.joblib using joblib.

    Returns:
        str — the file path written ("outputs/final_model.joblib").

    Raises:
        Exception: if final_model is None or absent in the sandbox.
    """
    import joblib

    model = _SANDBOX.get("final_model")
    if model is None:
        raise Exception(
            "final_model not found in sandbox. "
            "Ensure the model training phase completed successfully."
        )

    os.makedirs("outputs", exist_ok=True)
    path = "outputs/final_model.joblib"
    joblib.dump(model, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 5. build_results_manifest
# ─────────────────────────────────────────────────────────────────────────────

def build_results_manifest(
    dataset_id: str,
    pipeline_summary: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the results manifest dict with dynamic chart discovery.

    Scans:
      - charts/  for EDA chart PNGs
      - outputs/ for evaluation chart PNGs
    Charts not found on disk are silently skipped.

    Args:
        dataset_id:       Unique identifier for the dataset/run.
        pipeline_summary: Output of get_pipeline_summary().

    Returns:
        The manifest dict (also written to outputs/results_manifest.json
        on a best-effort basis).
    """
    os.makedirs("outputs", exist_ok=True)

    # ── Discover EDA charts ───────────────────────────────────────────────────
    eda_charts: list[str] = []
    if os.path.isdir("charts"):
        for fname in sorted(os.listdir("charts")):
            if fname.lower().endswith(".png") and os.path.isfile(os.path.join("charts", fname)):
                eda_charts.append(fname)

    # ── Discover evaluation charts ────────────────────────────────────────────
    evaluation_charts: list[str] = []
    if os.path.isdir("outputs"):
        for fname in sorted(os.listdir("outputs")):
            if fname.lower().endswith(".png") and os.path.isfile(os.path.join("outputs", fname)):
                evaluation_charts.append(fname)

    # ── Download links for files present on disk ──────────────────────────────
    download_candidates = {
        "model":        "outputs/final_model.joblib",
        "cleaned_data": "outputs/cleaned_data.csv",
        "evaluation":   "outputs/evaluation_summary.json",
    }
    downloads: list[dict[str, str]] = []
    for file_type, path in download_candidates.items():
        if os.path.isfile(path):
            downloads.append({
                "file_type": file_type,
                "filename":  os.path.basename(path),
                "url":       f"/api/download/{dataset_id}/{file_type}",
            })

    # ── Assemble manifest ─────────────────────────────────────────────────────
    manifest: dict[str, Any] = {
        "dataset_id":        dataset_id,
        "pipeline_summary":  pipeline_summary,
        "eda_charts":        eda_charts,
        "evaluation_charts": evaluation_charts,
        "downloads":         downloads,
    }

    try:
        with open("outputs/results_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
    except Exception:
        pass  # best-effort write; caller can verify with verify_output_saved()

    return manifest
