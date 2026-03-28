"""
app/agents/me_orchestrator_agent.py

Model Evaluation Orchestrator — top-level manager of the ME pipeline.

Architecture:
  ┌────────────────────────────────────────────────────────────────┐
  │             Model Evaluation Orchestrator                      │
  │   4-phase pipeline (Analyze → Strategize → Execute → Report)  │
  └────────────────────────────────────────────────────────────────┘

Input:  ht_results dict from Hyperparameter Tuning phase
        (or mt_results if tuning was skipped).
Output: Structured JSON with performance grade, metrics, plots list.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from google.adk.agents import Agent

from app.agents.model_evaluation_agent.analyzer_agent import (
    me_analyzer_agent,
    run_me_analyzer,
)
from app.agents.model_evaluation_agent.strategist_agent import (
    me_strategist_agent,
    run_me_strategist,
)
from app.agents.model_evaluation_agent.executor_agent import (
    me_executor_agent,
    run_me_executor,
)
from app.tools.executor_tools import execute_python


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator system prompt
# ─────────────────────────────────────────────────────────────────────────────

_ORCHESTRATOR_PROMPT = """
You are the Model Evaluation Orchestrator.
You manage a 3-agent pipeline to fully evaluate the final
trained model and generate all visualization outputs for
the Next.js UI and final PDF report.

You have 3 sub-agents:
me_analyzer_agent, me_strategist_agent, me_implementor_agent

You will receive:
Sandbox already contains:
  final_model, final_model_name, X_test, y_test,
  X_train_bal, all_results, tuning_applied,
  tuning_results, smote_applied, target_col
ht_result: output from run_ht_orchestrator()
mt_result: output from run_mt_orchestrator()

YOUR EXACT WORKFLOW:

PHASE 1 — ANALYZE
Call me_analyzer_agent with:
final_model_name from sandbox
final metrics from ht_result or mt_result
tuning_applied and tuned/original metrics
X_test shape and y_test distribution
smote_applied boolean
task_type inferred from y_test unique values
all_model_comparison from mt_result
Wait for complete 6-section report before Phase 2.

PHASE 2 — STRATEGIZE
Call me_strategist_agent with:
Full analysis report from Phase 1
final_model_name, task_type
smote_applied, class_count
y_test_distribution
visualization requirements from Phase 1
Wait for complete evaluation and visualization plan
covering all 7 sections before Phase 3.

PHASE 3 — IMPLEMENT
Call me_implementor_agent with:
Full evaluation plan from Phase 2
Reminder: evaluate on X_test only
Reminder: plt.close() after every chart
Reminder: evaluation_summary.json is mandatory
Reminder: update sandbox globals after evaluation
Wait for confirmation that all files are saved
and sandbox globals are updated.

PHASE 4 — REPORT
Return structured JSON:

{
  "status": "success",
  "final_model_name": "tuned_XGBoost",
  "task_type": "CLASSIFICATION",
  "performance_rating": "Excellent",
  "final_metrics": {
    "accuracy": 0.96,
    "f1": 0.95,
    "precision": 0.96,
    "recall": 0.95,
    "auc_score": 0.98
  },
  "tuning_applied": true,
  "smote_applied": true,
  "model_comparison": {
    "Random Forest":       {"accuracy": 0.91, "f1": 0.90},
    "Logistic Regression": {"accuracy": 0.87, "f1": 0.86},
    "XGBoost":             {"accuracy": 0.96, "f1": 0.95}
  },
  "output_files": {
    "confusion_matrix":        "outputs/confusion_matrix.png",
    "classification_report":   "outputs/classification_report.json",
    "roc_curve":               "outputs/roc_curve.png",
    "pr_curve":                "outputs/pr_curve.png",
    "feature_importance":      "outputs/feature_importance.png",
    "model_comparison":        "outputs/model_comparison.png",
    "evaluation_summary":      "outputs/evaluation_summary.json"
  },
  "sandbox_ready": {
    "evaluation_results": true,
    "y_pred": true,
    "auc_score": true,
    "performance_rating": true
  },
  "next_phase": "Final Outputs"
}

STRICT RULES:
Never skip a phase
Never call Phase 3 before Phase 2 is complete
Always pass complete output between phases
evaluation_summary.json must exist before reporting success
sandbox_ready must show all 4 keys as true
If any chart fails — continue pipeline, note in report
evaluation_summary.json failure = pipeline failure
Next phase Final Outputs reads evaluation_results from sandbox
"""


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Agent definition
# ─────────────────────────────────────────────────────────────────────────────

me_orchestrator_agent = Agent(
    name="me_orchestrator_agent",
    model="gemini-2.0-flash",
    description=(
        "Manages the 4-phase model evaluation pipeline (Analyze → Strategize → "
        "Implement → Report). Produces performance_rating, all visualization files, "
        "evaluation_summary.json, and sandbox globals for the Final Outputs phase."
    ),
    instruction=_ORCHESTRATOR_PROMPT,
    tools=[],
    sub_agents=[
        me_analyzer_agent,
        me_strategist_agent,
        me_executor_agent,   # registered as me_implementor_agent alias
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_json_file(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _grade(score: float) -> str:
    if score >= 0.9:
        return "Excellent"
    if score >= 0.8:
        return "Good"
    if score >= 0.7:
        return "Fair"
    return "Poor"


def _class_count_from_distribution(dist: dict[str, Any]) -> int:
    if not dist:
        return 2
    return len(dist)


def _minority_pct(dist: dict[str, Any]) -> float:
    if not dist:
        return 50.0
    total = sum(int(v) for v in dist.values())
    if total == 0:
        return 50.0
    return round(min(int(v) for v in dist.values()) / total * 100, 2)


def _sandbox_snapshot() -> dict[str, bool]:
    code = """
import json
snap = {
    "evaluation_results": "evaluation_results" in globals(),
    "y_pred":             "y_pred" in globals(),
    "auc_score":          "auc_score" in globals(),
    "performance_rating": "performance_rating" in globals(),
}
print(json.dumps(snap))
"""
    out = execute_python(code).strip()
    if out.startswith("ERROR:"):
        return {
            "evaluation_results": False,
            "y_pred": False,
            "auc_score": False,
            "performance_rating": False,
        }
    try:
        return json.loads(out)
    except Exception:
        return {
            "evaluation_results": False,
            "y_pred": False,
            "auc_score": False,
            "performance_rating": False,
        }


def _collect_plots_saved() -> list[str]:
    """Return list of all output files actually written to outputs/."""
    candidate = [
        "outputs/confusion_matrix.png",
        "outputs/roc_curve.png",
        "outputs/pr_curve.png",
        "outputs/feature_importance.png",
        "outputs/model_comparison.png",
        "outputs/classification_report.txt",
        "outputs/classification_report.json",
        "outputs/evaluation_summary.json",
    ]
    return [p for p in candidate if os.path.isfile(p)]


# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_me_orchestrator(
    ht_results: dict[str, Any],
    mt_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the 3-phase model evaluation pipeline.

    Args:
        ht_results:  Output dict from run_ht_orchestrator(). If tuning was
                     skipped, pass the mt_results here with tuning_applied=False.
        mt_results:  Optional output from run_mt_orchestrator(), used to
                     supplement model_comparison and original metrics.

    Returns:
        Structured JSON result marking the end of the ML pipeline.
    """
    start = time.monotonic()
    errors: list[str] = []
    log_lines: list[str] = []

    def _log(msg: str) -> None:
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # ── Extract inputs ────────────────────────────────────────────────────
    tuning_applied: bool = bool(ht_results.get("status") == "success"
                                and ht_results.get("best_params"))

    # Final model name: tuned if tuning succeeded, else original best
    final_model_name: str = str(ht_results.get("best_model_name", ""))
    if not final_model_name and mt_results:
        final_model_name = str(mt_results.get("best_model", ""))

    # Metrics: prefer tuned_metrics, fall back to mt winner_metrics
    tuned_metrics: dict[str, Any] = ht_results.get("tuned_metrics", {})
    original_metrics: dict[str, Any] = {}
    if mt_results:
        original_metrics = mt_results.get("winner_metrics", {})

    final_model_metrics: dict[str, Any] = tuned_metrics if tuned_metrics else original_metrics

    best_params: dict[str, Any] = ht_results.get("best_params", {})
    smote_applied: bool = bool(
        (mt_results or {}).get("smote_applied", ht_results.get("smote_applied", False))
    )
    task_type: str = str(
        (mt_results or {}).get("task_type", "CLASSIFICATION")
    ).upper()

    # Model comparison from training results
    all_model_comparison: dict[str, Any] = {}
    if mt_results:
        all_model_comparison = mt_results.get("model_comparison", {})

    # Test set metadata from sandbox
    test_meta = _read_test_metadata_from_sandbox()
    X_test_shape: list[int] = test_meta.get("X_test_shape", [0, 0])
    y_test_distribution: dict[str, Any] = test_meta.get("y_test_distribution", {})
    training_shape: list[int] = (mt_results or {}).get("train_shape", [0, 0]) or [0, 0]

    class_count = _class_count_from_distribution(y_test_distribution)

    # ── Phase 1: Analyze ─────────────────────────────────────────────────
    _log("Phase 1 start — ME Analyzer.")
    analyzer_report = ""
    try:
        analyzer_report = await run_me_analyzer(
            final_model_name=final_model_name,
            final_model_metrics=final_model_metrics,
            tuning_applied=tuning_applied,
            tuned_metrics=tuned_metrics if tuned_metrics else None,
            original_metrics=original_metrics,
            X_test_shape=X_test_shape,
            y_test_distribution=y_test_distribution,
            smote_applied=smote_applied,
            task_type=task_type,
            best_params=best_params,
            all_model_comparison=all_model_comparison,
            training_shape=training_shape,
        )
        if not analyzer_report:
            raise RuntimeError("Empty analyzer report.")
        _log("Phase 1 complete.")
    except Exception as exc:
        errors.append(f"ME Analyzer failed: {exc}")
        return _error_result(final_model_name, final_model_metrics, errors, log_lines, start)

    # ── Phase 2: Strategize ──────────────────────────────────────────────
    _log("Phase 2 start — ME Strategist.")
    strategist_output: dict[str, Any] = {}
    try:
        strategist_output = await run_me_strategist(
            analyzer_report=analyzer_report,
            final_model_name=final_model_name,
            task_type=task_type,
            smote_applied=smote_applied,
            class_count=class_count,
            y_test_distribution=y_test_distribution,
            tuning_applied=tuning_applied,
            visualization_requirements=analyzer_report,  # Section 5 is embedded
        )
        if not strategist_output.get("evaluation_plan"):
            raise RuntimeError("Strategist output missing evaluation_plan.")
        _log("Phase 2 complete.")
    except Exception as exc:
        errors.append(f"ME Strategist failed: {exc}")
        return _error_result(
            final_model_name, final_model_metrics, errors, log_lines, start,
            analysis=analyzer_report,
        )

    # Inject all_model_comparison so executor can build the comparison chart
    strategist_output["all_model_comparison"] = all_model_comparison

    # ── Phase 3: Execute ─────────────────────────────────────────────────
    _log("Phase 3 start — ME Executor.")
    executor_output: dict[str, Any] = {}
    try:
        executor_output = await run_me_executor(strategist_output)
        if not isinstance(executor_output, dict):
            raise RuntimeError("Invalid executor output.")
        _log("Phase 3 complete.")
    except Exception as exc:
        errors.append(f"ME Executor failed: {exc}")
        executor_output = {}

    # ── Phase 4: Report ──────────────────────────────────────────────────
    eval_results = _read_json_file("outputs/evaluation_summary.json")

    final_metrics = eval_results.get("final_metrics", final_model_metrics)

    # Support both field names written by the executor
    performance_rating = (
        eval_results.get("performance_rating")
        or eval_results.get("performance_grade")
        or ""
    )
    if not performance_rating:
        primary_score = float(
            final_metrics.get("f1", final_metrics.get("accuracy", 0.0))
            if smote_applied
            else final_metrics.get("accuracy", final_metrics.get("f1", 0.0))
        )
        performance_rating = _grade(primary_score)

    auc_score = eval_results.get("auc_score", None)
    plots_saved = _collect_plots_saved()
    eval_file_exists = os.path.isfile("outputs/evaluation_summary.json")

    sandbox = _sandbox_snapshot()
    status = "success" if eval_file_exists and not errors else "error"
    if not eval_file_exists:
        errors.append("outputs/evaluation_summary.json not saved.")
    if not all(sandbox.values()):
        status = "error"
        errors.append("Required sandbox globals missing after evaluation.")

    # Build final_metrics with auc_score included
    final_metrics_out = dict(final_metrics)
    if auc_score is not None:
        final_metrics_out["auc_score"] = auc_score

    return {
        "status": status,
        "final_model_name": final_model_name,
        "task_type": task_type,
        "performance_rating": performance_rating,
        "final_metrics": final_metrics_out,
        "tuning_applied": tuning_applied,
        "smote_applied": smote_applied,
        "improvement_delta": ht_results.get("improvement_delta", 0.0),
        "model_comparison": all_model_comparison,
        "plots_saved": plots_saved,
        "output_files": {
            "confusion_matrix":      "outputs/confusion_matrix.png",
            "classification_report": "outputs/classification_report.json",
            "roc_curve":             "outputs/roc_curve.png",
            "pr_curve":              "outputs/pr_curve.png",
            "feature_importance":    "outputs/feature_importance.png",
            "model_comparison":      "outputs/model_comparison.png",
            "evaluation_summary":    "outputs/evaluation_summary.json",
        },
        "sandbox_ready": {
            "evaluation_results": bool(sandbox.get("evaluation_results", False)),
            "y_pred":             bool(sandbox.get("y_pred", False)),
            "auc_score":          bool(sandbox.get("auc_score", False)),
            "performance_rating": bool(sandbox.get("performance_rating", False)),
        },
        "next_phase": "Final Outputs",
        "analysis": analyzer_report,
        "strategy": strategist_output,
        "implementation": executor_output,
        "evaluation_results": eval_results,
        "errors": errors,
        "pipeline_log": "\n".join(log_lines),
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }


def _read_test_metadata_from_sandbox() -> dict[str, Any]:
    """Read X_test shape and y_test distribution from sandbox."""
    code = """
import json
X = globals().get("X_test")
y = globals().get("y_test")
x_shape = list(X.shape) if X is not None and hasattr(X, "shape") else [0, 0]
if y is not None:
    import pandas as pd
    import numpy as np
    y_arr = y if hasattr(y, "__iter__") else []
    counts = {}
    for val in y_arr:
        k = str(val)
        counts[k] = counts.get(k, 0) + 1
else:
    counts = {}
print(json.dumps({"X_test_shape": x_shape, "y_test_distribution": counts}))
"""
    out = execute_python(code).strip()
    if out.startswith("ERROR:"):
        return {"X_test_shape": [0, 0], "y_test_distribution": {}}
    try:
        return json.loads(out)
    except Exception:
        return {"X_test_shape": [0, 0], "y_test_distribution": {}}


def _error_result(
    final_model_name: str,
    final_metrics: dict[str, Any],
    errors: list[str],
    log_lines: list[str],
    start: float,
    analysis: str = "",
) -> dict[str, Any]:
    return {
        "status": "error",
        "final_model_name": final_model_name,
        "task_type": "CLASSIFICATION",
        "performance_rating": "Unknown",
        "final_metrics": final_metrics,
        "tuning_applied": False,
        "smote_applied": False,
        "improvement_delta": 0.0,
        "model_comparison": {},
        "plots_saved": [],
        "output_files": {
            "confusion_matrix":      "outputs/confusion_matrix.png",
            "classification_report": "outputs/classification_report.json",
            "roc_curve":             "outputs/roc_curve.png",
            "pr_curve":              "outputs/pr_curve.png",
            "feature_importance":    "outputs/feature_importance.png",
            "model_comparison":      "outputs/model_comparison.png",
            "evaluation_summary":    "outputs/evaluation_summary.json",
        },
        "sandbox_ready": {
            "evaluation_results": False,
            "y_pred":             False,
            "auc_score":          False,
            "performance_rating": False,
        },
        "next_phase": "Final Outputs",
        "analysis": analysis,
        "errors": errors,
        "pipeline_log": "\n".join(log_lines),
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }
