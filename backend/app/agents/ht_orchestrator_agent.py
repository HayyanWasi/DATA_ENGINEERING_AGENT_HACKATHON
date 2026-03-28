"""
app/agents/ht_orchestrator_agent.py

Hyperparameter Tuning Orchestrator — top-level manager of the HT pipeline.

Architecture:
  ┌───────────────────────────────────────────────────────────────┐
  │           Hyperparameter Tuning Orchestrator                  │
  │     4-phase pipeline (Analyze → Strategize → Execute → Report)│
  └───────────────────────────────────────────────────────────────┘

Input:  training_results dict from the Model Training phase.
Output: Structured JSON with tuned model name, best params, scores, files.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from google.adk.agents import Agent

from app.agents.hyperparameter_tuning_agent.analyzer_agent import (
    ht_analyzer_agent,
    run_ht_analyzer,
)
from app.agents.hyperparameter_tuning_agent.strategist_agent import (
    ht_strategist_agent,
    run_ht_strategist,
)
from app.agents.hyperparameter_tuning_agent.executor_agent import (
    ht_executor_agent,
    run_ht_executor,
)
from app.tools.executor_tools import execute_python


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator system prompt
# ─────────────────────────────────────────────────────────────────────────────

_ORCHESTRATOR_PROMPT = """
You are the Hyperparameter Tuning Orchestrator.
You manage a 3-agent pipeline to find optimal hyperparameters for the
best model from the Model Training phase.

You have 3 sub-agents:
ht_analyzer_agent, ht_strategist_agent, ht_executor_agent

You will receive:
training_results from the Model Training phase containing:
best_model, winner_metrics, model_comparison, smote_applied,
train_shape, task_type.

YOUR EXACT WORKFLOW:

PHASE 1 — ANALYZE
Call ht_analyzer_agent with:
best_model_name, best_model_metrics, model_comparison,
current_hyperparameters, X_train_bal_shape, smote_applied, task_type.
Wait for complete 4-section tuning readiness report before Phase 2.

PHASE 2 — STRATEGIZE
Call ht_strategist_agent with:
Full analyzer report from Phase 1.
best_model_name, current_hyperparameters, cv_folds (from report),
primary_metric, diminishing_returns flag, overfit_risk flag.
Wait for complete strategy with valid param_grid JSON before Phase 3.

PHASE 3 — EXECUTE
Call ht_executor_agent with:
Full strategist output including param_grid, cv_folds, scoring,
best_model_name, total_combinations, baseline_score.
Wait for confirmation that tuned_model.pkl and tuning_results.json are saved.

PHASE 4 — REPORT
Return structured JSON:

{
  "status": "success",
  "best_model_name": "XGBoost",
  "best_params": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 6},
  "baseline_score": 0.93,
  "tuned_score": 0.95,
  "improvement_delta": 0.02,
  "cv_folds": 5,
  "total_combinations": 27,
  "output_files": {
    "tuned_model": "outputs/tuned_model.pkl",
    "tuning_results": "outputs/tuning_results.json"
  },
  "sandbox_ready": {
    "tuned_model": true,
    "best_params": true,
    "tuned_score": true
  },
  "next_phase": "Model Evaluation"
}

STRICT RULES:
Never skip a phase.
Never call Phase 3 before Phase 2 param_grid is confirmed valid.
Always pass complete output between phases.
Evaluation must always be on X_test — never training data.
Both output files must exist before reporting success.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Agent definition
# ─────────────────────────────────────────────────────────────────────────────

ht_orchestrator_agent = Agent(
    name="ht_orchestrator_agent",
    model="gemini-2.0-flash",
    description=(
        "Manages the hyperparameter tuning 3-agent pipeline. Runs analysis, "
        "strategy, and GridSearchCV execution in strict order, then returns "
        "tuned model metadata and improvement delta."
    ),
    instruction=_ORCHESTRATOR_PROMPT,
    tools=[],
    sub_agents=[
        ht_analyzer_agent,
        ht_strategist_agent,
        ht_executor_agent,
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


def _cv_folds_from_n(n: int) -> int:
    """Apply size-based CV rule."""
    if n < 200:
        return 3
    return 5


def _dataset_flag(n: int) -> str:
    if n < 200:
        return "Small"
    if n <= 1000:
        return "Medium"
    return "Large"


def _default_hyperparameters(model_name: str) -> dict[str, Any]:
    """Return the fixed hyperparameters used in the training phase."""
    name = str(model_name).lower()
    if "random forest" in name or "rf" in name:
        return {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 2,
            "random_state": 42,
            "class_weight": "balanced",
        }
    if "logistic" in name or "lr" in name:
        return {
            "C": 1.0,
            "max_iter": 2000,
            "solver": "liblinear",
            "random_state": 42,
            "class_weight": "balanced",
        }
    # XGBoost
    return {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42,
        "eval_metric": "logloss",
    }


def _sandbox_snapshot() -> dict[str, bool]:
    code = """
import json
snap = {
    "tuned_model": "tuned_model" in globals(),
    "best_params": "best_params" in globals(),
    "tuned_score": "tuned_score" in globals(),
}
print(json.dumps(snap))
"""
    out = execute_python(code).strip()
    if out.startswith("ERROR:"):
        return {"tuned_model": False, "best_params": False, "tuned_score": False}
    try:
        return json.loads(out)
    except Exception:
        return {"tuned_model": False, "best_params": False, "tuned_score": False}


# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_ht_orchestrator(
    training_results: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the 3-phase hyperparameter tuning pipeline.

    Args:
        training_results: Output dict from run_mt_orchestrator() containing
                          best_model, winner_metrics, model_comparison,
                          smote_applied, train_shape, task_type.

    Returns:
        Structured JSON result for the downstream Model Evaluation phase.
    """
    start = time.monotonic()
    errors: list[str] = []
    log_lines: list[str] = []

    def _log(msg: str) -> None:
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # ── Extract inputs from training_results ──────────────────────────────
    best_model_name: str = str(training_results.get("best_model", ""))
    winner_metrics: dict[str, Any] = training_results.get("winner_metrics", {})
    model_comparison: dict[str, Any] = training_results.get("model_comparison", {})
    smote_applied: bool = bool(training_results.get("smote_applied", False))
    task_type: str = str(training_results.get("task_type", "CLASSIFICATION")).upper()
    train_shape: list[int] = training_results.get("train_shape", [0, 0]) or [0, 0]

    # N is the first dimension of train_shape
    n_rows = int(train_shape[0]) if len(train_shape) >= 1 else 0

    cv_folds = _cv_folds_from_n(n_rows)
    dataset_flag = _dataset_flag(n_rows)
    primary_metric = "f1" if smote_applied else "accuracy"
    baseline_score = float(
        winner_metrics.get("f1", winner_metrics.get("accuracy", 0.0))
        if smote_applied
        else winner_metrics.get("accuracy", winner_metrics.get("f1", 0.0))
    )
    diminishing_returns = baseline_score > 0.95
    current_hp = _default_hyperparameters(best_model_name)

    # Build X_train_bal_shape from train_shape
    x_train_bal_shape = list(train_shape) if len(train_shape) == 2 else [n_rows, 0]

    # Determine overfit risk from default hyperparameters
    name_lower = best_model_name.lower()
    if "random forest" in name_lower or "rf" in name_lower:
        overfit_risk = "High" if int(current_hp.get("max_depth", 0)) > 10 else "Low"
    elif "logistic" in name_lower or "lr" in name_lower:
        overfit_risk = "High" if float(current_hp.get("C", 1.0)) > 5 else "Low"
    else:
        overfit_risk = "High" if int(current_hp.get("max_depth", 0)) > 10 else "Low"

    # ── Phase 1: Analyze ─────────────────────────────────────────────────
    _log("Phase 1 start — HT Analyzer.")
    analyzer_report = ""
    try:
        analyzer_report = await run_ht_analyzer(
            best_model_name=best_model_name,
            best_model_metrics=winner_metrics,
            model_comparison=model_comparison,
            current_hyperparameters=current_hp,
            X_train_bal_shape=x_train_bal_shape,
            smote_applied=smote_applied,
            task_type=task_type,
        )
        if not analyzer_report:
            raise RuntimeError("Empty analyzer report.")
        _log("Phase 1 complete.")
    except Exception as exc:
        errors.append(f"HT Analyzer failed: {exc}")
        return _error_result(
            best_model_name, baseline_score, errors, log_lines, start
        )

    # ── Phase 2: Strategize ──────────────────────────────────────────────
    _log("Phase 2 start — HT Strategist.")
    strategist_output: dict[str, Any] = {}
    try:
        strategist_output = await run_ht_strategist(
            analyzer_report=analyzer_report,
            best_model_name=best_model_name,
            current_hyperparameters=current_hp,
            cv_folds=cv_folds,
            primary_metric=primary_metric,
            diminishing_returns=diminishing_returns,
            overfit_risk=overfit_risk,
        )
        if not strategist_output.get("param_grid"):
            raise RuntimeError("Strategist output missing param_grid.")
        _log("Phase 2 complete.")
    except Exception as exc:
        errors.append(f"HT Strategist failed: {exc}")
        return _error_result(
            best_model_name, baseline_score, errors, log_lines, start,
            analysis=analyzer_report,
        )

    # Inject baseline_score so executor can compute improvement_delta
    strategist_output["baseline_score"] = baseline_score

    # ── Phase 3: Execute ─────────────────────────────────────────────────
    _log("Phase 3 start — HT Executor.")
    executor_output: dict[str, Any] = {}
    try:
        executor_output = await run_ht_executor(strategist_output)
        if not isinstance(executor_output, dict):
            raise RuntimeError("Invalid executor output.")
        _log("Phase 3 complete.")
    except Exception as exc:
        errors.append(f"HT Executor failed: {exc}")
        executor_output = {}

    # ── Phase 4: Report ──────────────────────────────────────────────────
    tuning_results = _read_json_file("outputs/tuning_results.json")

    best_params = tuning_results.get("best_params", {})
    tuned_score = float(tuning_results.get("tuned_score", baseline_score))
    improvement_delta = round(tuned_score - baseline_score, 6)
    tuned_metrics = tuning_results.get("tuned_metrics", {})

    output_files = {
        "tuned_model": "outputs/tuned_model.pkl",
        "tuning_results": "outputs/tuning_results.json",
    }
    files_exist = all(os.path.isfile(p) for p in output_files.values())

    sandbox = _sandbox_snapshot()
    status = "success" if files_exist and not errors else "error"

    if not files_exist:
        errors.append("One or more tuning output files are missing.")
    if not all(sandbox.values()):
        status = "error"
        errors.append("Required sandbox globals missing after tuning execution.")

    return {
        "status": status,
        "best_model_name": best_model_name,
        "best_params": best_params,
        "baseline_score": baseline_score,
        "tuned_score": tuned_score,
        "improvement_delta": improvement_delta,
        "cv_folds": cv_folds,
        "dataset_flag": dataset_flag,
        "total_combinations": strategist_output.get("total_combinations", 27),
        "primary_metric": primary_metric,
        "diminishing_returns": diminishing_returns,
        "overfit_risk": overfit_risk,
        "tuned_metrics": tuned_metrics,
        "output_files": output_files,
        "sandbox_ready": {
            "tuned_model": bool(sandbox.get("tuned_model", False)),
            "best_params": bool(sandbox.get("best_params", False)),
            "tuned_score": bool(sandbox.get("tuned_score", False)),
        },
        "next_phase": "Model Evaluation",
        "analysis": analyzer_report,
        "strategy": strategist_output,
        "implementation": executor_output,
        "tuning_results": tuning_results,
        "errors": errors,
        "pipeline_log": "\n".join(log_lines),
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }


def _error_result(
    best_model_name: str,
    baseline_score: float,
    errors: list[str],
    log_lines: list[str],
    start: float,
    analysis: str = "",
) -> dict[str, Any]:
    return {
        "status": "error",
        "best_model_name": best_model_name,
        "best_params": {},
        "baseline_score": baseline_score,
        "tuned_score": baseline_score,
        "improvement_delta": 0.0,
        "cv_folds": 5,
        "dataset_flag": "Unknown",
        "total_combinations": 27,
        "primary_metric": "f1",
        "diminishing_returns": False,
        "overfit_risk": "Low",
        "tuned_metrics": {},
        "output_files": {
            "tuned_model": "outputs/tuned_model.pkl",
            "tuning_results": "outputs/tuning_results.json",
        },
        "sandbox_ready": {
            "tuned_model": False,
            "best_params": False,
            "tuned_score": False,
        },
        "next_phase": "Model Evaluation",
        "analysis": analysis,
        "errors": errors,
        "pipeline_log": "\n".join(log_lines),
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }
