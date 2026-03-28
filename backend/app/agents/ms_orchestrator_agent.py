"""
app/agents/ms_orchestrator_agent.py

Model Selection Orchestrator — confirms best model from MT phase and
sets final_model / final_model_name in sandbox for Hyperparameter Tuning.

Architecture:
  ┌───────────────────────────────────────────────────────────────┐
  │            Model Selection Orchestrator                       │
  │   3-phase pipeline (Analyze → Strategize → Execute)          │
  └───────────────────────────────────────────────────────────────┘

Input:  mt_results dict from run_mt_orchestrator()
Output: Structured JSON with final_model_name, sandbox_ready flags,
        and next_phase = "Hyperparameter Tuning"
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from google.adk.agents import Agent

from app.agents.model_selection_agent.analyzer_agent import (
    model_selection_analyzer_agent,
    run_model_selection_analyzer,
)
from app.agents.model_selection_agent.strategist_agent import (
    model_selection_strategist_agent,
    run_model_selection_strategist,
)
from app.agents.model_selection_agent.executor_agent import (
    ms_executor_agent,
    run_ms_executor,
)
from app.tools.executor_tools import execute_python


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator system prompt
# ─────────────────────────────────────────────────────────────────────────────

_ORCHESTRATOR_PROMPT = """
You are the Model Selection Orchestrator.
You confirm the best model from the Model Training phase and prepare
sandbox globals for the Hyperparameter Tuning phase.

You have 3 sub-agents:
model_selection_analyzer_agent, model_selection_strategist_agent,
ms_executor_agent

You will receive:
model_comparison: dict of 3 model metrics from training
best_model_name: current best model name
task_type: CLASSIFICATION or REGRESSION
smote_applied: boolean
minority_class_percentage: float

YOUR EXACT WORKFLOW:

PHASE 1 — ANALYZE
Call model_selection_analyzer_agent with:
  X_train and X_test shapes
  y_train and y_test distributions
  feature names and count
  minority class percentage from training results
Wait for complete 6-section report.

PHASE 2 — STRATEGIZE
Call model_selection_strategist_agent with:
  Full analysis report from Phase 1
  task_type and minority_class_percentage
  feature_names
Wait for complete plan confirming best model and selection rationale.

PHASE 3 — EXECUTE
Call ms_executor_agent with:
  Full selection plan from Phase 2
  Reminder: set final_model and final_model_name in sandbox globals
  Reminder: save outputs/model_selection_summary.json
Wait for confirmation that final_model and final_model_name are set.

PHASE 4 — REPORT
Return structured JSON:

{
  "status": "success",
  "final_model_name": "XGBoost",
  "task_type": "CLASSIFICATION",
  "smote_applied": true,
  "primary_metric": "f1",
  "model_comparison": {
    "Random Forest":       {"accuracy": 0.91, "f1": 0.90},
    "Logistic Regression": {"accuracy": 0.87, "f1": 0.86},
    "XGBoost":             {"accuracy": 0.94, "f1": 0.93}
  },
  "winner_metrics": {"accuracy": 0.94, "f1": 0.93},
  "output_files": {
    "model_selection_summary": "outputs/model_selection_summary.json"
  },
  "sandbox_ready": {
    "final_model": true,
    "final_model_name": true
  },
  "next_phase": "Hyperparameter Tuning"
}

STRICT RULES:
Never skip a phase
Always pass complete output between phases
final_model and final_model_name MUST be set in sandbox before reporting success
"""


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Agent definition
# ─────────────────────────────────────────────────────────────────────────────

ms_orchestrator_agent = Agent(
    name="ms_orchestrator_agent",
    model="gemini-2.0-flash",
    description=(
        "Manages the model-selection 3-phase pipeline. Runs analysis, strategy, "
        "and execution in strict order. Sets final_model and final_model_name "
        "in sandbox and returns structured report for the HT phase."
    ),
    instruction=_ORCHESTRATOR_PROMPT,
    tools=[],
    sub_agents=[
        model_selection_analyzer_agent,
        model_selection_strategist_agent,
        ms_executor_agent,
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            return list(value.tolist())
        except Exception:
            pass
    return [value]


def _shape(value: Any) -> list[int]:
    if hasattr(value, "shape"):
        s = getattr(value, "shape")
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            return [int(s[0]), int(s[1])]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            pass
    return [0, 0]


def _sandbox_snapshot() -> dict[str, bool]:
    """Check that final_model and final_model_name are set in sandbox."""
    code = """
import json
snap = {
    "final_model":      "final_model" in globals(),
    "final_model_name": "final_model_name" in globals(),
}
print(json.dumps(snap))
"""
    out = execute_python(code).strip()
    if out.startswith("ERROR:"):
        return {"final_model": False, "final_model_name": False}
    try:
        return json.loads(out)
    except Exception:
        return {"final_model": False, "final_model_name": False}


def _read_final_model_name_from_sandbox() -> str:
    """Read final_model_name directly from sandbox after execution."""
    code = """
name = globals().get("final_model_name", "")
print(str(name))
"""
    out = execute_python(code).strip()
    return out if out and not out.startswith("ERROR:") else ""


def _read_json_file(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _error_result(
    best_model_name: str,
    winner_metrics: dict[str, Any],
    errors: list[str],
    log_lines: list[str],
    start: float,
) -> dict[str, Any]:
    return {
        "status": "error",
        "final_model_name": best_model_name,
        "task_type": "CLASSIFICATION",
        "smote_applied": False,
        "primary_metric": "f1",
        "model_comparison": {},
        "winner_metrics": winner_metrics,
        "output_files": {
            "model_selection_summary": "outputs/model_selection_summary.json"
        },
        "sandbox_ready": {
            "final_model": False,
            "final_model_name": False,
        },
        "next_phase": "Hyperparameter Tuning",
        "errors": errors,
        "pipeline_log": "\n".join(log_lines),
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_ms_orchestrator(
    mt_results: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the 3-phase model selection pipeline.

    Args:
        mt_results: Output dict from run_mt_orchestrator(). Must contain:
                    best_model, model_comparison, winner_metrics,
                    task_type, smote_applied, train_shape, test_shape.

    Returns:
        Structured JSON confirming final_model_name, sandbox_ready flags,
        and next_phase = "Hyperparameter Tuning".
        Compatible as input to run_ht_orchestrator(training_results=...).
    """
    start = time.monotonic()
    errors: list[str] = []
    log_lines: list[str] = []

    def _log(msg: str) -> None:
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # ── Extract MT inputs ─────────────────────────────────────────────────
    best_model_name: str = str(mt_results.get("best_model", ""))
    task_type: str = str(mt_results.get("task_type", "CLASSIFICATION")).upper()
    smote_applied: bool = bool(mt_results.get("smote_applied", False))
    model_comparison: dict[str, Any] = mt_results.get("model_comparison", {})
    winner_metrics: dict[str, Any] = mt_results.get("winner_metrics", {})

    train_shape: list[int] = _shape(mt_results.get("train_shape", [0, 0]))
    test_shape: list[int] = _shape(mt_results.get("test_shape", [0, 0]))

    # Extract feature names from strategy output if available
    strategy = mt_results.get("strategy", {})
    feature_names: list[str] = _to_list(strategy.get("feature_names", []))
    feature_count: int = int(
        strategy.get("feature_count", train_shape[1] if train_shape else 0)
    )
    target_col: str = str(strategy.get("target_col", "target"))
    target_dtype: str = ""

    # Build class distribution from training_results if available
    training_results: dict[str, Any] = mt_results.get("training_results", {})
    y_train_distribution: dict[str, Any] = training_results.get(
        "y_train_distribution", {}
    )
    y_test_distribution: dict[str, Any] = training_results.get(
        "y_test_distribution", {}
    )

    # Infer minority_class_percentage from model_comparison or training_results
    minority_class_percentage: float = float(
        mt_results.get("strategy", {}).get("minority_class_percentage", 50.0)
        if isinstance(mt_results.get("strategy"), dict)
        else 50.0
    )
    # Use smote_applied as a strong signal: if SMOTE ran, minority < 20%
    if smote_applied and minority_class_percentage >= 20.0:
        minority_class_percentage = 15.0

    imbalanced = minority_class_percentage < 20.0
    primary_metric = "f1" if imbalanced else "accuracy"

    # ── Phase 1: Analyze ─────────────────────────────────────────────────
    _log("Phase 1 start — Model Selection Analyzer.")
    analyzer_report = ""
    try:
        analyzer_report = await run_model_selection_analyzer(
            X_train_shape=train_shape,
            X_test_shape=test_shape,
            y_train_distribution=y_train_distribution,
            y_test_distribution=y_test_distribution,
            feature_count=feature_count,
            feature_names=feature_names,
            target_col=target_col,
            target_dtype=target_dtype,
            class_counts={k: v for k, v in y_train_distribution.items()},
            class_percentages={},
            minority_class_percentage=minority_class_percentage,
        )
        if not analyzer_report:
            raise RuntimeError("Empty analyzer report.")
        _log("Phase 1 complete.")
    except Exception as exc:
        errors.append(f"MS Analyzer failed: {exc}")
        return _error_result(best_model_name, winner_metrics, errors, log_lines, start)

    # ── Phase 2: Strategize ──────────────────────────────────────────────
    _log("Phase 2 start — Model Selection Strategist.")
    strategist_output: dict[str, Any] = {}
    try:
        strategist_output = await run_model_selection_strategist(
            analyzer_report=analyzer_report,
            target_col=target_col,
            feature_names=feature_names,
            task_type=task_type,
            minority_class_percentage=minority_class_percentage,
        )
        if not isinstance(strategist_output, dict) or not strategist_output.get(
            "training_plan"
        ):
            raise RuntimeError("Strategist output missing training_plan.")
        _log("Phase 2 complete.")
    except Exception as exc:
        errors.append(f"MS Strategist failed: {exc}")
        return _error_result(best_model_name, winner_metrics, errors, log_lines, start)

    # ── Phase 3: Execute ─────────────────────────────────────────────────
    _log("Phase 3 start — MS Executor.")
    executor_output: dict[str, Any] = {}
    try:
        executor_output = await run_ms_executor(strategist_output)
        if not isinstance(executor_output, dict):
            raise RuntimeError("Invalid executor output.")
        _log("Phase 3 complete.")
    except Exception as exc:
        errors.append(f"MS Executor failed: {exc}")
        executor_output = {}

    # ── Phase 4: Report ──────────────────────────────────────────────────
    sandbox = _sandbox_snapshot()
    final_model_name = _read_final_model_name_from_sandbox() or best_model_name

    selection_summary = _read_json_file("outputs/model_selection_summary.json")
    summary_exists = os.path.isfile("outputs/model_selection_summary.json")

    status = (
        "success"
        if sandbox.get("final_model") and sandbox.get("final_model_name")
        else "error"
    )
    if not sandbox.get("final_model"):
        errors.append("final_model not found in sandbox after execution.")
    if not sandbox.get("final_model_name"):
        errors.append("final_model_name not found in sandbox after execution.")

    return {
        "status": status,
        # MT-compatible keys so output can also feed run_ht_orchestrator
        "best_model": final_model_name,
        "best_model_name": final_model_name,
        "final_model_name": final_model_name,
        "task_type": task_type,
        "smote_applied": smote_applied,
        "primary_metric": primary_metric,
        "model_comparison": model_comparison,
        "winner_metrics": winner_metrics,
        "output_files": {
            "model_selection_summary": "outputs/model_selection_summary.json",
        },
        "sandbox_ready": {
            "final_model": bool(sandbox.get("final_model", False)),
            "final_model_name": bool(sandbox.get("final_model_name", False)),
        },
        # Pass through full MT outputs so HT orchestrator has all training info
        "all_results": training_results.get("all_results", {}),
        "training_results": mt_results,
        "selection_summary": selection_summary,
        "next_phase": "Hyperparameter Tuning",
        "analysis": analyzer_report,
        "strategy": strategist_output,
        "execution": executor_output,
        "errors": errors,
        "pipeline_log": "\n".join(log_lines),
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }
