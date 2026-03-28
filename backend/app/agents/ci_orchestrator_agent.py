"""
app/agents/ci_orchestrator_agent.py

Class Imbalance Orchestrator — top-level manager of the imbalance-handling
pipeline before model training.

Architecture:
  ┌───────────────────────────────────────────────────────────────┐
  │         Class Imbalance Orchestrator                        │
  │        4-phase pipeline (Analyze → Strategize → Implement)   │
  └──────────────┬──────────────────┬───────────────────────────┘
                 │                  │
                 ▼                  ▼
         CI Analyzer       CI Strategist       CI Implementor
         (distribution)    (plan selection)    (sandbox execution)
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.agents.ci.ci_analyzer_agent import ci_analyzer_agent, run_ci_analyzer
from app.agents.ci.ci_executor_agent import ci_implementor_agent, run_ci_implementor
from app.agents.ci.ci_strategist_agent import (
    ci_strategist_agent,
    run_ci_strategist,
)
from app.tools.ci_analysis_tools import analyze_class_imbalance
from app.tools.executor_tools import execute_python, verify_output_saved


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator-owned tools
# ─────────────────────────────────────────────────────────────────────────────

def validate_ci_input(
    target_col: str,
    task_type: str,
    total_train_samples: int,
) -> dict[str, Any]:
    """
    Validate inputs before CI pipeline starts.
    """
    errors: list[str] = []

    if not target_col or not target_col.strip():
        errors.append("target_col is empty or missing.")

    if total_train_samples < 0:
        errors.append("total_train_samples cannot be negative.")

    if task_type not in {"CLASSIFICATION", "REGRESSION"}:
        errors.append(f"task_type '{task_type}' must be CLASSIFICATION or REGRESSION.")

    valid = len(errors) == 0
    summary = (
        f"CI validation passed. target_col='{target_col}', task_type='{task_type}'."
        if valid
        else f"CI validation FAILED: {'; '.join(errors)}"
    )
    return {"valid": valid, "errors": errors, "summary": summary}


def ci_pipeline_status(
    stage: str,
    status: str,
    message: str,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    """
    Report the current CI pipeline stage for observability.
    """
    return {
        "stage": stage,
        "status": status,
        "message": message,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


_ORCHESTRATOR_TOOLS: list[FunctionTool] = [
    FunctionTool(validate_ci_input),
    FunctionTool(ci_pipeline_status),
]


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_ORCHESTRATOR_PROMPT = """
You are the Class Imbalance Handling Orchestrator.
You manage a 3-agent pipeline to detect and handle class imbalance in
training data before model training begins.

You have 3 sub-agents:
ci_analyzer_agent, ci_strategist_agent, ci_implementor_agent

You will receive:
Sandbox already contains X_train, X_test, y_train, y_test
target_col already set in sandbox
task_type: CLASSIFICATION or REGRESSION

YOUR EXACT WORKFLOW:

PHASE 1 — ANALYZE
Call ci_analyzer_agent with:
y_train value counts and percentages
y_test value counts
total train and test sample counts
target_col and target dtype
unique classes list and count
Wait for complete 6-section report before Phase 2.
If task is REGRESSION — skip to Phase 4 immediately
with status: skipped, reason: regression task.

PHASE 2 — STRATEGIZE
Call ci_strategist_agent with:
Full analysis report from Phase 1
target_col, total_train_samples
minority_class_percentage and minority_class_count
task_type and severity from Phase 1
Wait for complete strategy covering technique decision,
expected outcome, sandbox update plan, saving plan.

PHASE 3 — IMPLEMENT
Call ci_implementor_agent with:
Full strategy from Phase 2
Reminder: NEVER touch X_test or y_test
Reminder: update globals with X_train_bal, y_train_bal
Reminder: save outputs/balance_report.json
Wait for confirmation that:
X_train_bal and y_train_bal are in sandbox
X_test and y_test are unchanged
outputs/balance_report.json is saved

PHASE 4 — REPORT
Return structured JSON:

{
  \"status\": \"success\",
  \"task_type\": \"CLASSIFICATION\",
  \"imbalance_severity\": \"MODERATE\",
  \"technique_applied\": \"SMOTE\",
  \"smote_applied\": true,
  \"before_balance\": {
    \"class_0\": {\"count\": 800, \"percentage\": 80.0},
    \"class_1\": {\"count\": 200, \"percentage\": 20.0}
  },
  \"after_balance\": {
    \"class_0\": {\"count\": 800, \"percentage\": 50.0},
    \"class_1\": {\"count\": 800, \"percentage\": 50.0}
  },
  \"training_size_before\": 1000,
  \"training_size_after\": 1600,
  \"sandbox_ready\": {
    \"X_train_bal\": true,
    \"y_train_bal\": true,
    \"X_test_unchanged\": true,
    \"y_test_unchanged\": true,
    \"smote_applied\": true
  },
  \"output_files\": {
    \"balance_report\": \"outputs/balance_report.json\"
  },
  \"next_phase\": \"Model Training\"
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Agent definition
# ─────────────────────────────────────────────────────────────────────────────

ci_orchestrator_agent = Agent(
    name="ci_orchestrator_agent",
    model="gemini-2.0-flash",
    description=(
        "3-agent Class Imbalance orchestrator. Runs Analyze -> Strategist -> "
        "Implement in strict order, then returns training-ready balancing outputs."
    ),
    instruction=_ORCHESTRATOR_PROMPT,
    tools=_ORCHESTRATOR_TOOLS,
    sub_agents=[
        ci_analyzer_agent,
        ci_strategist_agent,
        ci_implementor_agent,
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_list(values: Any) -> list[Any]:
    """
    Convert a scalar/sequence to a plain Python list.
    """
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    try:
        return list(values)
    except TypeError:
        return [values]


def _balance_payload(distribution: Any, percentages: Any | None = None) -> dict[str, dict[str, Any]]:
    """
    Normalize class -> count/percentage into report-safe payload.
    """
    if not isinstance(distribution, dict):
        return {}
    total = 0
    payload: dict[str, dict[str, Any]] = {}
    for _, count in distribution.items():
        try:
            total += int(count)
        except (TypeError, ValueError):
            continue

    for label, count_raw in distribution.items():
        if isinstance(count_raw, dict):
            pct_from_value = count_raw.get("percentage", None)
            try:
                count = int(count_raw.get("count", 0))
            except (TypeError, ValueError):
                count = 0
        else:
            pct_from_value = None
            try:
                count = int(count_raw)
            except (TypeError, ValueError):
                count = 0

        if isinstance(percentages, dict):
            pct = percentages.get(
                label,
                percentages.get(str(label), 0.0),
            )
            if pct == 0.0 and pct_from_value is not None:
                pct = pct_from_value
        else:
            pct = round((count / total) * 100, 2) if total > 0 else 0.0
        try:
            pct_val = float(pct)
        except (TypeError, ValueError):
            pct_val = 0.0
        payload[str(label)] = {"count": count, "percentage": round(pct_val, 2)}
    return payload


def _extract_selected_option(plan_text: str) -> str:
    """
    Extract the exact option letter selected by strategist.
    """
    txt = str(plan_text or "").upper()
    if re.search(r"OPTION\s*B", txt):
        return "B"
    if re.search(r"OPTION\s*C", txt):
        return "C"
    if re.search(r"OPTION\s*D", txt):
        return "D"
    if re.search(r"OPTION\s*A", txt):
        return "A"
    return "A"


def _technique_from_option(option: str) -> str:
    """
    Convert strategist option to expected final technique label.
    """
    return {
        "A": "none",
        "B": "SMOTE",
        "C": "Random Undersampling",
        "D": "SMOTE + Tomek Links",
    }.get(option, "none")


def _smote_flag(option: str) -> bool:
    """
    Return True only when synthetic over-sampling should set smote_applied.
    """
    return option in {"B", "D"}


def _run_python_snippet(code: str) -> dict[str, Any]:
    """
    Run a snippet in the executor sandbox and parse JSON output.
    """
    raw = execute_python(code)
    if not raw or raw.startswith("ERROR:"):
        return {}
    try:
        return ast.literal_eval(raw.strip())
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(raw.strip())
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _test_state_snapshot() -> dict[str, Any]:
    """
    Read the immutable-test-data sandbox state for pre/post comparison.
    """
    code = """
import json
state = {
    "x_test_exists": False,
    "y_test_exists": False,
    "x_test_shape": None,
    "y_test_shape": None,
    "y_test_head": [],
}
x_test = globals().get("X_test")
y_test = globals().get("y_test")

if x_test is not None:
    state["x_test_exists"] = True
    try:
        state["x_test_shape"] = list(x_test.shape)
    except Exception:
        pass

if y_test is not None:
    state["y_test_exists"] = True
    try:
        values = y_test.tolist() if hasattr(y_test, "tolist") else list(y_test)
    except Exception:
        values = []
    state["y_test_shape"] = len(values)
    state["y_test_head"] = values[:50]

print(json.dumps(state))
"""
    return _run_python_snippet(code)


def _extract_distribution_from_report(balance_report: dict[str, Any], phase: str) -> dict[str, int]:
    """
    Extract before/after class counts from a balance report payload.
    """
    if not isinstance(balance_report, dict):
        return {}

    direct_before_keys = [
        "before",
        "before_balance",
        "before_distribution",
        "y_train_distribution_before",
        "distribution_before",
    ]
    direct_after_keys = [
        "after",
        "after_balance",
        "after_distribution",
        "y_train_distribution_after",
        "distribution_after",
    ]

    def _first_dict(candidates: list[str]) -> dict[str, Any] | None:
        for key in candidates:
            value = balance_report.get(key)
            if isinstance(value, dict):
                return value
            if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                merged: dict[str, Any] = {}
                for entry in value:
                    if isinstance(entry, dict):
                        merged.update(entry)
                if merged:
                    return merged
        return None

    if phase == "before":
        candidate = _first_dict(direct_before_keys)
        if isinstance(candidate, dict):
            maybe_counts = candidate.get("class_counts")
            if isinstance(maybe_counts, dict):
                return maybe_counts
            if not any(isinstance(v, dict) for v in candidate.values()):
                return candidate
            normalized = {}
            for key, value in candidate.items():
                if isinstance(value, dict):
                    count = value.get("count")
                    if isinstance(count, dict):
                        continue
                    if count is not None:
                        try:
                            normalized[key] = int(count)
                        except (TypeError, ValueError):
                            normalized[key] = 0
                else:
                    try:
                        normalized[key] = int(value)
                    except (TypeError, ValueError):
                        pass
            return normalized
    else:
        candidate = _first_dict(direct_after_keys)
        if isinstance(candidate, dict):
            maybe_counts = candidate.get("class_counts")
            if isinstance(maybe_counts, dict):
                return maybe_counts
            if not any(isinstance(v, dict) for v in candidate.values()):
                return candidate
            normalized = {}
            for key, value in candidate.items():
                if isinstance(value, dict):
                    count = value.get("count")
                    if isinstance(count, dict):
                        continue
                    if count is not None:
                        try:
                            normalized[key] = int(count)
                        except (TypeError, ValueError):
                            normalized[key] = 0
                else:
                    try:
                        normalized[key] = int(value)
                    except (TypeError, ValueError):
                        pass
            return normalized
    return {}


def _extract_percentages_from_report(
    balance_report: dict[str, Any],
    phase: str,
) -> dict[str, float]:
    """
    Extract before/after percentages from a balance report payload.
    """
    if not isinstance(balance_report, dict):
        return {}

    direct_before_keys = [
        "before",
        "before_balance",
        "before_distribution",
        "y_train_distribution_before",
        "distribution_before",
    ]
    direct_after_keys = [
        "after",
        "after_balance",
        "after_distribution",
        "y_train_distribution_after",
        "distribution_after",
    ]

    def _first_dict(candidates: list[str]) -> dict[str, Any] | None:
        for key in candidates:
            value = balance_report.get(key)
            if isinstance(value, dict):
                return value
        return None

    if phase == "before":
        candidate = _first_dict(direct_before_keys)
    else:
        candidate = _first_dict(direct_after_keys)
    if isinstance(candidate, dict):
        for key in ("percentages", "class_percentages"):
            vals = candidate.get(key)
            if isinstance(vals, dict):
                return vals
        # Nested case: {"classA": {"count": 10, "percentage": 50.0}, ...}
        nested_percentages: dict[str, float] = {}
        for key, value in candidate.items():
            if isinstance(value, dict) and "percentage" in value:
                raw = value.get("percentage", 0.0)
                if isinstance(raw, dict):
                    continue
                try:
                    nested_percentages[str(key)] = float(raw)
                except (TypeError, ValueError):
                    nested_percentages[str(key)] = 0.0
        if nested_percentages:
            return nested_percentages
    return {}


def _read_balance_report(path: str = "outputs/balance_report.json") -> dict[str, Any]:
    """
    Load outputs/balance_report.json when it exists.
    """
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline runner — called by model pipeline once train/test splits exist
# ─────────────────────────────────────────────────────────────────────────────

async def run_ci_orchestrator(
    y_train: list[Any] | Any,
    y_test: list[Any] | Any,
    target_col: str,
    target_dtype: str,
    task_type: str,
) -> dict[str, Any]:
    """
    Run the full Class Imbalance pipeline in strict phase order.

    Args:
        y_train:      Training labels.
        y_test:       Test labels.
        target_col:    ML target column name.
        target_dtype:  Declared dtype string of target column.
        task_type:     CLASSIFICATION or REGRESSION.

    Returns:
        Structured JSON payload consumed by Model Training.
    """
    start = time.monotonic()
    log_lines: list[str] = []
    errors: list[str] = []

    y_train_list = _to_list(y_train)
    y_test_list = _to_list(y_test)

    def _log(msg: str) -> None:
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    normalized_task = (task_type or "").strip().upper()
    total_train_samples = len(y_train_list)

    validation = validate_ci_input(
        target_col=target_col,
        task_type=normalized_task,
        total_train_samples=total_train_samples,
    )
    if not validation["valid"]:
        return {
            "status": "error",
            "task_type": normalized_task,
            "imbalance_severity": "UNKNOWN",
            "technique_applied": "none",
            "smote_applied": False,
            "before_balance": {},
            "after_balance": {},
            "training_size_before": 0,
            "training_size_after": 0,
            "sandbox_ready": {
                "X_train_bal": False,
                "y_train_bal": False,
                "X_test_unchanged": False,
                "y_test_unchanged": False,
                "smote_applied": False,
            },
            "output_files": {"balance_report": "outputs/balance_report.json"},
            "next_phase": "Model Training",
            "errors": validation["errors"],
            "pipeline_log": f"Validation failed: {validation['summary']}",
        }

    _log("Validation passed.")

    # Phase 1 — Analyze
    if normalized_task == "REGRESSION":
        _log("Phase 1/4 skipped (regression task) and moving to Phase 4.")
        analysis_stats = analyze_class_imbalance(
            y_train_list,
            y_test_list,
            target_col,
            target_dtype,
        )
        before = _balance_payload(
            analysis_stats.get("y_train_distribution"),
            analysis_stats.get("y_train_percentages"),
        )
        return {
            "status": "skipped",
            "reason": "REGRESSION — balancing not applicable",
            "task_type": "REGRESSION",
            "imbalance_severity": "N/A",
            "technique_applied": "none",
            "smote_applied": False,
            "before_balance": before,
            "after_balance": before,
            "training_size_before": analysis_stats.get("total_train_samples", total_train_samples),
            "training_size_after": analysis_stats.get("total_train_samples", total_train_samples),
            "sandbox_ready": {
                "X_train_bal": False,
                "y_train_bal": False,
                "X_test_unchanged": True,
                "y_test_unchanged": True,
                "smote_applied": False,
            },
            "output_files": {"balance_report": "outputs/balance_report.json"},
            "next_phase": "Model Training",
            "errors": [],
            "pipeline_log": "\n".join(log_lines),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }

    _log("Phase 1/4 — Analyzer starting.")
    analyzer_report: str = ""
    try:
        analyzer_report = await run_ci_analyzer(
            y_train=y_train_list,
            y_test=y_test_list,
            target_col=target_col,
            target_dtype=target_dtype,
        )
        if not analyzer_report:
            raise RuntimeError("Empty analyzer report.")
        _log("Phase 1/4 — Analyzer done ✓")
    except Exception as exc:
        msg = f"CI Analyzer failed: {exc}"
        errors.append(msg)
        return {
            "status": "error",
            "task_type": normalized_task,
            "imbalance_severity": "UNKNOWN",
            "technique_applied": "none",
            "smote_applied": False,
            "before_balance": {},
            "after_balance": {},
            "training_size_before": total_train_samples,
            "training_size_after": 0,
            "sandbox_ready": {
                "X_train_bal": False,
                "y_train_bal": False,
                "X_test_unchanged": False,
                "y_test_unchanged": False,
                "smote_applied": False,
            },
            "output_files": {"balance_report": "outputs/balance_report.json"},
            "next_phase": "Model Training",
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }

    analysis_stats = analyze_class_imbalance(
        y_train_list,
        y_test_list,
        target_col,
        target_dtype,
    )
    before = _balance_payload(
        analysis_stats.get("y_train_distribution", {}),
        analysis_stats.get("y_train_percentages", {}),
    )
    total_train_samples = int(analysis_stats.get("total_train_samples", total_train_samples))
    minority_class_percentage = float(
        analysis_stats.get("minority_class_percentage", 0.0) or 0.0
    )
    minority_class_count = (
        analysis_stats.get("y_train_distribution", {}).get(
            analysis_stats.get("minority_class"), 0
        )
        if analysis_stats.get("minority_class") is not None
        else 0
    )
    if minority_class_percentage >= 40:
        imbalance_severity = "BALANCED"
    elif minority_class_percentage >= 20:
        imbalance_severity = "MILD"
    elif minority_class_percentage >= 10:
        imbalance_severity = "MODERATE"
    else:
        imbalance_severity = "SEVERE"

    # Phase 2 — Strategize
    _log("Phase 2/4 — Strategist starting.")
    strategist_output: dict[str, Any] | None = None
    try:
        strategist_output = await run_ci_strategist(
            analyzer_report=analyzer_report,
            target_col=target_col,
            total_train_samples=total_train_samples,
            minority_class_percentage=minority_class_percentage,
            minority_class_count=minority_class_count,
            task_type=normalized_task,
            severity=imbalance_severity,
        )
        if not strategist_output:
            raise RuntimeError("Strategist output missing.")
        _log("Phase 2/4 — Strategist done ✓")
    except Exception as exc:
        msg = f"CI Strategist failed: {exc}"
        errors.append(msg)
        return {
            "status": "error",
            "task_type": normalized_task,
            "imbalance_severity": imbalance_severity,
            "technique_applied": "none",
            "smote_applied": False,
            "before_balance": before,
            "after_balance": before,
            "training_size_before": total_train_samples,
            "training_size_after": total_train_samples,
            "sandbox_ready": {
                "X_train_bal": False,
                "y_train_bal": False,
                "X_test_unchanged": False,
                "y_test_unchanged": False,
                "smote_applied": False,
            },
            "output_files": {"balance_report": "outputs/balance_report.json"},
            "next_phase": "Model Training",
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }

    # Phase 3 — Implementor
    _log("Phase 3/4 — Implementor starting.")
    pre_test_state = _test_state_snapshot()
    implementor_output: dict[str, Any] | None = None
    try:
        implementor_output = await run_ci_implementor(strategist_output)
        if not isinstance(implementor_output, dict):
            raise RuntimeError("Implementor output invalid.")
        _log("Phase 3/4 — Implementor done ✓")
    except Exception as exc:
        msg = f"CI Implementor failed: {exc}"
        errors.append(msg)
        implementor_output = {}
        _log(f"Phase 3/4 — ERROR: {msg}")

    post_test_state = _test_state_snapshot()

    # Read output artifacts / artifacts
    balance_report = _read_balance_report("outputs/balance_report.json")
    verify_balance_file = verify_output_saved("outputs/balance_report.json")

    selected_option = _extract_selected_option(
        strategist_output.get("balance_plan", "") if isinstance(strategist_output, dict) else ""
    )
    technique_applied = _technique_from_option(selected_option)
    smote_applied = _smote_flag(selected_option)

    # If execution indicates fallback to skip, normalize final technique.
    if (
        isinstance(implementor_output, dict)
        and implementor_output.get("selected_technique") == "skip"
    ):
        technique_applied = "none"
        smote_applied = False

    # Determine whether implementation artifacts indicate completion.
    sandbox_globals = _run_python_snippet(
        """
import json
print(
    json.dumps({
        "x_bal_exists": "X_train_bal" in globals(),
        "y_bal_exists": "y_train_bal" in globals(),
        "smote_value": globals().get("smote_applied", False),
    })
)
"""
    )
    x_train_bal_exists = bool(sandbox_globals.get("x_bal_exists", False))
    y_train_bal_exists = bool(sandbox_globals.get("y_bal_exists", False))
    smote_flag_runtime = bool(sandbox_globals.get("smote_value", False))

    x_test_unchanged = (
        pre_test_state.get("x_test_exists")
        and post_test_state.get("x_test_exists")
        and pre_test_state.get("x_test_shape") == post_test_state.get("x_test_shape")
    )
    y_test_unchanged = (
        pre_test_state.get("y_test_exists")
        and post_test_state.get("y_test_exists")
        and pre_test_state.get("y_test_shape") == post_test_state.get("y_test_shape")
        and pre_test_state.get("y_test_head") == post_test_state.get("y_test_head")
    )

    # Derive post-balance class distribution.
    after_distribution = _extract_distribution_from_report(balance_report, "after")
    after_percentages = _extract_percentages_from_report(balance_report, "after")
    if not after_distribution:
        if y_train_bal_exists:
            derived = _run_python_snippet(
                """
import json
import pandas as pd
y = globals().get("y_train_bal")
if y is None:
    print("{}")
else:
    s = pd.Series(y)
    print(json.dumps({
        "counts": s.value_counts(dropna=False).to_dict(),
        "total": int(len(s)),
    }))
"""
            )
            if isinstance(derived, dict):
                after_distribution = {
                    k: int(v) for k, v in derived.get("counts", {}).items()
                } if isinstance(derived.get("counts", {}), dict) else {}
                total_after = int(derived.get("total", 0) or 0)
                if total_after:
                    after_percentages = {
                        k: round((int(v) / total_after) * 100, 2) for k, v in after_distribution.items()
                    }

    after = _balance_payload(after_distribution, after_percentages)

    if not after:
        after = before
    if not after_percentages and isinstance(after_distribution, dict):
        total_after_fallback = sum(int(v) for v in after_distribution.values())
        after = _balance_payload(after_distribution, None)

    training_size_after = 0
    if x_train_bal_exists:
        y_train_bal_count = _run_python_snippet(
            """
import json
y = globals().get("y_train_bal")
if y is None:
    print("{}")
else:
    try:
        size = int(len(y))
    except Exception:
        size = 0
    print(json.dumps({"size": size}))
"""
        )
        if isinstance(y_train_bal_count, dict):
            training_size_after = int(y_train_bal_count.get("size", 0) or 0)

    if training_size_after == 0:
        training_size_after = sum(
            entry.get("count", 0) for entry in after.values() if isinstance(entry, dict)
        )

    status = "success"
    if verify_balance_file.get("exists") is False:
        status = "error"
        errors.append("outputs/balance_report.json was not saved.")
    if not x_train_bal_exists or not y_train_bal_exists:
        status = "error"
        errors.append("X_train_bal and/or y_train_bal not present in sandbox globals.")

    if x_train_bal_exists and y_train_bal_exists and verify_balance_file.get("exists"):
        # If implementation is SMOTE-derived, prefer runtime-smote flag from sandbox.
        smote_applied = bool(bool(smote_applied) or bool(smote_flag_runtime))

    return {
        "status": status,
        "task_type": normalized_task,
        "imbalance_severity": imbalance_severity,
        "technique_applied": technique_applied if status != "error" else technique_applied,
        "smote_applied": smote_applied,
        "before_balance": before,
        "after_balance": after,
        "training_size_before": total_train_samples,
        "training_size_after": training_size_after,
        "sandbox_ready": {
            "X_train_bal": x_train_bal_exists,
            "y_train_bal": y_train_bal_exists,
            "X_test_unchanged": bool(x_test_unchanged),
            "y_test_unchanged": bool(y_test_unchanged),
            "smote_applied": bool(smote_applied),
        },
        "output_files": {
            "balance_report": "outputs/balance_report.json",
        },
        "next_phase": "Model Training",
        "analysis_report": analyzer_report,
        "strategy": strategist_output,
        "implementation": implementor_output,
        "balance_report": balance_report,
        "errors": errors,
        "pipeline_log": "\n".join(log_lines),
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }


# Backward-compatible alias used by different pipeline entrypoints
run_ci_pipeline = run_ci_orchestrator
