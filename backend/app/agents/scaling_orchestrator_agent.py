"""
app/agents/scaling_orchestrator_agent.py

Feature Scaling Orchestrator — top-level manager of the scaling pipeline.

Architecture:
  ┌───────────────────────────────────────────────────────────────┐
  │              Scaling Orchestrator                             │
  │            3-phase pipeline (Analyze → Strategize → Implement) │
  └──────────────┬──────────────────┬───────────────────────────┘
                 │                  │
                 ▼                  ▼
         Scaling Analyzer     Scaling Strategist        Scaling Implementor
         (analysis)           (plan)                   (execute)
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

from app.agents.scaling_agent.analyzer_agent import (
    scaling_analyzer_agent,
    run_scaling_analyzer,
)
from app.agents.scaling_agent.executor_agent import (
    scaling_implementor_agent,
    run_scaling_implementor,
)
from app.agents.scaling_agent.strategist_agent import (
    scaling_strategist_agent,
    run_scaling_strategist,
)
from app.tools.executor_tools import execute_python

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator-owned tools
# ─────────────────────────────────────────────────────────────────────────────


def validate_fs_input(
    target_col: str,
    record_count: int,
    column_names: list[str],
    numeric_cols: list[str],
    encoded_cols: list[str],
    transformed_cols: list[str],
) -> dict[str, Any]:
    """Validate scaling pipeline inputs before running any phase."""
    errors: list[str] = []

    if not target_col or not target_col.strip():
        errors.append("target_col is empty or missing.")

    if record_count < 1:
        errors.append("df_engineered has no rows.")

    if not column_names:
        errors.append("df_engineered has no columns.")
    elif target_col and target_col not in column_names:
        errors.append(
            f"target_col '{target_col}' not found in columns: {column_names}."
        )

    if target_col in encoded_cols:
        errors.append(
            f"target_col '{target_col}' should not be in encoded_cols (already encoded list)."
        )

    if target_col in transformed_cols:
        errors.append(
            f"target_col '{target_col}' should not be in transformed_cols (already transformed list)."
        )

    valid = len(errors) == 0
    summary = (
        f"FS validation passed. Rows={record_count}, columns={len(column_names)}. "
        f"Target '{target_col}' found."
        if valid
        else f"FS validation FAILED: {'; '.join(errors)}"
    )
    return {"valid": valid, "errors": errors, "summary": summary}


def scaling_pipeline_status(
    stage: str,
    status: str,
    message: str,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    """Report current scaling pipeline stage and status."""
    return {
        "stage": stage,
        "status": status,
        "message": message,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


_ORCHESTRATOR_TOOLS: list[FunctionTool] = [
    FunctionTool(validate_fs_input),
    FunctionTool(scaling_pipeline_status),
]


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_ORCHESTRATOR_PROMPT = """
You are the Feature Scaling Orchestrator.
You manage a 3-agent pipeline to scale features and produce
train/test splits ready for model training.

You have 3 sub-agents:
scaling_analyzer_agent, scaling_strategist_agent, scaling_implementor_agent

You will receive:
- df_stats: statistics from df_engineered
- target_col: ML target column name
- numeric_cols: list of numeric columns
- encoded_cols: columns encoded in FE phase
- transformed_cols: columns transformed in FE phase

YOUR EXACT WORKFLOW:

PHASE 1 — ANALYZE
Call scaling_analyzer_agent with df_stats from df_engineered.
Must receive 5-section report covering range analysis,
distribution check, already-scaled columns, target check,
and scaling recommendations basis.
Wait for complete report before Phase 2.

PHASE 2 — STRATEGIZE
Call scaling_strategist_agent with:
Full analysis report from Phase 1
Target/metadata
Must receive plan covering: skip list, StandardScaler cols,
RobustScaler cols, MinMaxScaler cols, saving plan.
Wait for complete plan before Phase 3.

PHASE 3 — IMPLEMENT
Call scaling_implementor_agent with:
Full scaling plan from Phase 2
Reminder: split BEFORE scaling, fit on train only
Reminder: save X_train, X_test, y_train, y_test to sandbox
Wait for confirmation that these files are saved:
outputs/train_data.csv
outputs/test_data.csv
outputs/scaling_summary.json
outputs/scaler_*.pkl files

PHASE 4 — REPORT
Return structured JSON:

{
  "status": "success",
  "train_shape": [800, 14],
  "test_shape": [200, 14],
  "scaling_applied": {
    "standard": ["age", "experience_years"],
    "robust": ["salary"],
    "minmax": [],
    "skipped": ["gender", "dept_encoded", "promoted"]
  },
  "scalers_saved": [
    "outputs/scaler_standard.pkl",
    "outputs/scaler_robust.pkl"
  ],
  "output_files": {
    "train": "outputs/train_data.csv",
    "test": "outputs/test_data.csv",
    "summary": "outputs/scaling_summary.json"
  },
  "sandbox_ready": {
    "X_train": true,
    "X_test": true,
    "y_train": true,
    "y_test": true
  },
  "next_phase": "Model Training"
}

STRICT RULES:
- Never skip a phase
- target_col must NEVER be in any scaling list — verify this
- Split before scale — verify implementor followed this order
- Always pass COMPLETE output between phases
- If sandbox not updated correctly — retry implementor once
"""


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Agent definition
# ─────────────────────────────────────────────────────────────────────────────

scaling_orchestrator_agent = Agent(
    name="scaling_orchestrator_agent",
    model="gemini-2.0-flash",
    description=(
        "Manages scaling pipeline phases: Analyze -> Strategist -> Implementor. "
        "Runs all 3 phases in strict order and returns structured JSON "
        "consumed by Model Training."
    ),
    instruction=_ORCHESTRATOR_PROMPT,
    tools=_ORCHESTRATOR_TOOLS,
    sub_agents=[
        scaling_analyzer_agent,
        scaling_strategist_agent,
        scaling_implementor_agent,
    ],
)

# Backward-compatible alias
fs_orchestrator_agent = scaling_orchestrator_agent


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_PATTERNS = {
    "skip": re.compile(r"^\s*1\s*[.)]?\s*.*SKIP", re.IGNORECASE),
    "standard": re.compile(r"^\s*2\s*[.)]?\s*.*STANDARD", re.IGNORECASE),
    "robust": re.compile(r"^\s*3\s*[.)]?\s*.*ROBUST", re.IGNORECASE),
    "minmax": re.compile(r"^\s*4\s*[.)]?\s*.*MINMAX", re.IGNORECASE),
}


def _normalize_shape(shape: Any) -> list[int]:
    """Normalize shape values to [rows, cols]."""
    if isinstance(shape, list) and len(shape) == 2:
        try:
            return [int(shape[0]), int(shape[1])]
        except (TypeError, ValueError):
            return [0, 0]
    if isinstance(shape, tuple) and len(shape) == 2:
        try:
            return [int(shape[0]), int(shape[1])]
        except (TypeError, ValueError):
            return [0, 0]
    if isinstance(shape, dict):
        rows = shape.get("rows")
        cols = shape.get("columns")
        if isinstance(rows, int) and isinstance(cols, int):
            return [rows, cols]
        values = shape.get("shape")
        if isinstance(values, (list, tuple)) and len(values) == 2:
            try:
                return [int(values[0]), int(values[1])]
            except (TypeError, ValueError):
                return [0, 0]
    return [0, 0]


def _dedupe(items: list[str]) -> list[str]:
    """Deduplicate while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_plan_lines(plan_text: str, section: str) -> list[str]:
    """Extract lines under a numbered scaling-plan section."""
    if not plan_text:
        return []

    lines = [ln.strip() for ln in plan_text.splitlines() if ln.strip()]
    start_idx: int | None = None
    stop_idx: int | None = None

    pat = _SECTION_PATTERNS.get(section)
    if not pat:
        return []

    header_indices = [i for i, line in enumerate(lines) if pat.match(line)]
    if not header_indices:
        return []

    start_idx = header_indices[0] + 1
    for i in range(start_idx, len(lines)):
        if any(regex.match(lines[i]) for regex in _SECTION_PATTERNS.values()):
            stop_idx = i
            break
    stop_idx = stop_idx if stop_idx is not None else len(lines)
    return lines[start_idx:stop_idx]


def _parse_scaling_lists(plan_text: str) -> dict[str, list[str]]:
    """Parse exact scaling lists from a numbered scaling plan."""
    out = {"skip": [], "standard": [], "robust": [], "minmax": []}

    def _line_tokens(line: str) -> list[str]:
        if not line:
            return []
        clean = re.split(r"[;|:]", line, maxsplit=1)[0]
        tokens = re.split(r",|\band\b", clean)
        cleaned: list[str] = []
        for tok in tokens:
            t = tok.strip().lstrip("-•* ")
            t = re.split(r"\s*->\s*", t)[0]
            t = t.split(" ", 1)[0]
            t = t.strip("`\"'”’")
            if t:
                cleaned.append(t)
        return cleaned

    for section in out:
        for ln in _extract_plan_lines(plan_text, section):
            for token in _line_tokens(ln):
                out[section].append(token)

    return {
        "skip": _dedupe([x for x in out["skip"] if x]),
        "standard": _dedupe([x for x in out["standard"] if x]),
        "robust": _dedupe([x for x in out["robust"] if x]),
        "minmax": _dedupe([x for x in out["minmax"] if x]),
    }


def _read_json_summary(path: str) -> dict[str, Any]:
    """Load a JSON artifact if it exists, else return {}."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as file:
            return json.load(file)
    except Exception:
        return {}


def _read_shape_csv(path: str) -> list[int]:
    """Read CSV row/column counts without loading full file."""
    if not os.path.isfile(path):
        return []
    with open(path, "r") as file:
        lines = file.readlines()
    if not lines:
        return [0, 0]
    cols = lines[0].strip().split(",") if lines[0].strip() else []
    rows = max(0, len(lines) - 1)
    return [rows, len(cols)]


def _check_sandbox_vars() -> dict[str, bool]:
    """Check if expected split variables exist in sandbox globals."""
    result = {"X_train": False, "X_test": False, "y_train": False, "y_test": False}
    snippet = (
        "_tmp = {\n"
        "    'X_train': 'X_train' in globals(),\n"
        "    'X_test': 'X_test' in globals(),\n"
        "    'y_train': 'y_train' in globals(),\n"
        "    'y_test': 'y_test' in globals(),\n"
        "}\n"
        "print(_tmp)"
    )
    out = execute_python(snippet)
    if out and out.startswith("ERROR:"):
        return result

    try:
        payload = ast.literal_eval(out.strip()) if out else {}
        if isinstance(payload, dict):
            for key in result:
                result[key] = bool(payload.get(key, False))
    except Exception:
        pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_scaling_orchestrator(
    df_stats: dict[str, Any],
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    encoded_cols: list[str],
    transformed_cols: list[str],
) -> dict[str, Any]:
    """
    Run feature scaling pipeline in strict 3-phase order.

    Args:
        df_stats:        Precomputed statistics from engineered data.
        target_col:      ML target column name.
        numeric_cols:    Numeric columns to scale or consider.
        categorical_cols: Categorical columns retained in engineered set.
        encoded_cols:    FE encoded columns.
        transformed_cols: FE transformed columns.

    Returns:
        Structured JSON payload used by downstream model-training flow.
    """
    start = time.monotonic()
    log_lines: list[str] = []
    errors: list[str] = []

    def _log(msg: str) -> None:
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    records: list[dict[str, Any]] = []
    if isinstance(df_stats, dict):
        candidate = df_stats.get("records")
        if isinstance(candidate, list):
            records = candidate

    input_shape = _normalize_shape(
        df_stats.get("shape", [0, 0]) if isinstance(df_stats, dict) else [0, 0]
    )

    resolved_numeric = list(numeric_cols)
    resolved_encoded = list(encoded_cols)
    resolved_transformed = list(transformed_cols)
    resolved_categorical = list(categorical_cols)

    if not resolved_numeric and isinstance(df_stats, dict):
        resolved_numeric = list(df_stats.get("numeric_cols", []))
    if not resolved_categorical and isinstance(df_stats, dict):
        resolved_categorical = list(df_stats.get("categorical_cols", []))

    column_names: list[str] = []
    if records:
        column_names = list(records[0].keys())
    elif isinstance(df_stats, dict):
        dtypes = df_stats.get("dtypes")
        if isinstance(dtypes, dict):
            column_names = list(dtypes.keys())
    if not column_names and resolved_numeric:
        column_names = list(dict.fromkeys(resolved_numeric + resolved_categorical))

    if not input_shape[0] and records:
        input_shape = [len(records), len(records[0]) if records else 0]

    validation = validate_fs_input(
        target_col=target_col,
        record_count=input_shape[0],
        column_names=column_names,
        numeric_cols=resolved_numeric,
        encoded_cols=resolved_encoded,
        transformed_cols=resolved_transformed,
    )
    if not validation["valid"]:
        return {
            "status": "error",
            "train_shape": [],
            "test_shape": [],
            "scaling_applied": {"standard": [], "robust": [], "minmax": [], "skipped": []},
            "target_col": target_col,
            "scalers_saved": [],
            "output_files": {
                "train": "outputs/train_data.csv",
                "test": "outputs/test_data.csv",
                "summary": "outputs/scaling_summary.json",
            },
            "sandbox_ready": {"X_train": False, "X_test": False, "y_train": False, "y_test": False},
            "next_phase": "Model Training",
            "errors": validation["errors"],
            "pipeline_log": f"Validation failed: {validation['summary']}",
        }

    _log("Scaling validation passed.")

    # Phase 1 — Analyze
    _log("Phase 1/3 — Analyzer starting.")
    analyzer_report: str = ""
    try:
        if not records:
            raise RuntimeError("No records available for scaling analyzer bootstrap.")
        analyzer_report = await run_scaling_analyzer(
            records=records,
            target_col=target_col,
            encoded_cols=resolved_encoded,
            transformed_cols=resolved_transformed,
            value_ranges=df_stats.get("value_ranges") if isinstance(df_stats, dict) else {},
        )
        if not analyzer_report:
            raise RuntimeError("Empty analyzer report.")
        _log("Phase 1/3 — Analyzer done ✓")
    except Exception as exc:
        msg = f"Scaling Analyzer failed: {exc}"
        errors.append(msg)
        _log(f"Phase 1/3 — ERROR: {msg}")
        return {
            "status": "error",
            "train_shape": [],
            "test_shape": [],
            "scaling_applied": {"standard": [], "robust": [], "minmax": [], "skipped": []},
            "target_col": target_col,
            "scalers_saved": [],
            "output_files": {
                "train": "outputs/train_data.csv",
                "test": "outputs/test_data.csv",
                "summary": "outputs/scaling_summary.json",
            },
            "sandbox_ready": {"X_train": False, "X_test": False, "y_train": False, "y_test": False},
            "next_phase": "Model Training",
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }

    # Phase 2 — Strategize
    _log("Phase 2/3 — Strategist starting.")
    strategist_output: dict[str, Any]
    try:
        strategist_output = await run_scaling_strategist(
            analyzer_report=analyzer_report,
            target_col=target_col,
            numeric_cols=resolved_numeric,
            encoded_cols=resolved_encoded,
            transformed_cols=resolved_transformed,
        )
        scaling_plan = strategist_output.get("scaling_plan", "")
        if not isinstance(scaling_plan, str) or not scaling_plan.strip():
            raise ValueError("Strategist plan is empty.")
        _log("Phase 2/3 — Strategist done ✓")
    except Exception as exc:
        msg = f"Scaling Strategist failed: {exc}"
        errors.append(msg)
        _log(f"Phase 2/3 — ERROR: {msg}")
        return {
            "status": "error",
            "train_shape": [],
            "test_shape": [],
            "scaling_applied": {"standard": [], "robust": [], "minmax": [], "skipped": []},
            "target_col": target_col,
            "scalers_saved": [],
            "output_files": {
                "train": "outputs/train_data.csv",
                "test": "outputs/test_data.csv",
                "summary": "outputs/scaling_summary.json",
            },
            "sandbox_ready": {"X_train": False, "X_test": False, "y_train": False, "y_test": False},
            "next_phase": "Model Training",
            "analysis_report": analyzer_report,
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }

    # Phase 3 — Implementor (retry once if missing outputs)
    _log("Phase 3/3 — Implementor starting.")
    implementor_output: dict[str, Any] | None = None

    def _impl_output_ok(output: dict[str, Any]) -> bool:
        verify_train = output.get("verify_train", {})
        verify_test = output.get("verify_test", {})
        verify_summary = output.get("verify_summary", {})
        train_ok = isinstance(verify_train, dict) and bool(verify_train.get("exists", False))
        test_ok = isinstance(verify_test, dict) and bool(verify_test.get("exists", False))
        summary_ok = isinstance(verify_summary, dict) and bool(verify_summary.get("exists", False))
        return train_ok and test_ok and summary_ok

    impl_errors: list[str] = []
    for attempt in range(1, 3):
        try:
            implementor_output = await run_scaling_implementor(
                records=records,
                strategist_output=strategist_output,
            )
            if _impl_output_ok(implementor_output):
                _log(f"Phase 3/3 — Implementor attempt {attempt} done ✓")
                break
            err = (
                f"Scaling Implementor attempt {attempt} returned incomplete artifacts: "
                f"{implementor_output.get('verify_train')}, "
                f"{implementor_output.get('verify_test')}, "
                f"{implementor_output.get('verify_summary')}"
            )
            impl_errors.append(err)
            errors.append(err)
            if attempt == 1:
                _log("Phase 3/3 — retrying implementor once.")
                continue
            implementor_output = None
        except Exception as exc:
            err = f"Scaling Implementor attempt {attempt} failed: {exc}"
            impl_errors.append(err)
            errors.append(err)
            _log(f"Phase 3/3 — {err}")
            if attempt == 1:
                _log("Phase 3/3 — retrying implementor once.")

    if not implementor_output:
        _log("Phase 3/3 — Implementor did not complete.")
        return {
            "status": "error",
            "train_shape": [],
            "test_shape": [],
            "scaling_applied": {"standard": [], "robust": [], "minmax": [], "skipped": []},
            "target_col": target_col,
            "scalers_saved": [],
            "output_files": {
                "train": "outputs/train_data.csv",
                "test": "outputs/test_data.csv",
                "summary": "outputs/scaling_summary.json",
            },
            "sandbox_ready": _check_sandbox_vars(),
            "next_phase": "Model Training",
            "analysis_report": analyzer_report,
            "strategy": strategist_output,
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }

    # Re-check output artifacts on disk.
    train_output = "outputs/train_data.csv"
    test_output = "outputs/test_data.csv"
    summary_output = "outputs/scaling_summary.json"

    train_shape = _read_shape_csv(train_output)
    test_shape = _read_shape_csv(test_output)
    summary_data = _read_json_summary(summary_output)
    if summary_data:
        ts = summary_data.get("train_shape")
        if isinstance(ts, list) and len(ts) == 2:
            train_shape = [int(ts[0]), int(ts[1])]
        tts = summary_data.get("test_shape")
        if isinstance(tts, list) and len(tts) == 2:
            test_shape = [int(tts[0]), int(tts[1])]

    scaling_applied = {
        "standard": [],
        "robust": [],
        "minmax": [],
        "skipped": [target_col],
    }

    parsed_lists = implementor_output.get("parsed_lists", {}) if isinstance(implementor_output, dict) else {}
    if isinstance(parsed_lists, dict) and parsed_lists.get("standard_cols"):
        scaling_applied["standard"] = _dedupe(parsed_lists.get("standard_cols", []))
        scaling_applied["robust"] = _dedupe(parsed_lists.get("robust_cols", []))
        scaling_applied["minmax"] = _dedupe(parsed_lists.get("minmax_cols", []))
        scaling_applied["skipped"] = _dedupe([target_col] + list(parsed_lists.get("skip_cols", [])))
    else:
        parsed = _parse_scaling_lists(strategist_output.get("scaling_plan", ""))
        scaling_applied["standard"] = parsed["standard"]
        scaling_applied["robust"] = parsed["robust"]
        scaling_applied["minmax"] = parsed["minmax"]
        scaling_applied["skipped"] = _dedupe([target_col] + parsed["skip"])

    for bucket in ("standard", "robust", "minmax"):
        scaling_applied[bucket] = [
            c for c in scaling_applied[bucket]
            if c and c != target_col
        ]
        if target_col in scaling_applied[bucket]:
            errors.append(
                f"Target column '{target_col}' was incorrectly included in {bucket} scaling list. "
                "Removed for safety."
            )
    scaling_applied["skipped"] = [
        c for c in _dedupe(scaling_applied["skipped"]) if c and c != target_col
    ] + [target_col]

    scaling_applied = {
        "standard": scaling_applied["standard"],
        "robust": scaling_applied["robust"],
        "minmax": scaling_applied["minmax"],
        "skipped": _dedupe(scaling_applied["skipped"]),
    }

    sandbox_ready = {
        "X_train": False,
        "X_test": False,
        "y_train": False,
        "y_test": False,
    }

    file_ready = {
        "X_train": os.path.isfile(train_output),
        "X_test": os.path.isfile(test_output),
        "summary": os.path.isfile(summary_output),
    }

    if all(file_ready.values()):
        sandbox_ready = _check_sandbox_vars()

    scalers_saved = []
    summary_scalers = summary_data.get("scalers_saved") if isinstance(summary_data, dict) else None
    if isinstance(summary_scalers, list) and summary_scalers:
        scalers_saved = summary_scalers
    else:
        if scaling_applied["standard"]:
            scalers_saved.append("outputs/scaler_standard.pkl")
        if scaling_applied["robust"]:
            scalers_saved.append("outputs/scaler_robust.pkl")
        if scaling_applied["minmax"]:
            scalers_saved.append("outputs/scaler_minmax.pkl")

    status = "success" if file_ready["X_train"] and file_ready["X_test"] and file_ready["summary"] else "error"

    if not sandbox_ready["X_train"] or not sandbox_ready["X_test"] or not sandbox_ready["y_train"] or not sandbox_ready["y_test"]:
        status = "error"
        errors.append("Sandbox variables X_train/X_test/y_train/y_test are not all present after scaling.")

    if target_col in scaling_applied["standard"] + scaling_applied["robust"] + scaling_applied["minmax"]:
        status = "error"
        errors.append("target_col must never be in any scaling list")

    return {
        "status": status,
        "train_shape": train_shape,
        "test_shape": test_shape,
        "scaling_applied": scaling_applied,
        "target_col": target_col,
        "scalers_saved": scalers_saved,
        "output_files": {
            "train": train_output,
            "test": test_output,
            "summary": summary_output,
        },
        "sandbox_ready": sandbox_ready,
        "next_phase": "Model Training",
        "analysis_report": analyzer_report,
        "strategy": strategist_output,
        "implementation": implementor_output,
        "errors": errors + impl_errors,
        "pipeline_log": "\n".join(log_lines),
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }


# Backward-compatible alias
run_fs_orchestrator = run_scaling_orchestrator
