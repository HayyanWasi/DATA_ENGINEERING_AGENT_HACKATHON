"""
app/agents/fe_orchestrator_agent.py

Feature Engineering Orchestrator — top-level manager of the FE pipeline.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                 FE Orchestrator                             │
  │            3-phase pipeline (Analyze → Strategize → Implement)│
  └──────────────┬──────────────────┬───────────────────────────┘
                │                  │
                ▼                  ▼
         FE Analyzer         FE Strategist            FE Implementor
         (analysis)          (plan)                  (execute)
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.agents.fe_agent.analyzer_agent import fe_analyzer_agent, run_fe_analyzer
from app.agents.fe_agent.executor_agent import (
    fe_implementor_agent,
    run_fe_implementor,
)
from app.agents.fe_agent.strategist_agent import (
    fe_strategist_agent,
    run_fe_strategist,
)

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator-owned tools
# ─────────────────────────────────────────────────────────────────────────────

def validate_fe_input(
    target_col: str,
    record_count: int,
    column_names: list[str],
) -> dict[str, Any]:
    """
    Validate FE pipeline inputs before running any stage.

    Args:
        target_col:   ML target column name.
        record_count: Number of rows in df_clean.
        column_names: Column names available in df_clean.

    Returns:
        dict with keys:
          valid   (bool) — True if all checks pass
          errors  (list) — validation failures
          summary (str)  — compact validation summary
    """
    errors: list[str] = []

    if not target_col or not target_col.strip():
        errors.append("target_col is empty or missing.")

    if record_count < 1:
        errors.append("df_clean has no rows.")

    if not column_names:
        errors.append("df_clean has no columns.")
    elif target_col and target_col not in column_names:
        errors.append(
            f"target_col '{target_col}' not found in columns: {column_names}."
        )

    valid = len(errors) == 0
    summary = (
        f"FE validation passed. Rows={record_count}, columns={len(column_names)}. "
        f"Target '{target_col}' found."
        if valid
        else f"FE validation FAILED: {'; '.join(errors)}"
    )
    return {"valid": valid, "errors": errors, "summary": summary}


def fe_pipeline_status(
    stage: str,
    status: str,
    message: str,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    """
    Report current FE pipeline stage for observability.
    """
    return {
        "stage": stage,
        "status": status,
        "message": message,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


_ORCHESTRATOR_TOOLS: list[FunctionTool] = [
    FunctionTool(validate_fe_input),
    FunctionTool(fe_pipeline_status),
]

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_ORCHESTRATOR_PROMPT = """
You are the Feature Engineering Orchestrator.
You manage a 3-agent pipeline to engineer features from cleaned data
and prepare it for model training.

You have 3 sub-agents:
fe_analyzer_agent, fe_strategist_agent, fe_implementor_agent

## EXACT WORKFLOW:

PHASE 1 — ANALYZE
Call fe_analyzer_agent with complete df_stats.
Wait for complete 6-section analysis report before Phase 2.

PHASE 2 — STRATEGIZE
Call fe_strategist_agent with analysis report + target_col + numeric/categorical cols.
Wait for complete plan before Phase 3.

PHASE 3 — IMPLEMENT
Call fe_implementor_agent with the final plan.
Wait for confirmation engineered_data.csv and feature_summary.json are saved.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Agent definition
# ─────────────────────────────────────────────────────────────────────────────

fe_orchestrator_agent = Agent(
    name="fe_orchestrator_agent",
    model="gemini-2.0-flash",
    description=(
        "Manages FE pipeline stages: Analyze -> Strategist -> Implementor. "
        "Runs all 3 phases in strict order and returns structured JSON "
        "describing engineered output artifacts."
    ),
    instruction=_ORCHESTRATOR_PROMPT,
    tools=_ORCHESTRATOR_TOOLS,
    sub_agents=[
        fe_analyzer_agent,
        fe_strategist_agent,
        fe_implementor_agent,
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Private helpers — plan parsing + CSV inspection
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_shape(shape: Any) -> list[int]:
    """
    Normalize shape value from multiple forms to [rows, cols].
    """
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
        r = shape.get("rows")
        c = shape.get("columns")
        if isinstance(r, int) and isinstance(c, int):
            return [r, c]
        values = shape.get("shape")
        if isinstance(values, (list, tuple)) and len(values) == 2:
            try:
                return [int(values[0]), int(values[1])]
            except (TypeError, ValueError):
                return [0, 0]
    return [0, 0]


def _dedupe_preserve(items: list[str]) -> list[str]:
    """
    Deduplicate items while preserving original order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_sections(plan_text: str) -> dict[str, list[str]]:
    """
    Split the strategist plan into five numbered sections.
    """
    sections = {
        "drop": [],
        "encode": [],
        "transform": [],
        "interaction": [],
        "final": [],
    }
    current: str | None = None

    for raw in plan_text.splitlines():
        line = raw.strip()
        if not line:
            continue

        up = line.upper()
        if re.match(r"^(SECTION\s+)?1[.)]?\s*.*DROP", up):
            current = "drop"
            continue
        if re.match(r"^(SECTION\s+)?2[.)]?\s*.*ENCODING", up):
            current = "encode"
            continue
        if re.match(r"^(SECTION\s+)?3[.)]?\s*.*SKEWNESS", up):
            current = "transform"
            continue
        if re.match(r"^(SECTION\s+)?4[.)]?\s*.*INTERACTION", up):
            current = "interaction"
            continue
        if re.match(r"^(SECTION\s+)?5[.)]?\s*.*FINAL", up):
            current = "final"
            continue
        if current:
            if line.startswith(("#", "=")) and "SECTION" in up:
                current = None
                continue
            sections[current].append(line)

    return sections


def _sanitize_token(token: str | None) -> str:
    """
    Strip numbering, bullets, punctuation, and bracketed explanations.
    """
    if not token:
        return ""
    txt = token.strip()
    txt = txt.lstrip("-•* ").strip()
    txt = re.sub(r"^\d+\s*[.)]?\s*", "", txt)
    txt = re.sub(r"\s*\[.*?\]\s*", "", txt)
    txt = txt.split(" ", 1)[0]
    txt = txt.strip("`\"'“”")
    return txt


def _left_token(line: str) -> str:
    """
    Extract feature name from the left side of a plan bullet line.
    """
    txt = line.strip()
    for sep in ("→", "->", ":", "-"):
        if sep in txt:
            return _sanitize_token(txt.split(sep, 1)[0])
    return _sanitize_token(txt)


def _right_token(line: str) -> str:
    """
    Extract created feature name from an interaction bullet.
    """
    txt = line.strip()
    for sep in ("→", "->", "=", ":"):
        if sep in txt:
            return _sanitize_token(txt.split(sep, 1)[1])
    return _sanitize_token(txt)


def _parse_plan_features(plan_text: str) -> dict[str, list[str]]:
    """
    Parse FE plan sections into deduplicated feature lists.
    """
    sections = _extract_sections(plan_text)
    dropped = [_left_token(line) for line in sections["drop"]]
    encoded = [_left_token(line) for line in sections["encode"]]
    transformed = [_left_token(line) for line in sections["transform"]]
    created = []
    for line in sections["interaction"]:
        created.append(_right_token(line))

    return {
        "features_dropped": _dedupe_preserve([x for x in dropped if x]),
        "features_encoded": _dedupe_preserve([x for x in encoded if x]),
        "features_transformed": _dedupe_preserve([x for x in transformed if x]),
        "features_created": _dedupe_preserve([x for x in created if x]),
    }


def _read_csv_header(path: str) -> tuple[list[str], list[int]]:
    """
    Read only CSV header and row count without loading full file.
    """
    if not os.path.isfile(path):
        return [], []
    with open(path, newline="") as file:
        reader = csv.reader(file)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return list(header or []), [rows, len(header)] if header is not None else [0, 0]


def _read_summary(path: str) -> dict[str, Any]:
    """
    Load feature_summary.json if it exists.
    """
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as file:
            return json.load(file)
    except Exception:
        return {}


async def _run_analyzer_from_stats(
    df_stats: dict[str, Any],
    target_col: str,
) -> str:
    """
    Analyzer fallback when full records are not available.
    Sends df_stats directly to the analyzer agent.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    runner = InMemoryRunner(agent=fe_analyzer_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="system",
    )
    payload = json.dumps(
        {
            "df_stats": df_stats,
            "target_col": target_col,
        },
        default=str,
    )
    report = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=payload)],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            report = event.content.parts[0].text or ""
    return report.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline runner — called by /api/fe or the master pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def run_fe_orchestrator(
    df_stats: dict[str, Any],
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Run FE pipeline in strict 3-phase order:
    Analyze -> Strategize -> Implement -> Report.

    Args:
        df_stats:        Precomputed feature stats for df_clean.
        target_col:      ML target column name.
        numeric_cols:    Numeric columns in the cleaned dataframe.
        categorical_cols: Categorical columns in the cleaned dataframe.
        records:         Full cleaned rows; used by analyzer+implementor.

    Returns:
        Structured JSON payload for downstream model-training consumers.
    """
    start = time.monotonic()
    log_lines: list[str] = []
    errors: list[str] = []

    def _log(msg: str) -> None:
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    record_rows: list[dict[str, Any]] = records or []
    if not record_rows and isinstance(df_stats, dict):
        cached_rows = df_stats.get("records")
        if isinstance(cached_rows, list):
            record_rows = cached_rows

    input_shape = _normalize_shape(
        df_stats.get("shape")
        if isinstance(df_stats, dict) and "shape" in df_stats
        else [df_stats.get("rows", 0), df_stats.get("columns", 0)]
        if isinstance(df_stats, dict)
        else [0, 0]
    )
    if input_shape == [0, 0] and record_rows:
        input_shape = [len(record_rows), len(record_rows[0]) if record_rows else 0]

    resolved_numeric = numeric_cols[:] or []
    resolved_categorical = categorical_cols[:] or []
    if not resolved_numeric and isinstance(df_stats, dict):
        resolved_numeric = list(df_stats.get("numeric_cols", []))
    if not resolved_categorical and isinstance(df_stats, dict):
        resolved_categorical = list(df_stats.get("categorical_cols", []))

    if not resolved_numeric and not resolved_categorical and record_rows:
        resolved_columns = list(record_rows[0].keys())
        # best-effort: if analyzer hasn't supplied types, infer from dtypes map
        if isinstance(df_stats, dict):
            dtypes = df_stats.get("dtypes")
            if isinstance(dtypes, dict):
                resolved_numeric = [
                    c for c, t in dtypes.items() if str(t).lower() in {"numeric", "int64", "float64", "number"}
                ]
                resolved_categorical = [
                    c for c in resolved_columns if c not in resolved_numeric
                ]
            else:
                resolved_numeric = resolved_columns
                resolved_categorical = []

    column_names: list[str] = []
    if record_rows:
        column_names = list(record_rows[0].keys())
    elif isinstance(df_stats, dict):
        dtypes = df_stats.get("dtypes")
        if isinstance(dtypes, dict):
            column_names = list(dtypes.keys())
        if not column_names:
            columns_meta = df_stats.get("columns")
            if isinstance(columns_meta, list):
                column_names = columns_meta
            elif isinstance(dtypes, dict):
                column_names = list(dtypes.keys())

    if not column_names and resolved_numeric:
        column_names = list(dict.fromkeys(resolved_numeric + resolved_categorical))
    if not column_names and df_stats.get("columns") and isinstance(
        df_stats["columns"], list
    ):
        column_names = df_stats["columns"]

    validation = validate_fe_input(
        target_col=target_col,
        record_count=input_shape[0],
        column_names=column_names,
    )
    if not validation["valid"]:
        return {
            "status": "error",
            "input_shape": input_shape,
            "output_shape": [],
            "features_dropped": [],
            "features_encoded": [],
            "features_transformed": [],
            "features_created": [],
            "final_features": [],
            "target_col": target_col,
            "output_files": {
                "engineered_csv": "outputs/engineered_data.csv",
                "feature_summary": "outputs/feature_summary.json",
            },
            "errors": validation["errors"],
            "pipeline_log": f"Validation failed: {validation['summary']}",
        }

    _log("FE validation passed.")

    # Phase 1 — Analyze
    _log("Phase 1/3 — Analyzer starting.")
    analyzer_report: str = ""
    try:
        if record_rows:
            analyzer_report = await run_fe_analyzer(
                records=record_rows,
                target_col=target_col,
            )
        else:
            analyzer_report = await _run_analyzer_from_stats(
                df_stats=df_stats,
                target_col=target_col,
            )
        if not analyzer_report:
            raise RuntimeError("Empty analyzer report.")
        _log("Phase 1/3 — Analyzer done ✓")
    except Exception as exc:
        msg = f"FE Analyzer failed: {exc}"
        errors.append(msg)
        _log(f"Phase 1/3 — ERROR: {msg}")
        return {
            "status": "error",
            "input_shape": input_shape,
            "output_shape": [],
            "features_dropped": [],
            "features_encoded": [],
            "features_transformed": [],
            "features_created": [],
            "final_features": [],
            "target_col": target_col,
            "output_files": {
                "engineered_csv": "outputs/engineered_data.csv",
                "feature_summary": "outputs/feature_summary.json",
            },
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
        }

    # Phase 2 — Strategize
    _log("Phase 2/3 — Strategist starting.")
    strategist_output: dict[str, Any]
    try:
        strategist_output = await run_fe_strategist(
            analyzer_report=analyzer_report,  # backward-compatible kw accepted by run_fe_strategist
            target_col=target_col,
            numeric_cols=resolved_numeric,
            categorical_cols=resolved_categorical,
        )
        if not isinstance(strategist_output, dict):
            raise TypeError("Invalid strategist output.")
        plan_text = strategist_output.get("feature_plan", "")
        if not isinstance(plan_text, str) or not plan_text:
            raise ValueError("Strategist plan is empty.")
        _log("Phase 2/3 — Strategist done ✓")
    except Exception as exc:
        msg = f"FE Strategist failed: {exc}"
        errors.append(msg)
        _log(f"Phase 2/3 — ERROR: {msg}")
        return {
            "status": "error",
            "input_shape": input_shape,
            "output_shape": [],
            "features_dropped": [],
            "features_encoded": [],
            "features_transformed": [],
            "features_created": [],
            "final_features": [],
            "target_col": target_col,
            "output_files": {
                "engineered_csv": "outputs/engineered_data.csv",
                "feature_summary": "outputs/feature_summary.json",
            },
            "analysis_report": analyzer_report,
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
        }

    # Phase 3 — Implement (retry once on failure)
    _log("Phase 3/3 — Implementor starting.")
    if not record_rows:
        msg = "Phase 3 blocked: no records available for implementation."
        errors.append(msg)
        _log(f"Phase 3/3 — ERROR: {msg}")
        return {
            "status": "error",
            "input_shape": input_shape,
            "output_shape": [],
            "features_dropped": [],
            "features_encoded": [],
            "features_transformed": [],
            "features_created": [],
            "final_features": [],
            "target_col": target_col,
            "output_files": {
                "engineered_csv": "outputs/engineered_data.csv",
                "feature_summary": "outputs/feature_summary.json",
            },
            "analysis_report": analyzer_report,
            "strategy": strategist_output,
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
        }

    implementor_output: dict[str, Any] | None = None
    impl_errors: list[str] = []
    for attempt in range(1, 3):
        try:
            implementor_output = await run_fe_implementor(
                records=record_rows,
                strategist_output=strategist_output,
            )
            _log(
                f"Phase 3/3 — Implementor attempt {attempt} done ✓ "
                f"{implementor_output.get('output_file', 'unknown')}"
            )
            break
        except Exception as exc:
            err = f"FE Implementor attempt {attempt} failed: {exc}"
            impl_errors.append(err)
            errors.append(err)
            _log(f"Phase 3/3 — {err}")
            if attempt == 1:
                _log("Phase 3/3 — retrying implementor once.")

    if not implementor_output:
        elapsed = round(time.monotonic() - start, 2)
        return {
            "status": "error",
            "input_shape": input_shape,
            "output_shape": [],
            "features_dropped": [],
            "features_encoded": [],
            "features_transformed": [],
            "features_created": [],
            "final_features": [],
            "target_col": target_col,
            "output_files": {
                "engineered_csv": "outputs/engineered_data.csv",
                "feature_summary": "outputs/feature_summary.json",
            },
            "analysis_report": analyzer_report,
            "strategy": strategist_output,
            "implementation": None,
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
            "elapsed_seconds": elapsed,
        }

    # Phase 4 style report payload
    final_cols: list[str]
    final_shape: list[int]
    final_cols, final_shape = _read_csv_header("outputs/engineered_data.csv")
    if not final_cols:
        summary = _read_summary("outputs/feature_summary.json")
        numeric_features = summary.get("numeric_features", [])
        categorical_features = summary.get("categorical_features", [])
        if isinstance(numeric_features, list) and isinstance(categorical_features, list):
            final_cols = [c for c in (list(numeric_features) + list(categorical_features)) if c]
        final_shape = summary.get("shape", [])
        if isinstance(final_shape, list) and len(final_shape) == 2:
            final_shape = [int(final_shape[0]), int(final_shape[1])]
        else:
            final_shape = []

    plan_lists = _parse_plan_features(strategist_output.get("feature_plan", ""))
    final_features = [c for c in final_cols if c != target_col]
    status = "success"

    if final_shape == []:
        status = "error"
        errors.append("Missing outputs/engineered_data.csv file.")

    if target_col not in final_cols:
        status = "error"
        errors.append(f"target_col '{target_col}' missing from engineered output.")

    # Validate transformed/encoded/create candidates against final output where
    # we can do a direct check.
    for fld in (
        plan_lists.get("features_encoded", [])
        + plan_lists.get("features_transformed", [])
        + plan_lists.get("features_created", [])
    ):
        if not fld or fld == target_col:
            continue
        if fld not in final_cols and status == "success":
            errors.append(f"Planned feature '{fld}' not found in engineered columns.")
            status = "error"

    return {
        "status": status,
        "input_shape": input_shape,
        "output_shape": final_shape,
        "features_dropped": _dedupe_preserve([
            c for c in plan_lists.get("features_dropped", []) if c and c != target_col
        ]),
        "features_encoded": _dedupe_preserve([
            c for c in plan_lists.get("features_encoded", []) if c and c != target_col
        ]),
        "features_transformed": _dedupe_preserve([
            c for c in plan_lists.get("features_transformed", []) if c and c != target_col
        ]),
        "features_created": _dedupe_preserve([
            c for c in plan_lists.get("features_created", []) if c
        ]),
        "final_features": final_features,
        "target_col": target_col,
        "output_files": {
            "engineered_csv": "outputs/engineered_data.csv",
            "feature_summary": "outputs/feature_summary.json",
        },
        "analysis_report": analyzer_report,
        "strategy": strategist_output,
        "implementation": implementor_output,
        "errors": errors + impl_errors,
        "pipeline_log": "\n".join(log_lines),
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }
