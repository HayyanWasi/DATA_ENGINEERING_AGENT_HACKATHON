"""
app/agents/eda_orchestrator_agent.py

EDA Orchestrator Agent — top-level manager of the EDA pipeline.

Architecture:
  ┌──────────────────────────────────────────────────────────────────┐
  │                      EDA Orchestrator                            │
  │  (4-phase pipeline: Analyze → Strategize → Implement → Report)  │
  └──────────┬──────────────────┬──────────────────┬────────────────┘
             │                  │                  │
             ▼                  ▼                  ▼
     EDA Analyzer Agent  EDA Strategist Agent  EDA Implementor Agent
     (profile df_stats)  (plan chart types)   (generate PNGs + JSON)

Phase 4 returns structured JSON consumed directly by the FastAPI endpoint
which serves it to the Next.js frontend.
"""

from __future__ import annotations

import os
import time
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.agents.eda_agent.analyzer_agent import (
    eda_analyzer_agent,
    run_eda_analyzer,
)
from app.agents.eda_agent.strategist_agent import (
    eda_strategist_agent,
    run_eda_strategist,
)
from app.agents.eda_agent.executor_agent import (
    eda_implementor_agent,
    run_eda_implementor,
)
from app.tools.eda_analysis_tools import compute_eda_stats
from app.tools.executor_tools import init_eda_sandbox, verify_charts_saved

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator-owned tools
# ─────────────────────────────────────────────────────────────────────────────

def validate_eda_input(
    dataset_id: str,
    target_col: str,
    record_count: int,
    column_names: list[str],
) -> dict[str, Any]:
    """
    Validate EDA pipeline inputs before running any child agents.

    Args:
        dataset_id:    UUID string of the dataset.
        target_col:    Name of the ML target column.
        record_count:  Total number of rows in df_clean.
        column_names:  List of column headers from df_clean.

    Returns:
        dict with keys: valid (bool), errors (list), summary (str)
    """
    errors: list[str] = []

    if not dataset_id or not dataset_id.strip():
        errors.append("dataset_id is empty or missing.")

    if record_count < 1:
        errors.append("df_clean has no rows — cannot run EDA on empty data.")

    if not column_names:
        errors.append("df_clean has no columns.")
    elif target_col and target_col not in column_names:
        errors.append(
            f"target_col '{target_col}' not found in df_clean columns: {column_names}."
        )

    valid = len(errors) == 0
    summary = (
        f"EDA validation passed. Dataset '{dataset_id}' has {record_count} rows, "
        f"{len(column_names)} columns. Target column '{target_col}' found."
        if valid
        else f"EDA validation FAILED: {'; '.join(errors)}"
    )
    return {"valid": valid, "errors": errors, "summary": summary}


def eda_pipeline_status(
    stage: str,
    status: str,
    message: str,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    """Report the current EDA pipeline phase for observability."""
    return {
        "stage": stage,
        "status": status,
        "message": message,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


_ORCHESTRATOR_TOOLS: list[FunctionTool] = [
    FunctionTool(validate_eda_input),
    FunctionTool(eda_pipeline_status),
]

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_ORCHESTRATOR_PROMPT = """
You are the EDA (Exploratory Data Analysis) Orchestrator.
You manage a 3-agent pipeline to analyze cleaned data and generate
visualizations for display in the UI and inclusion in the final report.

You have 3 sub-agents: eda_analyzer_agent, eda_strategist_agent, eda_implementor_agent

You will receive:
- df_stats: statistics computed from BOTH df (raw) and df_clean (cleaned)
- target_col: ML target column name
- numeric_cols: list of numeric columns
- categorical_cols: list of categorical columns

## YOUR EXACT WORKFLOW:

### PHASE 1 — ANALYZE
Call eda_analyzer_agent with complete df_stats.
Pass both raw and cleaned data statistics so analyzer can compare.
Wait for complete 6-section analysis report before Phase 2.

### PHASE 2 — STRATEGIZE
Call eda_strategist_agent with:
- Full analysis report from Phase 1
- target_col, numeric_cols, categorical_cols
Wait for complete visualization plan with exact filenames before Phase 3.

### PHASE 3 — IMPLEMENT
Call eda_implementor_agent with:
- Full visualization plan from Phase 2
- Reminder: df = raw data, df_clean = cleaned data, both in sandbox
Wait for confirmation that all charts and eda_stats.json are saved.

### PHASE 4 — REPORT
After all 3 phases complete, return structured JSON result:

{
  "status": "success",
  "charts": {
    "distributions": ["charts/distribution_age.png", ...],
    "correlation": "charts/correlation_heatmap.png",
    "target": "charts/target_distribution.png",
    "boxplots": ["charts/boxplot_age.png", ...],
    "categoricals": ["charts/categorical_department.png", ...],
    "comparisons": ["charts/comparison_age.png", ...]
  },
  "stats_file": "charts/eda_stats.json",
  "insights": [
    "salary and experience highly correlated: 0.89",
    "Target is imbalanced: 80% class 0, 20% class 1",
    "age column right skewed: skewness 1.4"
  ],
  "before_after_available": true
}

## STRICT RULES
- Never skip a phase
- Never call Phase 3 before Phase 2 is complete
- Always pass COMPLETE output between phases — never summarize
- If any chart fails — continue pipeline, note failure in report
- before_after comparison charts are MANDATORY for UI
- Return structured JSON in Phase 4 — Next.js depends on this exact format
"""

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Agent definition
# ─────────────────────────────────────────────────────────────────────────────

eda_orchestrator_agent = Agent(
    name="eda_orchestrator_agent",
    model="gemini-2.0-flash",
    description=(
        "4-phase EDA pipeline manager. Computes dual raw/clean stats, delegates "
        "to Analyzer → Strategist → Implementor, then returns structured JSON "
        "consumed by the FastAPI/Next.js frontend."
    ),
    instruction=_ORCHESTRATOR_PROMPT,
    tools=_ORCHESTRATOR_TOOLS,
    sub_agents=[
        eda_analyzer_agent,
        eda_strategist_agent,
        eda_implementor_agent,
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 builder — builds the structured JSON result from charts/ directory
# ─────────────────────────────────────────────────────────────────────────────

def _build_phase4_result(
    df_stats: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    """
    Scan the charts/ directory and build the Phase 4 structured JSON result.

    The exact shape of this dict is consumed by the FastAPI endpoint
    and passed to the Next.js frontend — do not rename keys.
    """
    charts_dir = "charts"
    all_files: list[str] = []
    if os.path.isdir(charts_dir):
        all_files = sorted(os.listdir(charts_dir))

    def _chart(name: str) -> str:
        return f"{charts_dir}/{name}"

    distributions  = [_chart(f) for f in all_files if f.startswith("distribution_") and f.endswith(".png")]
    boxplots       = [_chart(f) for f in all_files if f.startswith("boxplot_")       and f.endswith(".png")]
    categoricals   = [_chart(f) for f in all_files if f.startswith("categorical_")   and f.endswith(".png")]
    comparisons    = [_chart(f) for f in all_files if f.startswith("comparison_")    and f.endswith(".png")]
    correlation    = _chart("correlation_heatmap.png") if "correlation_heatmap.png" in all_files else None
    target_chart   = _chart("target_distribution.png") if "target_distribution.png" in all_files else None
    pairplot       = _chart("pairplot.png")             if "pairplot.png"            in all_files else None
    stats_file     = _chart("eda_stats.json")           if "eda_stats.json"          in all_files else None

    # Build insights from clean stats
    clean: dict[str, Any] = df_stats.get("clean", df_stats)
    insights: list[str] = []

    # High correlations
    for pair in clean.get("correlation_pairs", []):
        r = pair.get("correlation", 0.0)
        if abs(r) >= 0.7:
            direction = "positively" if r > 0 else "negatively"
            strength = "strongly" if abs(r) >= 0.9 else "moderately"
            insights.append(
                f"{pair['col1']} and {pair['col2']} {strength} {direction} "
                f"correlated: r={r}"
            )
        if len(insights) >= 3:
            break

    # Class imbalance
    if clean.get("is_imbalanced"):
        cb = clean.get("class_balance", {})
        balance_str = ", ".join(f"{k}: {v}%" for k, v in cb.items())
        insights.append(f"Target is imbalanced: {balance_str}")

    # Skewed columns
    num_stats = clean.get("numeric_stats", {})
    for col in clean.get("skewed_cols", [])[:3]:
        skew = num_stats.get(col, {}).get("skewness", 0.0)
        direction = "right" if skew > 0 else "left"
        insights.append(f"{col} column {direction}-skewed: skewness {skew}")

    # Outlier-heavy columns
    for col in clean.get("outlier_cols", [])[:2]:
        pct = num_stats.get(col, {}).get("outlier_pct", 0.0)
        cnt = num_stats.get(col, {}).get("outlier_count", 0)
        insights.append(f"{col} has {cnt} outliers ({pct}%)")

    return {
        "status": "success" if not errors else "partial",
        "charts": {
            "distributions": distributions,
            "correlation": correlation,
            "target": target_chart,
            "boxplots": boxplots,
            "categoricals": categoricals,
            "comparisons": comparisons,
            "pairplot": pairplot,
        },
        "stats_file": stats_file,
        "insights": insights[:8],
        "before_after_available": len(comparisons) > 0,
        "total_charts": len(distributions) + len(boxplots) + len(categoricals)
            + len(comparisons) + (1 if correlation else 0)
            + (1 if target_chart else 0) + (1 if pairplot else 0),
        "errors": errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline runner — called by /api/eda or the master pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def run_eda_pipeline(
    dataset_id: str,
    records: list[dict[str, Any]],
    target_col: str,
    raw_records: list[dict[str, Any]] | None = None,
    skip_implementor: bool = False,
) -> dict[str, Any]:
    """
    Run the full 4-phase EDA pipeline.

    IMPORTANT: records must be df_clean rows (cleaned data). Call this AFTER
    the data cleaning pipeline has completed and outputs/cleaned_data.csv exists.

    Args:
        dataset_id:       UUID string of the dataset.
        records:          Cleaned rows as a list of plain dicts.
        target_col:       Name of the ML target column.
        raw_records:      Original uncleaned rows (for before/after comparison).
                          Falls back to records if not supplied.
        skip_implementor: If True, stop after the Strategist (no chart generation).

    Returns:
        Phase 4 structured JSON result (see _build_phase4_result) plus pipeline
        metadata: dataset_id, target_col, stages_completed, elapsed_seconds,
        analysis, strategy, execution, pipeline_log.
    """
    pipeline_start = time.monotonic()
    errors: list[str] = []
    log_lines: list[str] = []
    stages_completed = 0

    def _log(msg: str) -> None:
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # ── Pre-flight validation ────────────────────────────────────────────────
    if not records:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "target_col": target_col,
            "errors": ["df_clean has no records — aborting EDA pipeline."],
            "pipeline_log": "EDA pipeline aborted: empty dataset.",
        }

    column_names = list(records[0].keys())
    validation = validate_eda_input(
        dataset_id=dataset_id,
        target_col=target_col,
        record_count=len(records),
        column_names=column_names,
    )
    if not validation["valid"]:
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "target_col": target_col,
            "errors": validation["errors"],
            "pipeline_log": f"EDA validation failed: {validation['summary']}",
        }

    _log(f"EDA validation passed — {len(records)} rows, {len(column_names)} columns.")

    # ── Pre-compute dual-dataset stats (raw + clean) ──────────────────────────
    _log("Computing df_stats from raw + clean records…")
    effective_raw = raw_records if raw_records else records
    df_stats = compute_eda_stats(
        raw_records=effective_raw,
        clean_records=records,
        target_col=target_col,
    )
    numeric_cols: list[str]     = df_stats["numeric_cols"]
    categorical_cols: list[str] = df_stats["categorical_cols"]
    _log(
        f"df_stats ready — {len(numeric_cols)} numeric cols, "
        f"{len(categorical_cols)} categorical cols."
    )

    stages_planned = 2 if skip_implementor else 3

    # ── PHASE 1 — EDA Analyzer ───────────────────────────────────────────────
    _log("EDA Phase 1/3 — Analyzer Agent starting…")
    analyzer_output: dict[str, Any] | None = None
    try:
        analyzer_output = await run_eda_analyzer(
            dataset_id=dataset_id,
            df_stats=df_stats,
            target_col=target_col,
        )
        stages_completed += 1
        _log("EDA Phase 1/3 — Analyzer Agent done ✓")
    except Exception as exc:
        msg = f"EDA Analyzer Agent failed: {exc}"
        errors.append(msg)
        _log(f"EDA Phase 1/3 — ERROR: {msg}")
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "target_col": target_col,
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
        }

    # ── PHASE 2 — EDA Strategist ─────────────────────────────────────────────
    _log("EDA Phase 2/3 — Strategist Agent starting…")
    strategist_output: dict[str, Any] | None = None
    try:
        strategist_output = await run_eda_strategist(
            analyzer_output=analyzer_output,
            target_col=target_col,
        )
        stages_completed += 1
        _log("EDA Phase 2/3 — Strategist Agent done ✓")
    except Exception as exc:
        msg = f"EDA Strategist Agent failed: {exc}"
        errors.append(msg)
        _log(f"EDA Phase 2/3 — ERROR: {msg}")
        return {
            "status": "error",
            "dataset_id": dataset_id,
            "target_col": target_col,
            "stages_completed": stages_completed,
            "analysis": analyzer_output,
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
        }

    # ── PHASE 3 — EDA Implementor (optional) ────────────────────────────────
    implementor_output: dict[str, Any] | None = None

    if skip_implementor:
        _log("EDA Phase 3/3 — Implementor SKIPPED (skip_implementor=True).")
    else:
        # Init sandbox: df = raw, df_clean = cleaned, column lists injected
        _log("EDA Phase 3/3 — Initialising EDA sandbox…")
        sandbox_msg = init_eda_sandbox(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        _log(f"EDA Sandbox: {sandbox_msg}")

        if sandbox_msg.startswith("ERROR:"):
            errors.append(sandbox_msg)
            _log(f"EDA Phase 3/3 — Sandbox init failed.")
        else:
            _log("EDA Phase 3/3 — Implementor Agent starting…")
            try:
                implementor_output = await run_eda_implementor(
                    strategist_output=strategist_output,
                )
                stages_completed += 1

                charts_result = verify_charts_saved()
                _log(
                    f"EDA Phase 3/3 — Implementor done ✓  "
                    f"{charts_result.get('chart_count', 0)} files in charts/."
                )
            except Exception as exc:
                msg = f"EDA Implementor Agent failed: {exc}"
                errors.append(msg)
                _log(f"EDA Phase 3/3 — ERROR: {msg}")

    elapsed = round(time.monotonic() - pipeline_start, 2)
    _log(
        f"EDA pipeline finished — {stages_completed}/{stages_planned} phases "
        f"completed in {elapsed}s."
    )

    # ── PHASE 4 — Build structured JSON result ───────────────────────────────
    phase4 = _build_phase4_result(df_stats=df_stats, errors=errors)

    # Merge pipeline metadata into the Phase 4 result
    return {
        **phase4,
        "dataset_id": dataset_id,
        "target_col": target_col,
        "stages_attempted": stages_planned,
        "stages_completed": stages_completed,
        "elapsed_seconds": elapsed,
        # Sub-agent outputs for debugging / downstream use
        "analysis": analyzer_output,
        "strategy": strategist_output,
        "implementation": implementor_output,
        "pipeline_log": "\n".join(log_lines),
    }
