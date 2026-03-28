"""
app/agents/data_orchestrator_agent.py

Data Orchestrator Agent — the top-level manager of the entire
data-cleaning pipeline.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                   Orchestrator Agent                        │
  │  (routes tasks, tracks state, decides what runs next)       │
  └──────────┬──────────────┬────────────────┬─────────────────┘
             │              │                │
             ▼              ▼                ▼
     Analyzer Agent  Strategist Agent  Executor Agent
     (profile data)  (plan cleaning)  (run the code)

Responsibilities:
  1. Receive a pipeline request (dataset_id, target_column, records)
  2. Delegate to Analyzer  → get analysis report + stats
  3. Pass analysis to Strategist → get numbered cleaning strategy
  4. Pass strategy + records to Executor → get cleaned CSV + log
  5. Aggregate all outputs into a single PipelineResult
  6. Handle failures: if any child fails, surface a clear error
     and skip downstream steps gracefully

The Orchestrator itself has TWO tools:
  - pipeline_status(...)  → reports current pipeline stage
  - validate_pipeline_input(...) → checks dataset_id + records before firing
"""

from __future__ import annotations

import json
import time
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.agents.datacleaning_agent.analyzer_agent import (
    analyzer_agent,
    run_analyzer,
)
from app.agents.datacleaning_agent.strategist_agent import (
    strategist_agent,
    run_strategist,
)
from app.agents.datacleaning_agent.executor_agent import (
    executor_agent,
    run_executor,
)

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator-owned tools
# ─────────────────────────────────────────────────────────────────────────────

def validate_pipeline_input(
    dataset_id: str,
    target_column: str,
    record_count: int,
    column_names: list[str],
) -> dict[str, Any]:
    """
    Validate the pipeline inputs before kicking off any child agents.

    Checks:
      - dataset_id is a non-empty string
      - target_column is present in column_names
      - record_count >= 1 (non-empty dataset)

    Args:
        dataset_id:    UUID string of the dataset.
        target_column: Name of the ML prediction target column.
        record_count:  Total number of rows in the dataset.
        column_names:  List of column headers from the dataset.

    Returns:
        dict with keys:
          valid   (bool)   — True if all checks pass
          errors  (list)   — list of validation error messages (empty if valid)
          summary (str)    — human-readable validation summary
    """
    errors: list[str] = []

    if not dataset_id or not dataset_id.strip():
        errors.append("dataset_id is empty or missing.")

    if record_count < 1:
        errors.append("Dataset has no rows — cannot run pipeline on empty data.")

    if not column_names:
        errors.append("Dataset has no columns.")
    elif target_column not in column_names:
        errors.append(
            f"target_column '{target_column}' not found in dataset columns: "
            f"{column_names}."
        )

    valid = len(errors) == 0
    summary = (
        f"Validation passed. Dataset '{dataset_id}' has {record_count} rows, "
        f"{len(column_names)} columns. Target column '{target_column}' found."
        if valid
        else f"Validation FAILED: {'; '.join(errors)}"
    )
    return {"valid": valid, "errors": errors, "summary": summary}


def pipeline_status(
    stage: str,
    status: str,
    message: str,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    """
    Report the current pipeline stage and status for observability.

    Args:
        stage:           Current stage name. One of:
                         'validation', 'analysis', 'strategy', 'execution', 'complete', 'error'
        status:          'running' | 'done' | 'skipped' | 'failed'
        message:         Human-readable status message.
        elapsed_seconds: Seconds elapsed since pipeline start (0 if unknown).

    Returns:
        dict with stage, status, message, elapsed_seconds, timestamp.
    """
    return {
        "stage": stage,
        "status": status,
        "message": message,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


_ORCHESTRATOR_TOOLS: list[FunctionTool] = [
    FunctionTool(validate_pipeline_input),
    FunctionTool(pipeline_status),
]

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_ORCHESTRATOR_PROMPT = """
You are the **Data Pipeline Orchestrator**. You manage a 3-agent data cleaning
pipeline and are responsible for coordinating all child agents in the correct
order, tracking status, and returning a complete PipelineResult.

You have access to two tools of your own:
  - `validate_pipeline_input` — call this FIRST to validate inputs before
    anything else runs
  - `pipeline_status`         — call this to report progress at each stage

## YOUR CHILD AGENTS
You have three sub-agents that will be invoked by the pipeline runner:
  1. **analyzer_agent**   — profiles raw data, returns analysis report
  2. **strategist_agent** — reads the report, returns cleaning strategy
  3. **executor_agent**   — implements the strategy, saves cleaned CSV

## YOUR WORKFLOW

### Phase 0 — Validate
Call `validate_pipeline_input` with the dataset details.
If valid=False, call `pipeline_status(stage='error', status='failed', ...)`
and STOP — return an error message explaining what is wrong.

### Phase 1 — Report Status: Analysis Starting
Call `pipeline_status(stage='analysis', status='running', ...)`

### Phase 2 — Report Status: Strategy Starting
After analysis completes, call:
`pipeline_status(stage='strategy', status='running', ...)`

### Phase 3 — Report Status: Execution Starting (or Skipped)
After strategy completes, call:
`pipeline_status(stage='execution', status='running', ...)` if executing.
Or `pipeline_status(stage='execution', status='skipped', ...)` if skipping.

### Phase 4 — Report Completion
Call `pipeline_status(stage='complete', status='done', ...)`
with a summary of all stages.

## FINAL OUTPUT
After all phases complete, produce a structured summary in this format:

---
## Pipeline Complete ✓

**Dataset:** <dataset_id>
**Target Column:** <target_column>

### Stage 1 — Analysis
Status: Done
<brief summary: rows, columns, key issues found>

### Stage 2 — Strategy
Status: Done
<brief summary: number of cleaning steps, main actions>

### Stage 3 — Execution
Status: Done | Skipped
<brief summary: output file, final shape if known>

### Pipeline Metadata
- Total stages: 3
- Stages completed: <N>
---

## RULES
- Always validate first — never skip Phase 0
- Always report status at each phase using pipeline_status
- If a child agent stage fails, report it clearly and skip downstream stages
- Keep your status messages concise and informative
"""

# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Agent definition (with sub_agents for ADK routing)
# ─────────────────────────────────────────────────────────────────────────────

orchestrator_agent = Agent(
    name="data_orchestrator_agent",
    model="gemini-2.5-flash",
    description=(
        "Top-level pipeline manager. Validates inputs, delegates to the Analyzer, "
        "Strategist, and Executor child agents in sequence, tracks pipeline status, "
        "and returns a complete PipelineResult."
    ),
    instruction=_ORCHESTRATOR_PROMPT,
    tools=_ORCHESTRATOR_TOOLS,
    sub_agents=[
        analyzer_agent,
        strategist_agent,
        executor_agent,
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline runner — called by /api/clean
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline(
    dataset_id: str,
    records: list[dict[str, Any]],
    target_column: str,
    skip_executor: bool = False,
) -> dict[str, Any]:
    """
    Run the full data-cleaning pipeline orchestrated by the Orchestrator Agent.

    Internally calls each child agent's run_* helper in sequence so data can
    be passed between stages (ADK sub_agent routing handles tool delegation;
    the helpers handle inter-stage data passing directly).

    Args:
        dataset_id:    UUID string of the dataset.
        records:       Raw rows as a list of plain dicts.
        target_column: Name of the ML target column (not encoded).
        skip_executor: If True, stop after the Strategist (no code execution).

    Returns:
        PipelineResult dict with keys:
          dataset_id, target_column, stages_attempted, stages_completed,
          analysis (sub-dict), strategy (sub-dict), execution (sub-dict | None),
          errors (list), pipeline_log (str)
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
            "dataset_id": dataset_id,
            "target_column": target_column,
            "stages_attempted": 0,
            "stages_completed": 0,
            "analysis": None,
            "strategy": None,
            "execution": None,
            "errors": ["Dataset has no records — aborting pipeline."],
            "pipeline_log": "Pipeline aborted: empty dataset.",
        }

    column_names = list(records[0].keys())
    validation = validate_pipeline_input(
        dataset_id=dataset_id,
        target_column=target_column,
        record_count=len(records),
        column_names=column_names,
    )
    if not validation["valid"]:
        return {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "stages_attempted": 0,
            "stages_completed": 0,
            "analysis": None,
            "strategy": None,
            "execution": None,
            "errors": validation["errors"],
            "pipeline_log": f"Validation failed: {validation['summary']}",
        }

    _log(f"Validation passed — {len(records)} rows, {len(column_names)} columns.")

    stages_planned = 2 if skip_executor else 3

    # ── Stage 1 — Analyzer ───────────────────────────────────────────────────
    _log("Stage 1/3 — Analyzer Agent starting…")
    analyzer_output: dict[str, Any] | None = None
    try:
        analyzer_output = await run_analyzer(
            dataset_id=dataset_id,
            records=records,
        )
        stages_completed += 1
        _log("Stage 1/3 — Analyzer Agent done ✓")
    except Exception as exc:
        msg = f"Analyzer Agent failed: {exc}"
        errors.append(msg)
        _log(f"Stage 1/3 — ERROR: {msg}")
        # Cannot proceed without analysis
        return {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "stages_attempted": 1,
            "stages_completed": 0,
            "analysis": None,
            "strategy": None,
            "execution": None,
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
        }

    # ── Stage 2 — Strategist ─────────────────────────────────────────────────
    _log("Stage 2/3 — Strategist Agent starting…")
    strategist_output: dict[str, Any] | None = None
    try:
        strategist_output = await run_strategist(
            analyzer_output=analyzer_output,
            target_column=target_column,
        )
        stages_completed += 1
        _log("Stage 2/3 — Strategist Agent done ✓")
    except Exception as exc:
        msg = f"Strategist Agent failed: {exc}"
        errors.append(msg)
        _log(f"Stage 2/3 — ERROR: {msg}")
        return {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "stages_attempted": 2,
            "stages_completed": stages_completed,
            "analysis": analyzer_output,
            "strategy": None,
            "execution": None,
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
        }

    # ── Stage 3 — Executor (optional) ────────────────────────────────────────
    executor_output: dict[str, Any] | None = None
    if skip_executor:
        _log("Stage 3/3 — Executor Agent SKIPPED (skip_executor=True).")
    else:
        _log("Stage 3/3 — Executor Agent starting…")
        try:
            executor_output = await run_executor(
                records=records,
                strategist_output=strategist_output,
            )
            stages_completed += 1
            _log(
                f"Stage 3/3 — Executor Agent done ✓  "
                f"Output: {executor_output.get('output_file', 'unknown')}"
            )
        except Exception as exc:
            msg = f"Executor Agent failed: {exc}"
            errors.append(msg)
            _log(f"Stage 3/3 — ERROR: {msg}")
            # Non-fatal — analysis + strategy are still valid

    elapsed = round(time.monotonic() - pipeline_start, 2)
    _log(
        f"Pipeline finished — {stages_completed}/{stages_planned} stages completed "
        f"in {elapsed}s."
    )

    return {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "stages_attempted": stages_planned,
        "stages_completed": stages_completed,
        "elapsed_seconds": elapsed,
        "analysis": analyzer_output,
        "strategy": strategist_output,
        "execution": executor_output,
        "errors": errors,
        "pipeline_log": "\n".join(log_lines),
    }
