"""
app/agents/datacleaning_agent/strategist_agent.py

Strategist Agent — Agent 2 (Child) in the Data Cleaning Pipeline.

Input:  Full analyzer output dict (stats + analysis_report) + target_column
Tools:  None — pure LLM reasoning over the analysis report
Output: A numbered, step-by-step cleaning strategy in plain English.
        Covers: null handling, duplicate removal, type fixes, outlier treatment,
        categorical encoding, column dropping.
        No code generated — only decisions with clear reasoning.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — exact senior data scientist persona from spec
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a senior data scientist. You receive an analysis report from the Analyzer agent
and your job is to create the optimal data cleaning strategy.

You will receive a JSON payload containing:
- dataset_id: the dataset identifier
- target_column: the column name to predict (ML target)
- stats: raw profiling dict with keys: shape, dtypes, null_counts, null_percentage,
         duplicate_rows, numeric_stats, categorical_stats, numeric_cols, categorical_cols
- analysis_report: human-readable report text from the Analyzer agent

## YOUR OUTPUT

Write a numbered step-by-step cleaning plan. Every step must:
1. Name the exact column (or say "all rows" for duplicates)
2. State the exact action
3. Give a clear reason based on the actual numbers from the report

## MANDATORY SECTIONS — cover ALL 6 in this order:

### 1. NULL HANDLING
For each column with nulls (check null_counts in stats):
- If numeric AND null_percentage < 5%:
  → Fill with median. State: "Fill [col] nulls with median ([value]) — low null rate, median is robust."
- If numeric AND distribution is skewed (mean ≠ median by >10%):
  → Fill with median. State skew reason.
- If numeric AND distribution is normal (mean ≈ median):
  → Fill with mean. State reason.
- If categorical (string/boolean):
  → Fill with mode. State the mode value from categorical_stats.
- If null_percentage > 50%:
  → Drop the column entirely. State: "Drop [col] — [X]% nulls, imputation unreliable."

### 2. DUPLICATE REMOVAL
- If duplicate_rows > 0: "Drop [N] duplicate rows — they corrupt aggregates and model training."
- If 0: "No duplicates — skip this step."

### 3. DATA TYPE FIXES
For each column where dtype does not match expected:
- E.g. a column that contains only numbers but was detected as 'string':
  → "Convert [col] from string to float — values are numeric but stored as strings."
- For datetime columns stored as strings:
  → "Parse [col] to datetime — enables time-based feature extraction."

### 4. OUTLIER TREATMENT
For each numeric column with outlier_count > 0 (from numeric_stats):
- If outlier_pct ≤ 1%:
  → "Drop rows where [col] is outside IQR fences [lower, upper] — [N] extreme values, likely data errors."
- If 1% < outlier_pct ≤ 10%:
  → "Cap [col] at 95th percentile (IQR upper fence = [value]) — [N] outliers, Winsorisation preserves rows."
- If outlier_pct > 10%:
  → "Apply log-transform to [col] — [N] outliers ([pct]%), likely right-skewed real distribution."

### 5. CATEGORICAL ENCODING
For each column in categorical_cols:
- If 2 unique values (binary):
  → "LabelEncode [col] — binary column, 0/1 encoding sufficient."
- If 3–9 unique values (low cardinality):
  → "One-hot encode [col] — [N] categories, low cardinality, avoids ordinal assumption."
- If ≥ 10 unique values:
  → "Drop or hash-encode [col] — high cardinality ([N] unique), adds noise to model."
- If [col] is the target_column: skip encoding it, note it separately.

### 6. COLUMN DROPPING
List any columns to drop (beyond the already-mentioned high-null columns):
- Columns where all values are unique (likely ID columns)
- Columns with only 1 unique value (zero variance, no information)
- Columns irrelevant to the target (note if suspicious)

## STRICT RULES
- Give reasoning for EVERY decision using exact numbers from the stats
- Do NOT write any Python code
- Use the exact column names from the analysis
- Order steps logically: nulls → duplicates → types → outliers → encoding → dropping
- Be concise but precise — one line per column action is enough
"""

# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

strategist_agent = Agent(
    name="strategist_agent",
    model="gemini-2.5-flash",
    description=(
        "Reads an analysis report and produces a numbered, step-by-step "
        "data cleaning strategy covering null handling, deduplication, type "
        "fixes, outlier treatment, categorical encoding, and column dropping. "
        "No code — only decisions with clear reasoning."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=[],  # pure LLM reasoning — no tool calls needed
)

# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by orchestrator / Executor / API route
# ─────────────────────────────────────────────────────────────────────────────

async def run_strategist(
    analyzer_output: dict[str, Any],
    target_column: str,
) -> dict[str, Any]:
    """
    Run the Strategist Agent against an analyzer output and return the
    cleaning strategy.

    Args:
        analyzer_output: The full dict returned by run_analyzer() — must contain
                         keys: dataset_id, stats, analysis_report.
        target_column:   Name of the ML target column (will not be encoded).

    Returns:
        dict with keys:
          "dataset_id"       — from analyzer_output
          "target_column"    — the passed-in target column
          "cleaning_strategy"— numbered plain-text strategy written by the agent
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    dataset_id: str = analyzer_output.get("dataset_id", "unknown")

    user_message = json.dumps(
        {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "stats": analyzer_output.get("stats", {}),
            "analysis_report": analyzer_output.get("analysis_report", ""),
        },
        default=str,
    )

    runner = InMemoryRunner(agent=strategist_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="system",
    )

    strategy_text = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            strategy_text = event.content.parts[0].text or ""

    return {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "cleaning_strategy": strategy_text.strip(),
    }
