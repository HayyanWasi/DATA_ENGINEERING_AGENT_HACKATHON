"""
app/agents/datacleaning_agent/analyzer_agent.py

Analyzer Agent — Agent 1 (Child) in the Data Cleaning Pipeline.

Input:  Raw dataset (list[dict]) + dataset_id
Tool:   analyze_dataset() — computes the full profiling dict in one call
Output: A structured analysis report covering nulls, duplicates, dtypes,
        outliers, categorical analysis, and general observations.
        No solutions, no code — only facts with exact numbers.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.tools.analysis_tools import analyze_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Tool registration
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS: list[FunctionTool] = [
    FunctionTool(analyze_dataset),
]

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — exact analyst persona from spec
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert data analyst. Your ONLY job is to analyze raw data and report what you find.

You will receive a JSON payload containing:
- dataset_id: the ID of the dataset
- records: a list of row objects (the raw data)

## YOUR WORKFLOW

1. Call the tool `analyze_dataset` with the `records` list.
   The tool returns a Python dictionary containing:
   - shape: dimensions of the dataset
   - dtypes: data types of each column
   - null_counts: number of nulls per column
   - null_percentage: percentage of nulls per column
   - duplicate_rows: total duplicate rows
   - numeric_stats: min, max, mean, median, std, IQR, outlier_count for numeric columns
   - categorical_stats: unique value counts and top frequencies for categorical columns
   - categorical_cols: list of categorical columns
   - numeric_cols: list of numeric columns
   - sample_rows: first 3 rows of data

2. Using ONLY the tool output, write your analysis report. Your output MUST cover ALL sections below.

## REQUIRED OUTPUT SECTIONS

### 1. NULL VALUES
List every column that has nulls. State exact count and percentage.
Example: "age column has 23 nulls (12.0%)"
If no nulls: state "No null values found."

### 2. DUPLICATE ROWS
State the exact duplicate row count.
Example: "14 exact duplicate rows detected (2.3% of total)."
If none: "No duplicate rows detected."

### 3. DATA TYPES
List every column and its detected type (numeric, string, boolean, datetime, empty).
Flag any mismatches — e.g. a column that looks numeric but was stored as string.

### 4. OUTLIERS
For each numeric column, report whether outliers exist using IQR fences.
State: column name, outlier count, outlier percentage, IQR lower/upper fences.
Example: "salary: 18 outliers (3.6%), IQR fences [18000, 95000]"
If no outliers in a column, skip it.

### 5. CATEGORICAL COLUMNS
For each categorical column, report:
- Exact number of unique values
- Top 5 most frequent values with counts
- Flag any suspicious values (e.g. typos, mixed case, unexpected categories)

### 6. GENERAL OBSERVATIONS
Note anything unusual: columns that are all-null, potential ID columns
(all unique values), columns with only one unique value, extreme value ranges, etc.

## STRICT RULES
- Report ONLY facts with exact numbers from the tool output
- Do NOT suggest any solutions
- Do NOT write any code
- Do NOT make assumptions — only report what the data shows
- Be specific: say "age column has 23 nulls (12%)" not "some nulls found"
- Output plain text, not JSON
"""

# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

analyzer_agent = Agent(
    name="analyzer_agent",
    model="gemini-2.5-flash",
    description=(
        "Profiles a raw dataset using the analyze_dataset tool and writes a "
        "structured, fact-only analysis report covering nulls, duplicates, "
        "dtypes, outliers, categorical columns, and general observations."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by the orchestrator / Strategist / API route
# ─────────────────────────────────────────────────────────────────────────────

async def run_analyzer(
    dataset_id: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Run the Analyzer Agent and return both the raw stats dict and the
    human-readable analysis report text.

    Args:
        dataset_id: Unique identifier for the dataset.
        records:    Raw rows as a list of plain dicts.

    Returns:
        dict with keys:
          "dataset_id"     — the passed-in id
          "stats"          — raw dict from analyze_dataset() tool
          "analysis_report"— full plain-text report written by the agent
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    # Pre-compute stats directly so we can include them in the response
    # even if the agent's text output is the primary deliverable
    stats: dict[str, Any] = analyze_dataset(records)

    user_message = json.dumps(
        {"dataset_id": dataset_id, "records": records},
        default=str,
    )

    runner = InMemoryRunner(agent=analyzer_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="system",
    )

    report_text = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            report_text = event.content.parts[0].text or ""

    return {
        "dataset_id": dataset_id,
        "stats": stats,
        "analysis_report": report_text.strip(),
    }
