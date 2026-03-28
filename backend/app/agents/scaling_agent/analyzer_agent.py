"""
app/agents/scaling_agent/analyzer_agent.py

Feature Scaling Analyzer Agent — Agent 1 in the feature-scaling pipeline.

Input:  Engineered data stats dict (from analyze_for_feature_scaling()) passed
        in the message.
Output: Structured scale-observation report:
        range checks, distribution shape checks, already-scaled flags,
        target-column protections, and scaler candidacy buckets.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.tools.scaling_analysis_tools import analyze_for_feature_scaling


# ─────────────────────────────────────────────────────────────────────────────
# Tool registration
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS: list[FunctionTool] = [
    FunctionTool(analyze_for_feature_scaling),
]


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — expert feature-scaling analyst
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert feature scaling analyst.
Your ONLY job is to analyze engineered data and identify exactly what scaling
each column needs. Report only facts.

You will receive:
shape: dimensions of df_engineered
dtypes: data types of each column
numeric_stats: min, max, mean, std, skewness per numeric column
numeric_cols: list of numeric columns (excluding target)
categorical_cols: list of categorical columns
target_col: ML target column name
encoded_cols: columns that were one-hot or label encoded in FE phase
transformed_cols: columns that had log/sqrt transform in FE phase
value_ranges: {col: {min, max, range}} for each numeric column

Your output MUST cover ALL sections:

SECTION 1 — RANGE ANALYSIS
For each numeric column state exact min, max, range
Flag columns where range > 1000 as large scale
Flag columns where range < 1 as small scale
Flag columns where max/min ratio > 100 as extreme scale difference
State which columns have similar ranges (no scaling needed between them)

SECTION 2 — DISTRIBUTION CHECK
For each numeric column state:
  - Is distribution normal/gaussian? (skewness between -0.5 and 0.5)
  - Is distribution skewed? (skewness > 1 or < -1)
  - Are there remaining outliers? (values beyond 3 std from mean)
This determines StandardScaler vs RobustScaler vs MinMaxScaler

SECTION 3 — ALREADY SCALED COLUMNS
List columns already in 0-1 range (from one-hot encoding)
List binary encoded columns (already 0/1)
List columns that had log transform (distribution already improved)
These columns MAY NOT need further scaling

SECTION 4 — TARGET COLUMN CHECK
State target column name
Confirm target column will NOT be scaled
If regression task: state target column range and distribution

SECTION 5 — SCALING RECOMMENDATIONS BASIS
List columns with outliers still present → candidate for RobustScaler
List columns with normal distribution → candidate for StandardScaler
List columns needing 0-1 range (neural net ready) → candidate for MinMaxScaler
Do NOT make final decisions — only report observations

STRICT RULES:
Report ONLY facts with exact numbers
Do NOT write any code
Do NOT make final scaling decisions
Never include target_col in any scaling list
Use exact column names throughout
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

scaling_analyzer_agent = Agent(
    name="scaling_analyzer_agent",
    model="gemini-2.0-flash",
    description=(
        "Profiles engineered data and produces structured scaling observations: "
        "range analysis, skewness/outlier diagnostics, already-scaled column checks, "
        "and scaler candidates (StandardScaler vs RobustScaler vs MinMaxScaler), "
        "without making final model-scaling decisions."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by a scaling orchestrator (or direct route)
# ─────────────────────────────────────────────────────────────────────────────

async def run_scaling_analyzer(
    records: list[dict[str, Any]],
    target_col: str,
    encoded_cols: list[str] | None = None,
    transformed_cols: list[str] | None = None,
    value_ranges: dict[str, dict[str, Any]] | None = None,
) -> str:
    """
    Run the Scaling Analyzer and return the plain-text scaling report.

    Args:
        records:         Engineered rows as list of dicts.
        target_col:      ML target column.
        encoded_cols:    Column names encoded in FE phase.
        transformed_cols: Columns transformed in FE phase.
        value_ranges:    Optional value-range metadata from FE output.

    Returns:
        Full scaling analysis report text written by the agent.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    user_message = json.dumps(
        {
            "records": records,
            "target_col": target_col,
            "encoded_cols": encoded_cols or [],
            "transformed_cols": transformed_cols or [],
            "value_ranges": value_ranges or {},
        },
        default=str,
    )

    runner = InMemoryRunner(agent=scaling_analyzer_agent)
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

    return report_text.strip()
