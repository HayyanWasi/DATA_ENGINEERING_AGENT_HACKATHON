"""
app/agents/fe_agent/analyzer_agent.py

Feature Engineering Analyzer Agent — Agent 1 (Child) in the FE Pipeline.

Input:  Cleaned dataset stats dict (from analyze_for_feature_engineering()) passed
        in the message.
Output: Structured feature-engineering recommendations report:
        encoding choices, transformation flags, multicollinearity issues,
        interaction opportunities, and drop candidates. No code — only facts.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.tools.fe_analysis_tools import analyze_for_feature_engineering

# ─────────────────────────────────────────────────────────────────────────────
# Tool registration
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS: list[FunctionTool] = [
    FunctionTool(analyze_for_feature_engineering),
]

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — expert feature-engineering analyst
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert feature engineering analyst.
Your ONLY job is to analyze cleaned data and identify feature engineering
opportunities. Report only facts.

You will receive:
shape: dimensions of df_clean
dtypes: data types of each column
numeric_stats: min, max, mean, std, skewness, kurtosis per numeric column
categorical_stats: unique counts and top frequencies per categorical column
correlation_matrix: correlations between all numeric columns
target_col: ML target column name
numeric_cols: list of numeric columns
categorical_cols: list of categorical columns
variance_stats: variance per numeric column

Your output MUST cover ALL sections:

SECTION 1 — ENCODING REQUIREMENTS
For each categorical column state:
  - Exact unique value count
  - If unique values <= 2: needs Binary Encoding
  - If unique values <= 10: needs One-Hot Encoding
  - If unique values > 10: needs Label Encoding (high cardinality)
  - If column is ordinal (low/medium/high): needs Ordinal Encoding

SECTION 2 — SKEWNESS REPORT
For each numeric column state exact skewness value
Flag if skewness > 1 or < -1: needs log transform
Flag if skewness > 2 or < -2: needs sqrt or box-cox transform
Do NOT suggest solution — only report skewness values

SECTION 3 — VARIANCE REPORT
For each numeric column state exact variance
Flag columns with variance < 0.01 as near-zero variance
Flag columns that are IDs or index-like (all unique values)
These are candidates for dropping

SECTION 4 — MULTICOLLINEARITY REPORT
List all column pairs with correlation > 0.85
State exact correlation value for each pair
Flag which column in each pair is less correlated with target
That column is candidate for dropping

SECTION 5 — INTERACTION OPPORTUNITIES
Identify numeric column pairs that could produce meaningful ratios
Example: salary/experience = salary_per_year_experience
Only suggest if both columns exist and ratio makes domain sense
List maximum 3 interaction opportunities

SECTION 6 — FEATURE SUMMARY
Total features currently: X
Features to encode: list them
Features to transform: list them
Features to drop: list them
Estimated features after engineering: X

STRICT RULES:
Report ONLY facts with exact numbers
Do NOT write any code
Do NOT make final decisions — only report observations
Use exact column names throughout
"""

# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

fe_analyzer_agent = Agent(
    name="fe_analyzer_agent",
    model="gemini-2.0-flash",
    description=(
        "Profiles cleaned data and writes a structured six-section feature "
        "engineering analysis report covering encoding, skewness, variance, "
        "multicollinearity, interaction opportunities, and feature summary."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by FE orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_fe_analyzer(
    records: list[dict[str, Any]],
    target_col: str,
) -> str:
    """
    Run the Feature Engineering Analyzer and return the plain-text analysis report.

    Args:
        records:    Cleaned dataset rows as a list of dicts.
        target_col: ML target column name.

    Returns:
        Full feature-engineering analysis report text written by the agent.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    user_message = json.dumps(
        {
            "records": records,
            "target_col": target_col,
        },
        default=str,
    )

    runner = InMemoryRunner(agent=fe_analyzer_agent)
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
