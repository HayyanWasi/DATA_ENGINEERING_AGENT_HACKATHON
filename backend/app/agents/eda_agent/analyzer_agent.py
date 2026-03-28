"""
app/agents/eda_agent/analyzer_agent.py

EDA Analyzer Agent — Agent 1 (Child) in the EDA Pipeline.

Input:  Pre-computed EDA stats dict (from analyze_eda()) passed in the message.
        Stats are computed by run_eda_analyzer() before calling the agent —
        the agent receives the full dict and reports findings. No tools needed.
Output: Structured 6-section EDA analysis report.
        Facts only — no solutions, no code, no visualization suggestions.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent

from app.tools.eda_analysis_tools import analyze_eda, compute_eda_stats  # noqa: F401

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — exact analyst persona from spec
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert data analyst specializing in Exploratory Data Analysis.
Your ONLY job is to analyze CLEANED data and report statistical findings.

You will receive a Python dictionary containing:
- shape: dimensions of the cleaned dataset
- dtypes: data types of each column
- numeric_stats: min, max, mean, median, std, IQR, skewness, kurtosis per numeric column
- null_counts: remaining nulls per column (should be near 0 after cleaning)
- duplicate_rows: remaining duplicates
- categorical_stats: unique counts and top frequencies per categorical column
- target_col: the ML target column name
- numeric_cols: list of numeric columns
- categorical_cols: list of categorical columns
- correlation_matrix: correlation values between all numeric columns
- target_distribution: value counts of target column
- class_balance: percentage of each class in target column
- sample_rows: first 3 rows of cleaned data

Your output MUST cover ALL of the following sections:

## SECTION 1 — DISTRIBUTION ANALYSIS
For each numeric column: is it normal, left-skewed, or right-skewed?
State exact skewness value.
Flag any columns with skewness > 1 or < -1 as highly skewed.
State min, max, mean, median for each.

## SECTION 2 — CORRELATION ANALYSIS
List ALL column pairs with correlation > 0.7 (high positive).
List ALL column pairs with correlation < -0.7 (high negative).
Identify which columns are most correlated with the target column.
Flag multicollinearity risks (two features correlated > 0.9 with each other).

## SECTION 3 — TARGET VARIABLE ANALYSIS
State exact class distribution with percentages.
Flag if imbalanced: minority class < 20% = severely imbalanced.
State whether this is classification or regression based on target dtype.

## SECTION 4 — OUTLIER SUMMARY
For each numeric column: how many outliers remain after cleaning.
Use IQR method: outlier if value < Q1 - 1.5*IQR or > Q3 + 1.5*IQR.
State outlier count and percentage for each column.

## SECTION 5 — CATEGORICAL ANALYSIS
For each categorical column: unique value count.
Flag high cardinality columns (unique values > 20).
List top 3 most frequent values per categorical column.

## SECTION 6 — KEY INSIGHTS FOR MODELING
Which columns are most predictive of target (based on correlation).
Which columns to potentially drop (near-zero variance, high multicollinearity).
Any data quality concerns remaining after cleaning.

## STRICT RULES
- Report ONLY facts with exact numbers from the input dictionary
- Do NOT suggest visualization types
- Do NOT write any code
- Do NOT make modeling decisions — only report observations
- Be specific: use exact column names, exact numbers, exact percentages
- Output plain text only — no JSON
"""

# ─────────────────────────────────────────────────────────────────────────────
# Agent definition — no tools needed (stats passed in message)
# ─────────────────────────────────────────────────────────────────────────────

eda_analyzer_agent = Agent(
    name="eda_analyzer_agent",
    model="gemini-2.0-flash",
    description=(
        "Receives a pre-computed EDA stats dict and writes a structured 6-section "
        "fact-only analysis report covering distributions, correlations, target "
        "variable, outliers, categoricals, and modeling insights. No tools — "
        "reports directly from the provided stats."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=[],  # stats are pre-computed and passed in the user message
)

# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by the EDA orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_eda_analyzer(
    dataset_id: str,
    df_stats: dict[str, Any],
    target_col: str,
) -> dict[str, Any]:
    """
    Run the EDA Analyzer Agent using pre-computed dual-dataset stats.

    The caller (EDA Orchestrator) pre-computes df_stats via compute_eda_stats()
    and passes it here. The agent receives the stats dict — not raw records —
    keeping the message compact even for large datasets.

    Args:
        dataset_id: Unique identifier for the dataset.
        df_stats:   Combined stats dict from compute_eda_stats() with keys:
                      "clean"  — full analyze_eda() result for cleaned data
                      "raw"    — lightweight raw stats (shape, nulls, basics)
                      "target_col", "numeric_cols", "categorical_cols"
        target_col: Name of the ML target column.

    Returns:
        dict with keys:
          "dataset_id"      — the passed-in id
          "target_col"      — the passed-in target column
          "stats"           — df_stats["clean"] (clean stats for downstream use)
          "analysis_report" — full plain-text EDA report written by the agent
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    # The agent receives the full dual-dataset stats — including raw stats
    # so it can state "before cleaning: 23% nulls → after cleaning: 0%"
    user_message = json.dumps(
        {
            "dataset_id": dataset_id,
            "target_col": target_col,
            "stats": df_stats,
        },
        default=str,
    )

    runner = InMemoryRunner(agent=eda_analyzer_agent)
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

    # Expose clean stats at top level — consumed by strategist + orchestrator
    return {
        "dataset_id": dataset_id,
        "target_col": target_col,
        "stats": df_stats.get("clean", df_stats),   # clean stats for downstream
        "df_stats": df_stats,                        # full dual-dataset stats
        "analysis_report": report_text.strip(),
    }
