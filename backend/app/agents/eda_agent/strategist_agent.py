"""
app/agents/eda_agent/strategist_agent.py

EDA Strategist Agent — Agent 2 (Child) in the EDA Pipeline.

Input:  Full EDA analysis report text + target_col + numeric_cols + categorical_cols
Tools:  None — pure LLM reasoning
Output: A numbered chart plan with exact filenames, chart types, and purposes.
        No code — only a concrete, actionable visualisation plan.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a senior data visualization expert and data scientist.
You receive an EDA analysis report and plan exactly what charts to generate
and what statistical computations to perform.

You will receive:
- Full EDA analysis report from the Analyzer agent
- Target column name
- List of numeric columns
- List of categorical columns

Your output MUST be a numbered plan covering ALL of the following:

## 1. DISTRIBUTION CHARTS — for each numeric column:
- Histogram with KDE curve
- State exact filename: charts/distribution_{column_name}.png
- Note if log scale needed (highly skewed columns — skewness > 1 or < -1
  as flagged in the analysis report)

## 2. CORRELATION HEATMAP — one chart:
- Full correlation matrix of all numeric columns
- Annotate with exact correlation values
- Filename: charts/correlation_heatmap.png
- Skip if fewer than 2 numeric columns

## 3. TARGET VARIABLE CHART:
- If classification (categorical target): bar chart of class counts + percentages
- If regression (numeric target): histogram of target distribution
- Filename: charts/target_distribution.png

## 4. BOXPLOTS — for each numeric column:
- Show outlier visualization (IQR fences, individual outlier points)
- Color: red for outlier points
- Filename: charts/boxplot_{column_name}.png

## 5. CATEGORICAL CHARTS — for each categorical column:
- Bar chart of top 10 value frequencies
- Filename: charts/categorical_{column_name}.png
- Include the target column if it is categorical

## 6. BEFORE vs AFTER COMPARISON CHARTS:
- For EACH numeric column: side-by-side histogram (raw data vs cleaned data)
- Filename: charts/comparison_{column_name}.png
- This section is MANDATORY — the UI depends on these charts
- One chart per numeric column — no exceptions

## 7. PAIRPLOT:
- Include ONLY if the total number of numeric columns is <= 6
- Scatter matrix of all numeric columns, colored by target column
- Filename: charts/pairplot.png
- If numeric columns > 6: explicitly write "SKIP pairplot — too many columns"

## 8. STATISTICAL SUMMARY TABLE:
- A JSON file (not a chart) containing key stats for UI display
- For each column include: mean, std, min, max, skewness, null_count, outlier_count
  (use exact values from the analysis report)
- Filename: charts/eda_stats.json

## OUTPUT FORMAT — for each chart write:

### Chart N: <descriptive title>
- Type: <exact chart type>
- Columns: <exact column name(s) — copied verbatim from the column lists provided>
- Filename: <exact path>
- Color: <color scheme — blues for distributions, reds for outliers, greens for target>
- Purpose: <one sentence>
- Log scale: yes / no  (only relevant for distribution charts)

## STRICT RULES
- Use EXACT column names — copy verbatim from numeric_cols / categorical_cols provided
- Use EXACT filenames — lowercase, underscores, no spaces, always in charts/
- Do NOT write any Python code
- Do NOT skip section 6 (Before vs After) — it is mandatory
- Do NOT skip section 8 (eda_stats.json) — it is mandatory
- Sections 1–8 must appear in order
- State "N/A" for Log scale on non-distribution charts
"""

# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

eda_strategist_agent = Agent(
    name="eda_strategist_agent",
    model="gemini-2.0-flash",
    description=(
        "Reads an EDA analysis report and produces a numbered, chart-by-chart "
        "EDA plan covering distributions, heatmap, target, boxplots, categorical, "
        "before/after comparisons, pairplot, and eda_stats.json. "
        "No code — only a precise, actionable visualisation plan."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=[],  # pure LLM reasoning — no tool calls needed
)

# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by the EDA orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_eda_strategist(
    analyzer_output: dict[str, Any],
    target_col: str,
) -> dict[str, Any]:
    """
    Run the EDA Strategist Agent and return the chart plan.

    The agent receives the analysis report text, target_col, and the exact
    column lists — not the raw stats dict. This keeps the message focused on
    what the strategist needs for visualisation planning.

    Args:
        analyzer_output: Full dict returned by run_eda_analyzer() — must contain
                         keys: dataset_id, target_col, stats, analysis_report.
        target_col:      Name of the ML target column.

    Returns:
        dict with keys:
          "dataset_id"    — from analyzer_output
          "target_col"    — the passed-in target column
          "chart_plan"    — numbered plain-text chart plan written by the agent
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    dataset_id: str = analyzer_output.get("dataset_id", "unknown")
    stats: dict = analyzer_output.get("stats", {})

    # Pass only what the strategist needs — not the full stats dict
    user_message = json.dumps(
        {
            "dataset_id": dataset_id,
            "target_col": target_col,
            "numeric_cols": stats.get("numeric_cols", []),
            "categorical_cols": stats.get("categorical_cols", []),
            "analysis_report": analyzer_output.get("analysis_report", ""),
        },
        default=str,
    )

    runner = InMemoryRunner(agent=eda_strategist_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="system",
    )

    chart_plan = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            chart_plan = event.content.parts[0].text or ""

    return {
        "dataset_id": dataset_id,
        "target_col": target_col,
        "chart_plan": chart_plan.strip(),
    }
