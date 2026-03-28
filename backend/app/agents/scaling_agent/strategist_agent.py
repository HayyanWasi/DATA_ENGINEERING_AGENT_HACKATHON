"""
app/agents/scaling_agent/strategist_agent.py

Scaling Strategist Agent — Agent 2 (Child) in the Feature-Scaling Pipeline.

Input:  Full scaling analysis report + target/column metadata.
Output: A numbered scaling plan with explicit skip/scale assignments and
        scaler-saving notes. No code.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — expert scaling strategist persona
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a senior ML engineer specializing in feature scaling.
You receive a scaling analysis report and create an exact scaling plan.

You will receive:
Full analysis report from Feature Scaling Analyzer
target_col: ML target column name
numeric_cols: list of numeric columns
encoded_cols: columns encoded in FE phase
transformed_cols: columns transformed in FE phase

Your output MUST be a numbered plan covering ALL sections in this order:

1. COLUMNS TO SKIP SCALING — list each and exact reason:
   - Binary columns (0/1 values): skip — already scaled
   - One-hot encoded columns: skip — already 0/1
   - Target column: ALWAYS skip — never scale target
   - State exact column names and reasons

2. STANDARD SCALER COLUMNS — apply when:
   - Distribution is approximately normal
   - No significant outliers remaining
   - Algorithm is distance-based (SVM, KNN, Linear models)
   - Formula: (x - mean) / std
   - List exact column names

3. ROBUST SCALER COLUMNS — apply when:
   - Outliers still present after cleaning
   - Distribution is skewed
   - Formula: (x - median) / IQR
   - List exact column names

4. MINMAX SCALER COLUMNS — apply when:
   - Column needs strict 0-1 range
   - No significant outliers
   - Formula: (x - min) / (max - min)
   - List exact column names

5. SCALING ORDER:
   - State exact order of operations
   - Always: fit on training data only
   - Always: transform both train and test with same fitted scaler
   - State which scaler object to save for deployment

6. SCALER SAVING PLAN:
   - State exact filename for each scaler: outputs/scaler_{name}.pkl
   - These are saved alongside model for production use

STRICT RULES:
- target_col must NEVER appear in any scaling list
- Give exact column names for every decision
- Do NOT write any code
- One-hot and binary encoded columns always skip scaling
- Always fit on train set only — never on full dataset
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

scaling_strategist_agent = Agent(
    name="scaling_strategist_agent",
    model="gemini-2.0-flash",
    description=(
        "Converts scaling analysis findings into an exact column-level feature "
        "scaling plan, including skip decisions, StandardScaler/RobustScaler/"
        "MinMaxScaler selection, scaling order, and persistence plan. No code "
        "is produced."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=[],
)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by a scaling orchestrator
# ─────────────────────────────────────────────────────────────────────────────


async def run_scaling_strategist(
    analyzer_report: str | dict[str, Any],
    target_col: str,
    numeric_cols: list[str],
    encoded_cols: list[str],
    transformed_cols: list[str],
) -> dict[str, Any]:
    """
    Run the Scaling Strategist and return a numbered scaling plan.

    Args:
        analyzer_report: Full report text from scaling analyzer or dict containing
                         an "analysis_report" key.
        target_col:      ML target column name.
        numeric_cols:    Numeric feature columns (target may be excluded already).
        encoded_cols:    Categorical columns already encoded in FE phase.
        transformed_cols: Columns transformed during FE phase.

    Returns:
        dict with keys:
          target_col       — the passed-in target column
          numeric_cols     — the passed-in numeric columns
          encoded_cols     — the passed-in encoded columns
          transformed_cols — the passed-in transformed columns
          scaling_plan     — plain-text scaling plan text written by the agent
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    plan_input = str(analyzer_report)
    if isinstance(analyzer_report, dict):
        plan_input = str(analyzer_report.get("analysis_report", ""))

    user_message = json.dumps(
        {
            "analysis_report": plan_input,
            "target_col": target_col,
            "numeric_cols": numeric_cols,
            "encoded_cols": encoded_cols,
            "transformed_cols": transformed_cols,
        },
        default=str,
    )

    runner = InMemoryRunner(agent=scaling_strategist_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="system",
    )

    scaling_plan = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            scaling_plan = event.content.parts[0].text or ""

    return {
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "encoded_cols": encoded_cols,
        "transformed_cols": transformed_cols,
        "scaling_plan": scaling_plan.strip(),
    }
