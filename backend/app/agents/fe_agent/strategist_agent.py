"""
app/agents/fe_agent/strategist_agent.py

Feature Engineering Strategist Agent — Agent 2 (Child) in the FE Pipeline.

Input:  Full analysis report from FE Analyzer + target column + numeric/categorical
        column lists.
Output: A numbered, implementation-agnostic feature engineering plan with:
        drop, encoding, transform, and interaction decisions.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — expert FE strategist persona
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a senior ML feature engineer.
You receive a feature analysis report and create an exact feature engineering plan.

You will receive:
Full analysis report from Feature Engineering Analyzer
target_col: ML target column name
numeric_cols: list of numeric columns
categorical_cols: list of categorical columns

Your output MUST be a numbered plan covering ALL sections in this order:

1. COLUMNS TO DROP — list each column and exact reason:
   - Near-zero variance columns: drop entirely
   - ID/index columns: drop entirely
   - High multicollinearity: drop the one less correlated with target
   - State exact column names

2. CATEGORICAL ENCODING PLAN — for each categorical column:
   - Binary columns (2 unique): LabelEncoder
     Example: "gender → LabelEncoder → 0/1"
   - Low cardinality (3-10 unique): pd.get_dummies, drop_first=True
     Example: "department → get_dummies → drop first category"
   - High cardinality (>10 unique): LabelEncoder
     Example: "city → LabelEncoder (52 unique values)"
   - Ordinal columns: OrdinalEncoder with exact order
     Example: "size → OrdinalEncoder → [small, medium, large]"

3. SKEWNESS TREATMENT PLAN — for each flagged column:
   - Skewness 1-2: np.log1p transform (handles zeros safely)
   - Skewness > 2: np.sqrt transform
   - Negative skewness < -1: reflect then log1p
   - State exact column and exact transform
   - Example: "salary → np.log1p (skewness: 2.3)"

4. INTERACTION FEATURES — for each approved interaction:
   - State exact formula
   - State exact new column name
   - Example: "salary / experience_years → salary_per_experience"
   - Maximum 3 new features

5. FINAL FEATURE LIST:
   - List ALL columns that will exist after engineering
   - Mark which are original, which are new, which are encoded
   - This is the exact input that goes to Model Training

Strict rules:
- Give exact column names for every decision.
- Give exact reasoning for every drop decision.
- Do NOT write any code.
- Output plain text, not JSON.
- Order: Drop first → Encode → Transform → Create interactions.
- Never drop target_col under any circumstance.
- If unsure about dropping — keep the column.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent definition — no tools needed (planner reasoner only)
# ─────────────────────────────────────────────────────────────────────────────

fe_strategist_agent = Agent(
    name="fe_strategist_agent",
    model="gemini-2.0-flash",
    description=(
        "Builds an exact feature-engineering action plan from the FE analyzer "
        "report. Produces an ordered plan with drop, categorical encoding, "
        "skewness treatment, interaction creation, and final feature list. "
        "No code — only concrete, column-level decisions."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=[],  # reasoning-only; no tools required
)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by FE orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_fe_strategist(
    analyzer_report: str | dict[str, Any],
    target_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, Any]:
    """
    Run the Feature Engineering Strategist and return a numbered engineering plan.

    Args:
        analyzer_report:  Full report text from the FE Analyzer (string) or
                          dict with an "analysis_report" key.
        target_col:       ML target column name.
        numeric_cols:     List of numeric columns.
        categorical_cols: List of categorical columns.

    Returns:
        dict with keys:
          dataset_id       — optional dataset id if provided by caller
          target_col       — the passed-in target column
          numeric_cols     — the passed-in numeric columns
          categorical_cols — the passed-in categorical columns
          feature_plan     — plain-text feature-engineering plan
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    report_text = ""
    dataset_id = "unknown"
    if isinstance(analyzer_report, dict):
        report_text = str(analyzer_report.get("analysis_report", ""))
        dataset_id = str(analyzer_report.get("dataset_id", dataset_id))
    else:
        report_text = str(analyzer_report or "")

    user_message = json.dumps(
        {
            "analysis_report": report_text,
            "target_col": target_col,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        },
        default=str,
    )

    runner = InMemoryRunner(agent=fe_strategist_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="system",
    )

    plan_text = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            plan_text = event.content.parts[0].text or ""

    return {
        "dataset_id": dataset_id,
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_plan": plan_text.strip(),
    }
