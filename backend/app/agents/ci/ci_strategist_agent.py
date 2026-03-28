"""
app/agents/ci/ci_strategist_agent.py

Class Imbalance Strategist Agent — Agent 2 in the Class Imbalance handling
pipeline.

Input:  Full class imbalance analysis report + target/imbalance metrics.
Output: A numbered implementation-agnostic balancing plan with exact technique
selection, expected outcome, sandbox variables, and saving plan.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — exact balancing strategist persona
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a senior ML engineer specializing in class imbalance handling.
You receive an imbalance analysis report and create an exact handling strategy.

You will receive:
Full analysis report from Class Imbalance Analyzer
target_col: ML target column name
total_train_samples: total training rows
minority_class_percentage: exact float
minority_class_count: exact integer
task_type: CLASSIFICATION or REGRESSION
severity: BALANCED / MILD / MODERATE / SEVERE

Your output MUST be a numbered plan covering ALL sections:

1. TECHNIQUE DECISION — choose exactly one:

   Option A — SKIP (no balancing):
   - Use when: severity is BALANCED (minority >= 40%)
   - Use when: regression task
   - Use when: dataset too small (< 100 rows)
   - Use when: minority class < 6 samples
   - State exact reason for skipping
   - X_train_bal = X_train, y_train_bal = y_train

   Option B — SMOTE (oversample minority):
   - Use when: severity is MODERATE or SEVERE
   - Use when: minority >= 6 samples
   - Use when: total samples >= 100
   - Parameters: random_state=42, k_neighbors=5
   - Expected output: equal class distribution

   Option C — Random Undersampling (shrink majority):
   - Use when: total samples > 1000 and severity is MILD
   - Use when: SMOTE would create too many synthetic samples
   - Parameters: random_state=42
   - Expected output: equal class distribution

   Option D — SMOTE + Tomek Links (combined):
   - Use when: severity is SEVERE and samples > 500
   - Cleans boundary samples after oversampling
   - Parameters: random_state=42

2. EXPECTED OUTCOME:
   - State exact expected class distribution after technique
   - State expected new training size
   - State which classes will be affected

3. WHAT NOT TO TOUCH:
   - X_test: NEVER modify — state this explicitly
   - y_test: NEVER modify — state this explicitly
   - df, df_clean, df_engineered: read only
   - Only X_train and y_train are modified

4. SANDBOX UPDATE PLAN:
   - X_train_bal: balanced training features
   - y_train_bal: balanced training labels
   - smote_applied: True or False boolean
   - balance_technique: exact technique name string
   - balance_report: dict with before/after distribution

5. SAVING PLAN:
   - Save balance report to outputs/balance_report.json
   - This is used by Model Training and final report

Strict rules:
- Do NOT write any code.
- Choose exactly ONE technique — never combine unless Option D.
- Never touch X_test or y_test.
- If regression — always choose Option A with clear reason.
- Give exact parameters for chosen technique.
- State exact expected class counts after balancing.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

ci_strategist_agent = Agent(
    name="ci_strategist_agent",
    model="gemini-2.0-flash",
    description=(
        "Converts class-imbalance analysis into an exact one-technique balancing "
        "strategy with expected class counts, explicit non-touching constraints, "
        "sandbox variable plan, and output persistence details."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=[],  # planning only, no tools required
)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by a class-imbalance orchestrator (or API route)
# ─────────────────────────────────────────────────────────────────────────────

async def run_ci_strategist(
    analyzer_report: str | dict[str, Any],
    target_col: str,
    total_train_samples: int,
    minority_class_percentage: float,
    minority_class_count: int,
    task_type: str,
    severity: str,
) -> dict[str, Any]:
    """
    Run the Class Imbalance Strategist and return a numbered balancing plan.

    Args:
        analyzer_report:         Full analysis report text from CI Analyzer or
                                 a dict containing "analysis_report".
        target_col:              ML target column name.
        total_train_samples:     Count of training rows.
        minority_class_percentage: Minority percentage from training labels.
        minority_class_count:    Minority class sample count.
        task_type:               CLASSIFICATION or REGRESSION.
        severity:                BALANCED / MILD / MODERATE / SEVERE.

    Returns:
        dict with keys:
          target_col:                  target column passed in
          total_train_samples:         total training rows
          minority_class_percentage:    minority percentage passed in
          minority_class_count:         minority count passed in
          task_type:                   task type passed in
          severity:                    severity passed in
          balance_plan:                full plain-text balancing plan
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    report_text = str(analyzer_report)
    if isinstance(analyzer_report, dict):
        report_text = str(analyzer_report.get("analysis_report", ""))

    user_message = json.dumps(
        {
            "analysis_report": report_text,
            "target_col": target_col,
            "total_train_samples": total_train_samples,
            "minority_class_percentage": minority_class_percentage,
            "minority_class_count": minority_class_count,
            "task_type": task_type,
            "severity": severity,
        },
        default=str,
    )

    runner = InMemoryRunner(agent=ci_strategist_agent)
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
        "target_col": target_col,
        "total_train_samples": total_train_samples,
        "minority_class_percentage": minority_class_percentage,
        "minority_class_count": minority_class_count,
        "task_type": task_type,
        "severity": severity,
        "balance_plan": plan_text.strip(),
    }
