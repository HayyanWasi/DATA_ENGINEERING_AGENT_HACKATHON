"""
app/agents/ci/ci_executor_agent.py

Class Imbalance Executor Agent — Agent 3 (Child) in the Class Imbalance
pipeline.

Input:  Balancing strategy (from CI Strategist) + balanced training context.
Tool:   execute_python(code: str) -> str
Output: Balanced split saved to globals for Model Training:
        X_train_bal, y_train_bal, smote_applied, balance_technique,
        and outputs/balance_report.json.
"""

from __future__ import annotations

import json
import re
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.tools.executor_tools import execute_python, verify_output_saved


# ─────────────────────────────────────────────────────────────────────────────
# Tool registration
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS: list[FunctionTool] = [
    FunctionTool(execute_python),
    FunctionTool(verify_output_saved),
]


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_selected_option(plan_text: str) -> str:
    """
    Extract chosen option letter from strategist section heading.
    """
    txt = str(plan_text or "").upper()
    for option in ("D", "B", "C", "A"):
        if re.search(rf"OPTION\s*{option}\b", txt):
            if option == "A":
                return "skip"
            if option == "B":
                return "smote"
            if option == "C":
                return "undersampling"
            return "smote_tomek"
    return "skip"


def _extract_param(plan_text: str, param: str, default: int | float | str) -> int | float | str:
    """
    Extract a single plan parameter from plain text using permissive regex.
    """
    txt = str(plan_text or "")
    patterns = {
        "random_state": r"random_state\s*=\s*([0-9]+)",
        "k_neighbors": r"k_neighbors\s*=\s*([0-9]+)",
    }
    raw = patterns.get(param)
    if not raw:
        return default
    match = re.search(raw, txt, flags=re.IGNORECASE)
    if not match:
        return default
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — exact implementation workflow
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert Python ML engineer specializing in class imbalance handling.
You receive a balancing strategy and implement it by writing and executing Python
code using the execute_python tool.

You have access to one tool: execute_python(code: str) -> str
This tool runs Python code in a shared sandbox that already contains:

X_train: scaled training features
X_test: scaled test features (DO NOT TOUCH)
y_train: training labels
y_test: test labels (DO NOT TOUCH)
target_col: ML target column name
pd, np: pandas and numpy

Your workflow — follow these steps in exact order:

STEP 1: Verify sandbox state
Write and run code to confirm X_train and y_train
are loaded correctly.

Print X_train shape, y_train shape.
Print exact class distribution and percentages of y_train.
Print target_col value.
Confirm X_test and y_test exist but DO NOT modify them.

STEP 2: Apply chosen technique from strategy plan
Write and run code implementing exactly the technique
the Strategist decided:

If SKIP:
Write code to set X_train_bal = X_train copy,
y_train_bal = y_train copy, smote_applied = False.
Print reason for skipping.

If SMOTE:
Write code to import SMOTE from imblearn.over_sampling.
Apply SMOTE with exact parameters from plan.
Store result as X_train_bal, y_train_bal.
Set smote_applied = True.

If Random Undersampling:
Write code to import RandomUnderSampler from
imblearn.under_sampling.
Apply with exact parameters from plan.
Store result as X_train_bal, y_train_bal.
Set smote_applied = False, balance_technique = undersampling.

If SMOTE + Tomek:
Write code to import SMOTETomek from imblearn.combine.
Apply with exact parameters from plan.
Store result as X_train_bal, y_train_bal.
Set smote_applied = True.

STEP 3: Verify balancing result
Write and run code to print:
X_train_bal shape
y_train_bal class distribution before and after
Confirm X_test and y_test are unchanged
Print exact class counts and percentages after balancing

STEP 4: Update sandbox globals
Write and run code to update sandbox with:
X_train_bal: balanced training features
y_train_bal: balanced training labels
smote_applied: boolean
balance_technique: technique name string
balance_report: dict with before/after distribution

STEP 5: Save balance report
Write and run code to save balance_report dict
to outputs/balance_report.json.
Print confirmation of save.

STEP 6: Final verification
Write and run code to confirm:
X_train_bal exists in sandbox
y_train_bal exists in sandbox
X_test unchanged
y_test unchanged
outputs/balance_report.json saved
Print PASS or FAIL for each check

ERROR HANDLING RULES:
If execute_python returns ERROR — read it, fix, retry
Maximum 3 retries per step
If SMOTE fails due to sample size — fall back to Skip
If undersampling fails — fall back to Skip
Log all fallbacks clearly
Never modify X_test or y_test under any circumstance
Always work on copies: X_train.copy(), y_train.copy()
Never update df, df_clean, df_engineered — read only
Always update globals() so Model Training gets balanced data
Always print distribution before AND after balancing
Always save balance_report.json
Print results after every step

Important constraints from strategy:
- Do not change technique from the chosen option.
- If strategy has regression task, always use SKIP.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

ci_implementor_agent = Agent(
    name="ci_implementor_agent",
    model="gemini-2.0-flash",
    description=(
        "Implements an exact balancing strategy from the CI strategist. "
        "Runs sandbox verification, applies the selected technique "
        "(SMOTE, RandomUnderSampler, SMOTETomek, or Skip), updates globals "
        "for Model Training, saves outputs/balance_report.json, and verifies "
        "all artifacts."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)


# Backward-compatible alias
ci_executor_agent = ci_implementor_agent


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by CI orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_ci_implementor(strategist_output: dict[str, Any]) -> dict[str, Any]:
    """
    Run the Class Imbalance Executor and return a compact artifact payload.

    Args:
        strategist_output: Dict from run_ci_strategist() containing:
                            target_col, balance_plan and optional metadata.

    Returns:
        dict with output path and execution summary used by the next phase.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    target_col: str = strategist_output.get("target_col", "")
    balance_plan: str = strategist_output.get("balance_plan", "")
    if not balance_plan:
        balance_plan = "Technician summary missing; default to Option A — SKIP."

    selected_option = _extract_selected_option(balance_plan)
    random_state = _extract_param(balance_plan, "random_state", 42)
    k_neighbors = _extract_param(balance_plan, "k_neighbors", 5)

    # Conservative defaults when parse fails
    if selected_option == "smote":
        selected_technique = "smote"
    elif selected_option == "undersampling":
        selected_technique = "undersampling"
    elif selected_option == "smote_tomek":
        selected_technique = "smote_tomek"
    else:
        selected_technique = "skip"

    # Pass explicit strategy metadata for deterministic LLM behaviour
    user_message = json.dumps(
        {
            "balance_plan": balance_plan,
            "target_col": target_col,
            "selected_technique": selected_technique,
            "total_train_samples": strategist_output.get("total_train_samples"),
            "minority_class_percentage": strategist_output.get(
                "minority_class_percentage",
            ),
            "minority_class_count": strategist_output.get("minority_class_count"),
            "task_type": strategist_output.get("task_type"),
            "severity": strategist_output.get("severity"),
            "random_state": random_state,
            "k_neighbors": k_neighbors,
            "instructions": (
                f"Technique must be treated as: {selected_technique}. "
                "Do not change technique unless execution fails and fallback rules "
                "explicitly apply."
            ),
        },
        default=str,
    )

    runner = InMemoryRunner(agent=ci_implementor_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="system",
    )

    execution_log = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            execution_log = event.content.parts[0].text or ""

    return {
        "status": "completed",
        "target_col": target_col,
        "selected_technique": selected_technique,
        "balance_output": "outputs/balance_report.json",
        "execution_log": execution_log.strip(),
        "verify_report": verify_output_saved("outputs/balance_report.json"),
    }


# Backward-compatible alias
run_ci_executor = run_ci_implementor
