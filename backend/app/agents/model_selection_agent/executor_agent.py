"""
app/agents/model_selection_agent/executor_agent.py

Model Selection Executor Agent — Agent 3 in the MS pipeline.

Input:  Strategist plan + training results from MT phase (already in sandbox).
Tool:   execute_python(code: str) -> str
Output: Confirms best model, sets final_model + final_model_name in sandbox,
        saves outputs/model_selection_summary.json.
"""

from __future__ import annotations

import json
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
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert Python ML engineer specializing in model selection.
You receive a model selection plan and implement it by writing
and executing Python code using the execute_python tool.

You have one tool: execute_python(code: str) -> str
This tool runs Python code in a shared sandbox that already contains:
  best_model: best trained model object from Model Training phase
  best_model_name: string name of the best model
  all_results: dict of all 3 model metrics from training
  smote_applied: boolean
  target_col: ML target column name
  X_test: test features
  y_test: test labels
  pd, np: pandas and numpy

YOUR WORKFLOW — follow these steps in exact order:

STEP 1: Verify sandbox
Write and run code to:
  Print best_model_name
  Print list(all_results.keys()) to see available models
  Print type(best_model)
  Confirm X_test and y_test shapes

STEP 2: Confirm model selection
Write and run code to:
  Extract metrics from all_results for each model
  Determine primary metric: f1_weighted if imbalanced, accuracy if balanced
  Rank all models by primary metric (descending)
  Print a comparison table: model name | accuracy | f1
  Confirm best_model_name matches the highest-ranked model
  If mismatch: load the correct model from outputs/ directory using joblib

STEP 3: Set final_model globals
Write and run code to:
  Set globals()["final_model"] = best_model
  Set globals()["final_model_name"] = best_model_name
  Print: "final_model set to:", globals()["final_model_name"]
  Print: "Type:", type(globals()["final_model"])

STEP 4: Save selection summary
Write and run code to:
  Build a selection_summary dict with keys:
    final_model_name: the chosen model name
    all_models_ranked: list of dicts [{name, accuracy, f1}] sorted by primary metric
    selection_metric: "f1_weighted" or "accuracy"
    smote_applied: bool from sandbox
  Import os; os.makedirs("outputs", exist_ok=True)
  Save to outputs/model_selection_summary.json
  Print: "Saved outputs/model_selection_summary.json"

STEP 5: Verify
Write and run code to:
  Check "final_model" in globals()
  Check "final_model_name" in globals()
  Import os; check os.path.isfile("outputs/model_selection_summary.json")
  Print PASS or FAIL for each check

ERROR HANDLING RULES:
If all_results is empty — set final_model = best_model, final_model_name = best_model_name
If a pkl file cannot be loaded — keep the existing best_model
Maximum 2 retries per step
Always set final_model and final_model_name — HT and ME depend on them
Always save model_selection_summary.json

STRICT RULES:
Always verify sandbox in Step 1
Set final_model and final_model_name in globals() — not just local variables
Print results after every step
"""


ms_executor_agent = Agent(
    name="ms_executor_agent",
    model="gemini-2.0-flash",
    description=(
        "Confirms the best model from the Model Training phase, sets final_model "
        "and final_model_name in sandbox globals, and saves "
        "outputs/model_selection_summary.json."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)


async def run_ms_executor(strategist_output: dict[str, Any]) -> dict[str, Any]:
    """
    Run the MS Executor agent end-to-end.

    Args:
        strategist_output: Output from run_model_selection_strategist() containing
                           training_plan, target_col, task_type,
                           minority_class_percentage.

    Returns:
        Dict with execution_log and verify records.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    user_message = json.dumps(
        {
            "training_plan": strategist_output.get("training_plan", ""),
            "target_col": strategist_output.get("target_col", ""),
            "task_type": strategist_output.get("task_type", "CLASSIFICATION"),
            "minority_class_percentage": strategist_output.get(
                "minority_class_percentage", 50.0
            ),
            "feature_count": strategist_output.get("feature_count", 0),
            "instructions": (
                "Follow the 5-step workflow exactly. "
                "Confirm best_model_name matches highest-ranked model. "
                "Always set final_model and final_model_name in globals(). "
                "Always save outputs/model_selection_summary.json."
            ),
        },
        default=str,
    )

    runner = InMemoryRunner(agent=ms_executor_agent)
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
        "execution_log": execution_log.strip(),
        "verify_selection_summary": verify_output_saved(
            "outputs/model_selection_summary.json"
        ),
    }
