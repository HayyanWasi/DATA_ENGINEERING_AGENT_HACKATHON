"""
app/agents/hyperparameter_tuning_agent/executor_agent.py

Hyperparameter Tuning Executor Agent — Agent 3 in the HT pipeline.

Input:  Strategist output (param_grid, cv_folds, scoring, best_model_name).
Tool:   execute_python(code: str) -> str
Output: Tuned model artifact + tuning_results.json saved in outputs/.
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
You are an expert Python ML engineer specializing in hyperparameter optimization.
You receive a param_grid configuration and run GridSearchCV to find the best
hyperparameters for the winning model.

You have access to one tool: execute_python(code: str) -> str
This tool runs Python code in a shared sandbox that already contains:

best_model: the trained winning model object
best_model_name: string name of the winning model
X_train_bal: SMOTE-balanced training features (or X_train if SMOTE not applied)
y_train_bal: SMOTE-balanced training labels (or y_train if SMOTE not applied)
X_test: test features
y_test: test labels
pd, np: pandas and numpy
RandomForestClassifier, LogisticRegression, XGBClassifier
classification_report, f1_score, accuracy_score
GridSearchCV
joblib: for saving models

You must follow this EXACT workflow:

STEP 1: Verify sandbox state
Write and run code to confirm these variables exist in globals():
best_model, best_model_name, X_train_bal, y_train_bal, X_test, y_test
Print shapes and best_model_name.
If X_train_bal is missing, fall back to X_train and y_train.
If anything critical is missing, stop and report.

STEP 2: Reconstruct the base estimator
Based on best_model_name, import and instantiate a FRESH base estimator
with the SAME fixed hyperparameters used during training (non-tuned ones).
This ensures GridSearchCV tunes on a clean estimator, not a pre-fitted one.
For Random Forest:
  from sklearn.ensemble import RandomForestClassifier
  base = RandomForestClassifier(random_state=42, class_weight="balanced")
For Logistic Regression:
  from sklearn.linear_model import LogisticRegression
  base = LogisticRegression(random_state=42, class_weight="balanced")
For XGBoost:
  from xgboost import XGBClassifier
  base = XGBClassifier(random_state=42, eval_metric="logloss")

STEP 3: Run GridSearchCV
Use param_grid, cv_folds, and scoring from the input.
Code pattern:
  from sklearn.model_selection import GridSearchCV
  grid_search = GridSearchCV(
      estimator=base,
      param_grid=param_grid,
      cv=cv_folds,
      scoring=scoring,
      refit=True,
      n_jobs=-1,
      verbose=0,
  )
  grid_search.fit(X_train_bal, y_train_bal)
Print:
  Best params: grid_search.best_params_
  Best CV score: grid_search.best_score_

STEP 4: Evaluate best estimator on test set
Use grid_search.best_estimator_ to predict on X_test.
Compute and print: accuracy, f1_weighted, precision, recall.
Compare tuned_score vs baseline_score (passed in input) and print delta.

STEP 5: Save tuned model
Save grid_search.best_estimator_ to outputs/tuned_model.pkl using joblib.
Update sandbox globals:
  tuned_model = grid_search.best_estimator_
  best_params  = grid_search.best_params_
  tuned_score  = <primary metric score on X_test>

STEP 6: Save tuning_results.json
Write outputs/tuning_results.json with:
{
  "status": "success",
  "best_model_name": <best_model_name>,
  "best_params": <grid_search.best_params_>,
  "baseline_score": <baseline_score from input>,
  "tuned_score": <primary metric on X_test>,
  "improvement_delta": <tuned_score - baseline_score>,
  "cv_folds": <cv_folds>,
  "scoring": <scoring>,
  "total_combinations": <total_combinations>,
  "tuned_metrics": {
    "accuracy": ...,
    "f1": ...,
    "precision": ...,
    "recall": ...
  },
  "timestamp": <datetime.now().isoformat()>
}

STEP 7: Final verification
Call verify_output_saved for:
  outputs/tuned_model.pkl
  outputs/tuning_results.json
Print PASS/FAIL per check.

ERROR HANDLING RULES:
If execute_python returns ERROR — fix and retry the step.
If GridSearchCV fails (memory/time) — reduce param_grid to 2 params and retry.
Never evaluate on training data — always evaluate on X_test.
Always use X_train_bal / y_train_bal for fitting GridSearchCV.
"""


ht_executor_agent = Agent(
    name="ht_executor_agent",
    model="gemini-2.0-flash",
    description=(
        "Runs GridSearchCV on the winning model using the strategist param_grid, "
        "evaluates on X_test, saves tuned_model.pkl and tuning_results.json, "
        "and updates sandbox globals."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)


async def run_ht_executor(strategist_output: dict[str, Any]) -> dict[str, Any]:
    """
    Run the HT Executor agent end-to-end.

    Args:
        strategist_output: Output from run_ht_strategist() containing
                           param_grid, cv_folds, scoring, best_model_name,
                           total_combinations, and baseline_score.

    Returns:
        Dict with execution_log, verify records, and tuning summary.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    best_model_name: str = strategist_output.get("best_model_name", "")
    param_grid: dict[str, Any] = strategist_output.get("param_grid", {})
    cv_folds: int = int(strategist_output.get("cv_folds", 5))
    scoring: str = strategist_output.get("scoring", "f1_weighted")
    total_combinations: int = int(strategist_output.get("total_combinations", 27))
    baseline_score: float = float(strategist_output.get("baseline_score", 0.0))

    user_message = json.dumps(
        {
            "best_model_name": best_model_name,
            "param_grid": param_grid,
            "cv_folds": cv_folds,
            "scoring": scoring,
            "total_combinations": total_combinations,
            "baseline_score": baseline_score,
            "instructions": (
                "Follow the 7-step workflow exactly. "
                "Use X_train_bal for fitting — fall back to X_train if missing. "
                "Evaluate only on X_test. "
                "Update globals: tuned_model, best_params, tuned_score."
            ),
        },
        default=str,
    )

    runner = InMemoryRunner(agent=ht_executor_agent)
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
        "best_model_name": best_model_name,
        "cv_folds": cv_folds,
        "scoring": scoring,
        "total_combinations": total_combinations,
        "execution_log": execution_log.strip(),
        "verify_tuned_model": verify_output_saved("outputs/tuned_model.pkl"),
        "verify_tuning_results": verify_output_saved("outputs/tuning_results.json"),
    }
