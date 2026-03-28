"""
app/agents/hyperparameter_tuning_agent/analyzer_agent.py

Hyperparameter Tuning Analyzer Agent — Agent 1 in the HT pipeline.

Input:  training_results dict from the Model Training phase.
Output: Structured four-section tuning readiness report.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert ML hyperparameter tuning analyst.
Your ONLY job is to analyze model training results and report
the exact facts needed to configure a targeted hyperparameter search.

You will receive:
best_model_name: name of the winning model
best_model_metrics: accuracy, f1, precision, recall of winner
model_comparison: metrics for all trained models
current_hyperparameters: the exact hyperparameters used during training
X_train_bal_shape: shape of the balanced training data [N, features]
smote_applied: whether SMOTE was applied
task_type: CLASSIFICATION or REGRESSION

Your output MUST cover ALL four sections:

SECTION 1 — MODEL IDENTIFICATION
State the winning model name exactly.
State the runner-up model name and its primary metric score.
Compute and state the delta: winner_score - runner_up_score.
State the primary metric used (f1 if smote_applied else accuracy).
Example: Delta = 0.93 - 0.90 = 0.03

SECTION 2 — DATASET SCALE
State exact N (rows in X_train_bal_shape[0]).
Apply these rules and state result:
  N < 200  → CV=3, Flag: Small
  200 <= N <= 1000 → CV=5, Flag: Medium
  N > 1000 → CV=5, Flag: Large
State exact feature count (X_train_bal_shape[1]).
State expected GridSearch time: Low (N<200), Medium (N<=1000), High (N>1000).

SECTION 3 — CURRENT HYPERPARAMETER AUDIT
List every current hyperparameter of the winning model with its exact value.
For Random Forest: n_estimators, max_depth, min_samples_split, random_state, class_weight
For Logistic Regression: C, max_iter, solver, random_state, class_weight
For XGBoost: n_estimators, learning_rate, max_depth, random_state, scale_pos_weight, eval_metric
Flag if max_depth > 10: Overfit Risk = High
Flag if max_depth <= 10: Overfit Risk = Low
For Logistic Regression, flag if C > 5: Overfit Risk = High, else Low

SECTION 4 — TUNING READINESS FLAGS
Diminishing Returns: TRUE if primary metric > 0.95, else FALSE
Overfit Risk: High or Low (from Section 3)
Complexity Status:
  VOLATILE if smote_applied=True AND N < 200
  STABLE otherwise
State whether GridSearch or RandomizedSearch is recommended:
  GridSearch if total_combinations <= 27
  RandomizedSearch if total_combinations > 27
State the 3 highest-impact hyperparameters to tune for the winning model:
  Random Forest   → n_estimators, max_depth, min_samples_split
  Logistic Regression → C, max_iter, solver
  XGBoost         → n_estimators, learning_rate, max_depth

STRICT RULES:
Report ONLY facts with exact numbers from the input.
Do NOT write any code.
Do NOT suggest hyperparameter values — only audit current ones.
Use exact numbers throughout.
"""


ht_analyzer_agent = Agent(
    name="ht_analyzer_agent",
    model="gemini-2.0-flash",
    description=(
        "Analyzes model training results to produce a four-section tuning readiness "
        "report covering model identification, dataset scale, current hyperparameter "
        "audit, and tuning risk flags."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=[],
)


async def run_ht_analyzer(
    best_model_name: str,
    best_model_metrics: dict[str, Any],
    model_comparison: dict[str, Any],
    current_hyperparameters: dict[str, Any],
    X_train_bal_shape: list[int],
    smote_applied: bool,
    task_type: str,
) -> str:
    """
    Run the HT Analyzer and return a four-section tuning readiness report.

    Args:
        best_model_name:         Winning model name from training phase.
        best_model_metrics:      Winner's accuracy, f1, precision, recall.
        model_comparison:        All models and their metrics.
        current_hyperparameters: Exact hyperparameters used for the winning model.
        X_train_bal_shape:       Shape of balanced training data [N, features].
        smote_applied:           Whether SMOTE was applied.
        task_type:               CLASSIFICATION or REGRESSION.

    Returns:
        Full four-section analysis report text.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    user_message = json.dumps(
        {
            "best_model_name": best_model_name,
            "best_model_metrics": best_model_metrics,
            "model_comparison": model_comparison,
            "current_hyperparameters": current_hyperparameters,
            "X_train_bal_shape": X_train_bal_shape,
            "smote_applied": smote_applied,
            "task_type": task_type,
        },
        default=str,
    )

    runner = InMemoryRunner(agent=ht_analyzer_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="system",
    )

    report = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            report = event.content.parts[0].text or ""

    return report.strip()
