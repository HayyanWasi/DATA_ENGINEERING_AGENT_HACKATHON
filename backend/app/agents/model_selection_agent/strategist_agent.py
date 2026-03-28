"""
app/agents/model_selection_agent/strategist_agent.py

Model Selection Strategist Agent — Agent 2 in the model-selection and training
pipeline.

Input:  Full analysis report from Model Training Analyzer + target column,
        feature names, task type, and minority-class percentage.
Output: A numbered execution plan for SMOTE use and exact model hyperparameters,
        comparison strategy, and saving plan.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — model training strategist persona
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a senior ML engineer specializing in model selection
and training strategy.
You receive an analysis report and produce an exact training plan.

You will receive:
Full analysis report from Model Training Analyzer
target_col: ML target column name
feature_names: list of all feature names
task_type: CLASSIFICATION or REGRESSION
minority_class_percentage: float

Your output MUST be a numbered plan covering ALL sections:

1. SMOTE DECISION:
   - State clearly: APPLY or SKIP
   - If minority < 20%: APPLY SMOTE, random_state=42
   - If minority >= 20%: SKIP, state reason
   - If regression: SKIP, state reason
   - State expected class distribution after SMOTE if applied

2. MODEL 1 — RANDOM FOREST:
   - State exact hyperparameters:
     n_estimators, max_depth, min_samples_split,
     random_state, class_weight
     Use fixed values:
     n_estimators=300, max_depth=12, min_samples_split=2,
     random_state=42, class_weight="balanced"
   - State variable name: rf_model
   - State evaluation metrics to collect

3. MODEL 2 — LOGISTIC REGRESSION:
   - State exact hyperparameters:
     max_iter, random_state, class_weight, solver
     Use fixed values:
     max_iter=2000, random_state=42, class_weight="balanced",
     solver="liblinear"
   - State variable name: lr_model
   - State evaluation metrics to collect

4. MODEL 3 — XGBOOST:
   - State exact hyperparameters:
     n_estimators, learning_rate, max_depth,
     random_state, scale_pos_weight, eval_metric
     Use fixed values:
     n_estimators=400, learning_rate=0.05, max_depth=6,
     random_state=42, scale_pos_weight is 1.0 for multiclass
     or scale_pos_weight = round((100 - minority_class_percentage)/minority_class_percentage, 4)
     for binary, and
     eval_metric="logloss" for binary or "mlogloss" for multiclass.
   - State variable name: xgb_model
   - State evaluation metrics to collect

5. COMPARISON AND SELECTION STRATEGY:
   - Primary metric: f1_weighted if imbalanced, accuracy if balanced
   - Secondary metrics: precision, recall
   - Winner selection logic: highest primary metric wins
   - Tie-breaking: use f1 score

6. SAVING PLAN:
   - All 3 models: outputs/model_rf.pkl,
     outputs/model_lr.pkl, outputs/model_xgb.pkl
   - Best model: outputs/final_model.pkl
   - Results JSON: outputs/training_results.json
   - Sandbox updates required:
     best_model, best_model_name,
     best_predictions, all_results, smote_applied

STRICT RULES:
Do NOT write any code
Give exact hyperparameter values for every model
State SMOTE decision with clear reasoning
If imbalanced — f1 is primary metric, not accuracy
target_col is already in sandbox
x_train and y_train are not passed separately
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

model_selection_strategist_agent = Agent(
    name="model_selection_strategist_agent",
    model="gemini-2.0-flash",
    description=(
        "Converts model-selection analysis into an exact training plan with "
        "SMOTE decision, fixed hyperparameters for 3 candidate models, "
        "comparison logic, and exact output persistence plan."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=[],
)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by model-selection orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_model_selection_strategist(
    analyzer_report: str | dict[str, Any],
    target_col: str,
    feature_names: list[str],
    task_type: str,
    minority_class_percentage: float,
) -> dict[str, Any]:
    """
    Run the Model Selection Strategist and return a full numbered training plan.

    Args:
        analyzer_report:         Full report text from the model selection analyzer
                                 or a dict containing "analysis_report".
        target_col:              ML target column name.
        feature_names:           Full list of feature names.
        task_type:               CLASSIFICATION or REGRESSION.
        minority_class_percentage: Percentage of the minority class in y_train.

    Returns:
        dict with keys:
          target_col:               target column passed in
          feature_count:            number of features
          task_type:                classification or regression
          minority_class_percentage: input minority percentage
          training_plan:            plain-text training strategy text
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    report_text = str(analyzer_report)
    if isinstance(analyzer_report, dict):
        report_text = str(analyzer_report.get("analysis_report", report_text))

    user_message = json.dumps(
        {
            "analysis_report": report_text,
            "target_col": target_col,
            "feature_names": feature_names,
            "task_type": task_type,
            "minority_class_percentage": minority_class_percentage,
        },
        default=str,
    )

    runner = InMemoryRunner(agent=model_selection_strategist_agent)
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
        "feature_count": len(feature_names),
        "task_type": task_type,
        "minority_class_percentage": minority_class_percentage,
        "training_plan": plan_text.strip(),
    }
