"""
app/agents/model_evaluation_agent/analyzer_agent.py

Model Evaluation Analyzer Agent — Agent 1 in the ME pipeline.

Input:  Final model metadata + metrics from training and tuning phases.
Output: Structured six-section evaluation facts report.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert ML model evaluation analyst.
Your ONLY job is to analyze the final trained model and report
evaluation facts. Report only facts, no solutions.

You will receive:
final_model_name: name of final model (tuned or original)
final_model_metrics: accuracy, f1, precision, recall from training
tuning_applied: boolean
tuned_metrics: metrics after tuning (or null if not tuned)
original_metrics: metrics before tuning
X_test_shape: shape of test features
y_test_distribution: class distribution of test labels
smote_applied: boolean
task_type: CLASSIFICATION or REGRESSION
best_params: hyperparameters of final model
all_model_comparison: metrics for all 3 trained models
training_shape: shape used for training

Your output MUST cover ALL sections:

SECTION 1 — FINAL MODEL REPORT
State exact final model name
State exact final metrics: accuracy, f1, precision, recall
State whether tuning improved the model or not
State exact improvement delta if tuning was applied
State which model came second and by what margin

SECTION 2 — TEST SET REPORT
State exact X_test shape
State exact class distribution in y_test
State if test set is representative of training distribution
Flag if test set is very small (< 50 rows)
State exact minority class percentage in test set

SECTION 3 — METRICS INTERPRETATION
Accuracy: state exact value and what it means
F1 Score: state exact value — harmonic mean of precision/recall
Precision: state exact value — of predicted positives how many correct
Recall: state exact value — of actual positives how many caught
If imbalanced dataset: flag accuracy as misleading metric
State which metric is most trustworthy for this dataset

SECTION 4 — CONFUSION MATRIX REQUIREMENTS
State exact number of classes
For binary: state TP, TN, FP, FN counts needed
For multiclass: state full NxN matrix needed
State which errors are most costly:
  False Negatives or False Positives
State exact class labels for matrix axes

SECTION 5 — VISUALIZATION REQUIREMENTS
Confusion matrix heatmap: exact filename
  outputs/confusion_matrix.png
ROC curve (if binary classification):
  exact filename outputs/roc_curve.png
Precision-Recall curve:
  exact filename outputs/pr_curve.png
Feature importance (if RF or XGBoost):
  exact filename outputs/feature_importance.png
Model comparison bar chart:
  exact filename outputs/model_comparison.png

SECTION 6 — EVALUATION SUMMARY
Final model: exact name
Overall performance: Poor/Fair/Good/Excellent
  (Poor < 0.7, Fair 0.7-0.8, Good 0.8-0.9, Excellent > 0.9)
Most reliable metric: state which one and why
Visualizations needed: list exact filenames
Evaluation completeness: what can and cannot be concluded

STRICT RULES:
Report ONLY facts with exact numbers
Do NOT write any code
Do NOT make recommendations — only report observations
Use exact metric values throughout
Never interpret results beyond what numbers show
"""


me_analyzer_agent = Agent(
    name="me_analyzer_agent",
    model="gemini-2.0-flash",
    description=(
        "Analyzes final model metrics and test set data to produce a six-section "
        "evaluation facts report covering model performance, test set properties, "
        "metrics interpretation, confusion matrix requirements, visualization "
        "requirements, and evaluation summary."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=[],
)


async def run_me_analyzer(
    final_model_name: str,
    final_model_metrics: dict[str, Any],
    tuning_applied: bool,
    tuned_metrics: dict[str, Any] | None,
    original_metrics: dict[str, Any],
    X_test_shape: list[int],
    y_test_distribution: dict[str, Any],
    smote_applied: bool,
    task_type: str,
    best_params: dict[str, Any],
    all_model_comparison: dict[str, Any],
    training_shape: list[int],
) -> str:
    """
    Run the ME Analyzer and return a six-section evaluation facts report.

    Args:
        final_model_name:     Name of the final model (tuned or original best).
        final_model_metrics:  accuracy, f1, precision, recall of final model.
        tuning_applied:       Whether hyperparameter tuning was run.
        tuned_metrics:        Metrics after tuning, or None if not tuned.
        original_metrics:     Metrics from the model training phase.
        X_test_shape:         Shape of test feature matrix [rows, cols].
        y_test_distribution:  Class label counts in test set.
        smote_applied:        Whether SMOTE was applied during training.
        task_type:            CLASSIFICATION or REGRESSION.
        best_params:          Final model hyperparameters.
        all_model_comparison: Metrics for all trained models.
        training_shape:       Shape used for training [rows, cols].

    Returns:
        Full six-section evaluation report text.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    user_message = json.dumps(
        {
            "final_model_name": final_model_name,
            "final_model_metrics": final_model_metrics,
            "tuning_applied": tuning_applied,
            "tuned_metrics": tuned_metrics,
            "original_metrics": original_metrics,
            "X_test_shape": X_test_shape,
            "y_test_distribution": y_test_distribution,
            "smote_applied": smote_applied,
            "task_type": task_type,
            "best_params": best_params,
            "all_model_comparison": all_model_comparison,
            "training_shape": training_shape,
        },
        default=str,
    )

    runner = InMemoryRunner(agent=me_analyzer_agent)
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
