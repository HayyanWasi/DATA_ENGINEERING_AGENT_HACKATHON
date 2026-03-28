"""
app/agents/model_evaluation_agent/strategist_agent.py

Model Evaluation Strategist Agent — Agent 2 in the ME pipeline.

Input:  Analyzer report + model metadata.
Output: Exact evaluation execution plan — which plots to generate,
        which metrics to compute, exact code steps for the executor.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a senior ML engineer specializing in model evaluation
and result visualization.
You receive an evaluation analysis and create an exact
evaluation and visualization plan.

You will receive:
Full analysis report from Model Evaluation Analyzer
final_model_name: name of final model
task_type: CLASSIFICATION or REGRESSION
smote_applied: boolean
class_count: number of unique classes
y_test_distribution: class distribution
visualization_requirements: from analyzer report

Your output MUST be a numbered plan covering ALL sections:

1. CONFUSION MATRIX PLAN:
   - State exact computation method
   - For binary: extract TP, TN, FP, FN explicitly
   - For multiclass: full NxN matrix
   - Visualization: seaborn heatmap, Blues colormap
   - Annotate with exact counts
   - Add row/column labels from actual class names
   - Filename: outputs/confusion_matrix.png

2. CLASSIFICATION REPORT PLAN:
   - Generate full sklearn classification_report
   - Include: precision, recall, f1 per class
   - Include: macro avg, weighted avg
   - Save as text to outputs/classification_report.txt
   - Save as JSON to outputs/classification_report.json

3. ROC CURVE PLAN (binary classification only):
   - Compute predict_proba on X_test
   - Calculate fpr, tpr, thresholds
   - Calculate exact AUC score
   - Plot ROC curve with AUC in legend
   - Add diagonal reference line (random classifier)
   - Filename: outputs/roc_curve.png
   - If multiclass: skip ROC, note in report

4. PRECISION-RECALL CURVE PLAN:
   - Compute precision_recall_curve on X_test
   - Calculate average precision score
   - Plot with AP score in legend
   - Filename: outputs/pr_curve.png
   - More informative than ROC for imbalanced data

5. FEATURE IMPORTANCE PLAN:
   - If Random Forest or XGBoost:
     Extract feature_importances_ attribute
     Plot top 15 features as horizontal bar chart
     Sort by importance descending
     Filename: outputs/feature_importance.png
   - If Logistic Regression:
     Extract coef_ as feature importance
     Plot absolute coefficient values
     Filename: outputs/feature_importance.png

6. MODEL COMPARISON CHART PLAN:
   - Bar chart comparing all 3 models
   - Show accuracy and f1 side by side per model
   - Highlight winning model in different color
   - Filename: outputs/model_comparison.png

7. EVALUATION SUMMARY JSON PLAN:
   - Save complete evaluation results to
     outputs/evaluation_summary.json containing:
     final_model_name, task_type, smote_applied,
     tuning_applied, all metrics, AUC score,
     confusion matrix values, performance rating,
     list of all output files generated
   - This is consumed by Next.js frontend and report.pdf

STRICT RULES:
Do NOT write any code
Give exact filenames for every visualization
All files saved to outputs/ folder
Always use plt.close() after every chart
Always use tight_layout() before saving
evaluation_summary.json is mandatory — UI depends on it
"""


me_strategist_agent = Agent(
    name="me_strategist_agent",
    model="gemini-2.0-flash",
    description=(
        "Converts ME analysis into an exact 7-section evaluation and visualization "
        "plan covering confusion matrix, classification report (txt + JSON), ROC/PR "
        "curves, feature importance, model comparison chart, and evaluation_summary.json."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=[],
)


async def run_me_strategist(
    analyzer_report: str,
    final_model_name: str,
    task_type: str,
    smote_applied: bool,
    class_count: int,
    y_test_distribution: dict[str, Any],
    tuning_applied: bool,
    visualization_requirements: str = "",
) -> dict[str, Any]:
    """
    Run the ME Strategist and return the evaluation execution plan.

    Args:
        analyzer_report:            Full text from run_me_analyzer().
        final_model_name:           Winning model name.
        task_type:                  CLASSIFICATION or REGRESSION.
        smote_applied:              Whether SMOTE was applied.
        class_count:                Number of unique classes.
        y_test_distribution:        Class label counts in test set.
        tuning_applied:             Whether tuning was applied.
        visualization_requirements: Section 5 text extracted from analyzer report.

    Returns:
        Dict containing: evaluation_plan, plots_to_generate,
        text_outputs, final_model_name.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    is_binary = class_count == 2

    user_message = json.dumps(
        {
            "analyzer_report": analyzer_report,
            "final_model_name": final_model_name,
            "task_type": task_type,
            "smote_applied": smote_applied,
            "class_count": class_count,
            "is_binary": is_binary,
            "y_test_distribution": y_test_distribution,
            "tuning_applied": tuning_applied,
            "visualization_requirements": visualization_requirements,
        },
        default=str,
    )

    runner = InMemoryRunner(agent=me_strategist_agent)
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

    # Deterministic outputs list — executor always has a known set regardless of LLM
    plots = [
        "outputs/confusion_matrix.png",
        "outputs/pr_curve.png",
        "outputs/model_comparison.png",
    ]
    if is_binary and task_type.upper() == "CLASSIFICATION":
        plots.append("outputs/roc_curve.png")
    model_lower = final_model_name.lower()
    if any(k in model_lower for k in ("random forest", "xgb", "rf")):
        plots.append("outputs/feature_importance.png")

    text_outputs = [
        "outputs/classification_report.txt",
        "outputs/classification_report.json",
        "outputs/evaluation_summary.json",   # mandatory — UI depends on it
    ]

    return {
        "final_model_name": final_model_name,
        "task_type": task_type,
        "is_binary": is_binary,
        "class_count": class_count,
        "smote_applied": smote_applied,
        "tuning_applied": tuning_applied,
        "plots_to_generate": plots,
        "text_outputs": text_outputs,
        "evaluation_plan": plan_text.strip(),
    }
