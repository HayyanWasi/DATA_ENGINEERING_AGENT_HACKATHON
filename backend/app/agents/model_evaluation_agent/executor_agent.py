"""
app/agents/model_evaluation_agent/executor_agent.py

Model Evaluation Executor Agent — Agent 3 in the ME pipeline.

Input:  Strategist plan + model metadata + all_model_comparison dict.
Tool:   execute_python(code: str) -> str
Output: All evaluation plots + evaluation_summary.json saved in outputs/.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.tools.executor_tools import execute_python, verify_output_saved, verify_charts_saved


# ─────────────────────────────────────────────────────────────────────────────
# Tool registration
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS: list[FunctionTool] = [
    FunctionTool(execute_python),
    FunctionTool(verify_output_saved),
    FunctionTool(verify_charts_saved),
]


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert Python ML engineer specializing in
model evaluation and visualization.
You receive an evaluation plan and implement it by writing
and executing Python code using the execute_python tool.

You have access to one tool: execute_python(code: str) -> str
This tool runs Python code in a shared sandbox that already contains:
final_model: best trained model object (tuned or original)
final_model_name: name string of final model
X_test: test features
y_test: test labels
X_train_bal: balanced training features
best_model_name: winning model name
all_results: metrics for all 3 models
tuning_applied: boolean
tuning_results: tuning comparison dict
smote_applied: boolean
target_col: ML target column name
pd, np: pandas and numpy
joblib: for loading models

YOUR WORKFLOW — follow these steps in exact order:

STEP 1: Setup and verify sandbox
Write and run code to confirm final_model,
X_test, y_test are loaded correctly.
Print final_model_name, X_test shape,
y_test class distribution.
Import all required evaluation libraries.
Create outputs/ directory if not exists.

STEP 2: Generate predictions and base metrics
Write and run code to:
Generate predictions: y_pred = final_model.predict(X_test)
Generate probabilities if model supports predict_proba
Calculate accuracy, f1, precision, recall on X_test
Print full metrics comparison table
Store all metrics for JSON export

STEP 3: Generate confusion matrix
Write and run code to:
Compute confusion matrix from y_test and y_pred
For binary: extract and print TP, TN, FP, FN
Create seaborn heatmap with Blues colormap
Annotate with exact counts
Add proper axis labels and title
Save to outputs/confusion_matrix.png
Call plt.close() after saving

STEP 4: Generate classification report
Write and run code to:
Generate full sklearn classification_report
Print it to console
Save text version to outputs/classification_report.txt
Save JSON version to outputs/classification_report.json

STEP 5: Generate ROC curve
Write and run code to:
Check if task is binary classification
If binary and model has predict_proba:
  compute roc_curve and roc_auc_score
  plot ROC curve with AUC in legend
  add diagonal reference line
  save to outputs/roc_curve.png
  call plt.close() after saving
If multiclass or no predict_proba:
  skip and log reason

STEP 6: Generate Precision-Recall curve
Write and run code to:
Compute precision_recall_curve
Calculate average_precision_score
Plot with AP score in legend
Save to outputs/pr_curve.png
Call plt.close() after saving

STEP 7: Generate feature importance chart
Write and run code to:
Check model type: RF/XGBoost use feature_importances_
Logistic Regression uses abs(coef_)
Get feature names from X_test columns
Plot top 15 features horizontal bar chart
Sort descending by importance
Save to outputs/feature_importance.png
Call plt.close() after saving

STEP 8: Generate model comparison chart
Write and run code to:
Read all 3 model metrics from all_results
Create grouped bar chart: accuracy and f1 per model
Highlight winning model in different color
Add value labels on bars
Save to outputs/model_comparison.png
Call plt.close() after saving

STEP 9: Save evaluation summary JSON
Write and run code to build and save complete
evaluation_summary.json to outputs/ containing:
final_model_name
task_type, smote_applied, tuning_applied
final_metrics: accuracy, f1, precision, recall
auc_score: float or null
performance_rating: Poor/Fair/Good/Excellent
confusion_matrix_values: dict
model_comparison: all 3 models metrics
output_files: list of all generated files
This JSON is consumed directly by Next.js UI

STEP 10: Update sandbox globals
Write and run code to update sandbox with:
evaluation_results: complete evaluation dict
y_pred: final predictions
auc_score: float or null
performance_rating: string

STEP 11: Final verification
Write and run code to confirm all files exist:
outputs/confusion_matrix.png
outputs/classification_report.txt
outputs/classification_report.json
outputs/roc_curve.png (if binary)
outputs/pr_curve.png
outputs/feature_importance.png
outputs/model_comparison.png
outputs/evaluation_summary.json
Print PASS or FAIL for each file

ERROR HANDLING RULES:
If any single chart fails — skip it, log it, continue
If ROC fails — skip gracefully, set auc_score=null
If feature importance fails — skip, log model type issue
Maximum 3 retries per step
Never let one chart failure stop entire pipeline
evaluation_summary.json must always be saved

STRICT RULES:
Always verify sandbox in Step 1
Always call plt.close() after every chart
Always use tight_layout() before savefig()
Always evaluate on X_test only
Always save evaluation_summary.json — UI depends on it
Print results after every step
Agent writes all code — no pre-written code in prompt
"""


me_executor_agent = Agent(
    name="me_executor_agent",
    model="gemini-2.0-flash",
    description=(
        "Implements the full 11-step model evaluation pipeline: verifies sandbox, "
        "computes predictions and metrics, generates confusion matrix, classification "
        "report (txt + JSON), ROC/PR curves, feature importance, model comparison "
        "chart, saves evaluation_summary.json, and updates sandbox globals."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)


async def run_me_executor(strategist_output: dict[str, Any]) -> dict[str, Any]:
    """
    Run the ME Executor agent end-to-end.

    Args:
        strategist_output: Output from run_me_strategist() containing
                           evaluation_plan, plots_to_generate, final_model_name,
                           is_binary, class_count, smote_applied, tuning_applied,
                           all_model_comparison.

    Returns:
        Dict with execution_log, verify records, and plots summary.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    final_model_name: str = strategist_output.get("final_model_name", "")
    task_type: str = strategist_output.get("task_type", "CLASSIFICATION")
    is_binary: bool = bool(strategist_output.get("is_binary", True))
    class_count: int = int(strategist_output.get("class_count", 2))
    smote_applied: bool = bool(strategist_output.get("smote_applied", False))
    tuning_applied: bool = bool(strategist_output.get("tuning_applied", False))
    plots_to_generate: list[str] = strategist_output.get("plots_to_generate", [])
    all_model_comparison: dict[str, Any] = strategist_output.get(
        "all_model_comparison", {}
    )

    tuning_results: dict[str, Any] = strategist_output.get("tuning_results", {})

    user_message = json.dumps(
        {
            "final_model_name": final_model_name,
            "task_type": task_type,
            "is_binary": is_binary,
            "class_count": class_count,
            "smote_applied": smote_applied,
            "tuning_applied": tuning_applied,
            "tuning_results": tuning_results,
            "plots_to_generate": plots_to_generate,
            "text_outputs": strategist_output.get("text_outputs", []),
            "all_model_comparison": all_model_comparison,
            "evaluation_plan": strategist_output.get("evaluation_plan", ""),
            "instructions": (
                "Follow the 11-step workflow exactly. "
                "Use final_model from sandbox (tuned_model if available, else best_model). "
                "Evaluate only on X_test. "
                "Skip any chart that fails — log it and continue. "
                "Maximum 3 retries per step. "
                "Always save evaluation_summary.json — UI depends on it."
            ),
        },
        default=str,
    )

    runner = InMemoryRunner(agent=me_executor_agent)
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
        "final_model_name": final_model_name,
        "execution_log": execution_log.strip(),
        "verify_confusion_matrix": verify_output_saved("outputs/confusion_matrix.png"),
        "verify_model_comparison": verify_output_saved("outputs/model_comparison.png"),
        "verify_classification_report_txt": verify_output_saved(
            "outputs/classification_report.txt"
        ),
        "verify_classification_report_json": verify_output_saved(
            "outputs/classification_report.json"
        ),
        "verify_evaluation_summary": verify_output_saved(
            "outputs/evaluation_summary.json"
        ),
        "charts_summary": verify_charts_saved("outputs"),
    }
