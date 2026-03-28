"""
app/agents/model_training_agent/executor_agent.py

Model Training Implementor Agent — Agent 3 in the Model Training pipeline.

Input:  Model training strategy + training data already in sandbox.
Tool:   execute_python(code: str) -> str
Output: Trained model artifacts and training result JSON saved in outputs.
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

def _extract_smote_required(plan_text: str) -> bool:
    """
    Parse whether the strategist plan applies SMOTE.
    """
    txt = str(plan_text or "").upper()
    if re.search(r"SECTION\s*1", txt) and "SMOTE DECISION" in txt:
        section1 = txt.split("SECTION 2")[0] if "SECTION 2" in txt else txt
        if re.search(r"\bAPPLY\b", section1):
            return True
        if re.search(r"\bSKIP\b", section1):
            return False
        # fallback: explicit short phrase in section 1
        return "APPLY SMOTE" in section1
    # fallback if only a short plan
    return "SMOTE" in txt and "APPLY" in txt


def _extract_plan_smote_rate(plan_text: str, default: int | float = 42) -> int:
    """
    Extract an integer hyperparameter from plan text.
    """
    if not plan_text:
        return int(default)
    match = re.search(r"random_state\s*=\s*([0-9]+)", str(plan_text), re.IGNORECASE)
    if not match:
        return int(default)
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return int(default)


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — exact implementation workflow
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert Python ML engineer.
You receive a training plan and implement it by writing and executing Python code
using the execute_python tool.

You have access to one tool: execute_python(code: str) -> str
This tool runs Python code in a shared sandbox that already contains:

X_train: scaled training features
X_test: scaled test features
y_train: training labels
y_test: test labels
target_col: ML target column name
pd, np: pandas and numpy
RandomForestClassifier, LogisticRegression, XGBClassifier
classification_report, confusion_matrix
joblib: for saving models

You must follow this EXACT workflow and strict constraints.

STEP 1: Verify sandbox state
Write and run code to confirm X_train, X_test, y_train, y_test are loaded correctly.
Print shapes and y_train class distribution.
Print target_col value.
If anything missing, stop and report.

STEP 2: Handle class imbalance
Check minority class percentage from y_train value counts.
If minority < 20%:
  - Import SMOTE from imblearn.over_sampling
  - Apply SMOTE with random_state=42
  - Store X_train_bal, y_train_bal
  - Set smote_applied = True
If minority >= 20%:
  - Set X_train_bal = X_train and y_train_bal = y_train
  - Set smote_applied = False
  - Print that data is balanced or skip balancing

STEP 3: Train Random Forest
Train RandomForestClassifier with exact hyperparameters:
n_estimators=300, max_depth=12, min_samples_split=2,
random_state=42, class_weight="balanced"
Save rf_model to outputs/model_rf.pkl.
Print accuracy, f1, precision, recall.

STEP 4: Train Logistic Regression
Train LogisticRegression with exact hyperparameters:
max_iter=2000, random_state=42, class_weight="balanced", solver="liblinear"
Save lr_model to outputs/model_lr.pkl.
Print accuracy, f1, precision, recall.

STEP 5: Train XGBoost
Use XGBClassifier with exact hyperparameters:
n_estimators=400, learning_rate=0.05, max_depth=6,
random_state=42, scale_pos_weight and eval_metric from plan.
Save xgb_model to outputs/model_xgb.pkl.
Print accuracy, f1, precision, recall.

STEP 6: Select best model
If all models train, select by:
primary metric f1_weighted when imbalanced, accuracy when balanced.
Tie-break by f1.
Save best_model to outputs/final_model.pkl.
Update sandbox globals:
best_model, best_model_name, best_predictions, all_results, smote_applied

STEP 7: Save training results
Write outputs/training_results.json with:
status per model, all metrics, winner name/metrics, train/test shapes,
smote_applied flag, and timestamp.

STEP 8: Final verification
Verify files exist:
outputs/model_rf.pkl, outputs/model_lr.pkl, outputs/model_xgb.pkl,
outputs/final_model.pkl, outputs/training_results.json.
Print PASS/FAIL per check.

ERROR HANDLING RULES:
If execute_python returns ERROR — fix and retry the step
If SMOTE fails — skip it and continue with X_train, y_train
If one model fails — skip it and continue with remaining models
Never evaluate on training data — always evaluate on X_test only
Always train using X_train_bal and y_train_bal if available
Always update globals() after Step 6
"""


mt_implementor_agent = Agent(
    name="mt_implementor_agent",
    model="gemini-2.0-flash",
    description=(
        "Implements the model-training strategy end-to-end with optional SMOTE, "
        "trains RandomForest, Logistic Regression and XGBoost, saves model artifacts, "
        "writes training_results.json and updates sandbox globals."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)


# Backward-compatible alias
mt_executor_agent = mt_implementor_agent


async def run_mt_implementor(strategist_output: dict[str, Any]) -> dict[str, Any]:
    """
    Run the Model Training implementor agent and return artifact summary.

    Args:
        strategist_output: Output from run_mt_strategist() containing training plan
                          and strategy metadata.

    Returns:
        Dict with execution logs and file-verify records used by the
        Model Training orchestrator.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    target_col: str = strategist_output.get("target_col", "")
    training_plan: str = strategist_output.get("training_plan", "")
    task_type: str = str(strategist_output.get("task_type", "CLASSIFICATION")).upper()
    minority_class_percentage = strategist_output.get("minority_class_percentage", 100.0)
    feature_count: int = int(strategist_output.get("feature_count", 0))
    class_counts = strategist_output.get("class_counts", {})
    class_count = (
        len(class_counts)
        if isinstance(class_counts, dict)
        else int(strategist_output.get("class_count", 1))
    )
    class_count = 1 if class_count <= 0 else int(class_count)

    apply_smote = _extract_smote_required(training_plan)
    if task_type != "CLASSIFICATION":
        apply_smote = False
    else:
        try:
            apply_smote = float(minority_class_percentage) < 20.0
        except (TypeError, ValueError):
            pass
    random_state = _extract_plan_smote_rate(training_plan, 42)

    scale_pos_weight = 1.0
    if class_count <= 2:
        try:
            minority_pct = float(minority_class_percentage)
            if minority_pct > 0:
                scale_pos_weight = round((100.0 - minority_pct) / minority_pct, 4)
        except (TypeError, ValueError):
            scale_pos_weight = 1.0

    user_message = json.dumps(
        {
            "target_col": target_col,
            "training_plan": training_plan,
            "task_type": task_type,
            "feature_count": feature_count,
            "minority_class_percentage": minority_class_percentage,
            "class_count": class_count,
            "apply_smote": apply_smote,
            "smote_random_state": random_state,
            "rf_params": {
                "n_estimators": 300,
                "max_depth": 12,
                "min_samples_split": 2,
                "random_state": 42,
                "class_weight": "balanced",
            },
            "lr_params": {
                "max_iter": 2000,
                "random_state": 42,
                "class_weight": "balanced",
                "solver": "liblinear",
            },
            "xgb_params": {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": 6,
                "random_state": 42,
                "scale_pos_weight": scale_pos_weight,
                "eval_metric": "mlogloss" if class_count > 2 else "logloss",
            },
            "instructions": (
                "Follow strategy exactly. "
                "Do not override selected technique unless SMOTE fails. "
                "Do not touch X_test/y_test. "
                "Use X_train_bal/y_train_bal for fitting, and evaluate only on X_test."
            ),
        },
        default=str,
    )

    runner = InMemoryRunner(agent=mt_implementor_agent)
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
        "task_type": task_type,
        "apply_smote": bool(apply_smote),
        "training_plan": training_plan,
        "execution_log": execution_log.strip(),
        "smote_applied": bool(apply_smote),
        "verify_model_rf": verify_output_saved("outputs/model_rf.pkl"),
        "verify_model_lr": verify_output_saved("outputs/model_lr.pkl"),
        "verify_model_xgb": verify_output_saved("outputs/model_xgb.pkl"),
        "verify_final_model": verify_output_saved("outputs/final_model.pkl"),
        "verify_training_results": verify_output_saved("outputs/training_results.json"),
    }
