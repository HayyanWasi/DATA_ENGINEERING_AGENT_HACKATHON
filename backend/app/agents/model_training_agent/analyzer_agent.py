"""
app/agents/model_training_agent/analyzer_agent.py

Model Training Analyzer Agent — Agent 1 (Child) in the Model Training pipeline.

Input:  Scaled train/test metadata and target class statistics.
Output: Structured six-section model training readiness report.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — model-training analyst persona
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert ML model selection analyst and training analyst.
Your ONLY job is to analyze scaled train/test data and report
facts needed to select and train the right models.

You will receive:
X_train_shape: shape of training features
X_test_shape: shape of test features
y_train_distribution: value counts of training labels
y_test_distribution: value counts of test labels
feature_count: total number of features
feature_names: list of all feature column names
target_col: ML target column name
target_dtype: dtype of target column
class_counts: exact count per class in y_train
class_percentages: percentage per class in y_train
minority_class_percentage: percentage of minority class
task_type: CLASSIFICATION or REGRESSION

Your output MUST cover ALL sections:

SECTION 1 — TASK TYPE
State clearly: CLASSIFICATION or REGRESSION
For classification: binary or multiclass
State exact number of classes with their labels
State exact class distribution with percentages

SECTION 2 — CLASS IMBALANCE REPORT
State exact minority class percentage
If minority class < 20%: IMBALANCED — SMOTE required
If minority class >= 20%: BALANCED — SMOTE not required
If regression: NOT APPLICABLE
State exact count of each class in y_train

SECTION 3 — DATASET SIZE REPORT
State exact train size and test size
State total feature count
Flag if dataset < 100 rows: small dataset warning
Flag if features > train rows: high dimensionality warning
State if dataset is sufficient for all 3 models

SECTION 4 — FEATURE QUALITY REPORT
State if any NaN values remain in X_train
State if any infinite values exist in X_train
State if all features are numeric
Flag any non-numeric columns still present

SECTION 5 — MODEL SUITABILITY REPORT
Random Forest: suitable if features > 5 and rows > 100
Logistic Regression: suitable if features scaled, no multicollinearity
XGBoost: suitable for any size dataset
State suitability for each with exact reason

SECTION 6 — TRAINING SUMMARY
Exact training input shape
SMOTE needed: yes or no with exact reason
Primary evaluation metric: f1 if imbalanced, accuracy if balanced
Expected output: 3 trained models, 1 best model

STRICT RULES:
Report ONLY facts with exact numbers
Do NOT write any code
Do NOT select models — only report suitability
Use exact numbers from data throughout
"""


mt_analyzer_agent = Agent(
    name="mt_analyzer_agent",
    model="gemini-2.0-flash",
    description=(
        "Analyzes scaled train/test shapes and target class statistics to produce a "
        "fact-only training analysis report with section coverage for imbalance, "
        "dataset size, feature quality, and training readiness."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=[],
)


async def run_mt_analyzer(
    X_train_shape: list[int] | tuple[int, int],
    X_test_shape: list[int] | tuple[int, int],
    y_train_distribution: dict[str, Any],
    y_test_distribution: dict[str, Any],
    feature_count: int,
    feature_names: list[str],
    target_col: str,
    target_dtype: str,
    class_counts: dict[str, Any],
    class_percentages: dict[str, Any],
    minority_class_percentage: float,
    task_type: str,
) -> str:
    """
    Run the Model Training Analyzer and return a full six-section analysis text.

    Args:
        X_train_shape:             Training feature shape, e.g. (rows, cols).
        X_test_shape:              Test feature shape, e.g. (rows, cols).
        y_train_distribution:       Distribution of labels in training split.
        y_test_distribution:       Distribution of labels in test split.
        feature_count:             Number of feature columns.
        feature_names:             Feature names used by model training.
        target_col:                ML target column.
        target_dtype:              Dtype string for target column.
        class_counts:              Exact class counts from y_train.
        class_percentages:         Class percentages from y_train.
        minority_class_percentage:  Minority class percentage value.
        task_type:                 CLASSIFICATION or REGRESSION.

    Returns:
        Full model-training analysis report text from the agent.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    user_message = json.dumps(
        {
            "X_train_shape": X_train_shape,
            "X_test_shape": X_test_shape,
            "y_train_distribution": y_train_distribution,
            "y_test_distribution": y_test_distribution,
            "feature_count": feature_count,
            "feature_names": feature_names,
            "target_col": target_col,
            "target_dtype": target_dtype,
            "class_counts": class_counts,
            "class_percentages": class_percentages,
            "minority_class_percentage": minority_class_percentage,
            "task_type": task_type,
        },
        default=str,
    )

    runner = InMemoryRunner(agent=mt_analyzer_agent)
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
