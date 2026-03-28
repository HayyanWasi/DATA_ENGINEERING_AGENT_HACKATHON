"""
app/agents/ci/ci_analyzer_agent.py

Class Imbalance Analyzer Agent — Agent 1 (Child) in the Class Imbalance
Pipeline.

Input:  Train and test target series + target metadata.
Tool:   analyze_class_imbalance()
Output: Structured report on class imbalance severity and balancing feasibility.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.tools.ci_analysis_tools import analyze_class_imbalance


# ─────────────────────────────────────────────────────────────────────────────
# Tool registration
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS: list[FunctionTool] = [
    FunctionTool(analyze_class_imbalance),
]


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — class imbalance analyst persona
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert ML data balance analyst.
Your ONLY job is to analyze class distribution in training labels
and report facts about imbalance. Report only facts, no solutions.

You will receive:
y_train_distribution: value counts of training labels
y_train_percentages: percentage of each class in y_train
y_test_distribution: value counts of test labels
total_train_samples: total rows in y_train
total_test_samples: total rows in y_test
target_col: ML target column name
target_dtype: dtype of target column
unique_classes: list of unique class labels
class_count: total number of unique classes

Your output MUST cover ALL sections:

SECTION 1 — TASK TYPE CONFIRMATION
State clearly: CLASSIFICATION or REGRESSION
If regression: flag as NOT APPLICABLE for balancing
For classification: binary (2 classes) or multiclass (3+ classes)
State exact number of classes with their labels

SECTION 2 — CLASS DISTRIBUTION REPORT
State exact count and percentage for every class
Identify majority class: name and exact percentage
Identify minority class: name and exact percentage
State exact imbalance ratio: majority/minority count

SECTION 3 — IMBALANCE SEVERITY
Calculate minority class percentage
If minority >= 40%: BALANCED — no action needed
If minority 20-40%: MILD IMBALANCE — borderline
If minority 10-20%: MODERATE IMBALANCE — action needed
If minority < 10%: SEVERE IMBALANCE — urgent action needed
State exact severity label with exact percentage

SECTION 4 — DATASET SIZE IMPACT
State total training samples
If total samples < 100: flag as too small for SMOTE
If total samples 100-500: flag as small dataset
If total samples > 500: flag as sufficient for SMOTE
State minimum samples in minority class
If minority class samples < 6: SMOTE not possible

SECTION 5 — MULTICLASS REPORT
If binary: state which class is positive and which is negative
If multiclass: state each class count and percentage
Identify if one-vs-rest strategy is needed
State if any class has fewer than 10 samples

SECTION 6 — SUMMARY
Imbalance action needed: YES or NO
Severity: BALANCED / MILD / MODERATE / SEVERE
Dataset size for balancing: SUFFICIENT / TOO SMALL
SMOTE feasibility: POSSIBLE / NOT POSSIBLE with exact reason
Recommended approach basis: oversample / undersample / skip

STRICT RULES:
Report ONLY facts with exact numbers
Do NOT write any code
Do NOT make final technique decision — only report observations
If regression task: report NOT APPLICABLE and stop
Use exact class labels from the data
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

ci_analyzer_agent = Agent(
    name="ci_analyzer_agent",
    model="gemini-2.0-flash",
    description=(
        "Profiles training and test target distributions and writes a structured "
        "six-section class imbalance report with imbalance severity, dataset-size "
        "impact, multiclass summary, and balancing feasibility observations."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by CI orchestrator or API route
# ─────────────────────────────────────────────────────────────────────────────

async def run_ci_analyzer(
    y_train: list[Any],
    y_test: list[Any],
    target_col: str,
    target_dtype: str,
) -> str:
    """
    Run the Class Imbalance Analyzer and return the full imbalance analysis report.

    Args:
        y_train:      Training labels as a list-like sequence.
        y_test:       Test labels as a list-like sequence.
        target_col:   ML target column name.
        target_dtype: Dtype of target column.

    Returns:
        Full class-imbalance analysis report text written by the agent.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    user_message = json.dumps(
        {
            "y_train": y_train,
            "y_test": y_test,
            "target_col": target_col,
            "target_dtype": target_dtype,
        },
        default=str,
    )

    runner = InMemoryRunner(agent=ci_analyzer_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="system",
    )

    report_text = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            report_text = event.content.parts[0].text or ""

    return report_text.strip()
