"""
app/agents/hyperparameter_tuning_agent/strategist_agent.py

Hyperparameter Tuning Strategist Agent — Agent 2 in the HT pipeline.

Input:  Analyzer report + best_model_name + current_hyperparameters + cv_folds.
Output: Exact param_grid JSON (3 params × ≤3 values, total combinations ≤ 27).
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a senior ML engineer specializing in hyperparameter optimization.
You receive a tuning readiness analysis and produce an exact, executable
search space configuration.

You will receive:
Full analyzer report from the HT Analyzer
best_model_name: exact winning model name
current_hyperparameters: current values for the winning model
cv_folds: 3 or 5 (from dataset scale logic)
primary_metric: f1 or accuracy
diminishing_returns: True or False
overfit_risk: High or Low

Your output MUST be a numbered plan covering ALL sections:

1. SEARCH STRATEGY:
   State: GridSearchCV (combinations <= 27)
   State cv_folds to use (from input).
   State scoring metric: f1_weighted if primary_metric=f1, else accuracy.
   State refit=True so the best estimator is automatically refitted.

2. PARAMETER SELECTION RATIONALE:
   State exactly 3 hyperparameters chosen for the winning model.
   For Random Forest   → n_estimators, max_depth, min_samples_split
   For Logistic Regression → C, max_iter, solver
   For XGBoost         → n_estimators, learning_rate, max_depth
   For each parameter state WHY it is high-impact (one sentence).

3. SEARCH GRID (JSON):
   Provide an exact raw JSON block. Rules:
   - Exactly 3 parameters.
   - Each parameter gets exactly 3 values: [current_value, lower_variation, higher_variation].
   - For Random Forest:
       n_estimators: current ± ~33% → [200, 300, 400]  (use current as middle)
       max_depth: current ± 4       → [current-4, current, current+4], min 4
       min_samples_split: [2, 5, 10]
   - For Logistic Regression:
       C: [0.1, 1.0, 10.0] (log scale, current as middle if possible)
       max_iter: [1000, 2000, 3000] (current as middle if possible)
       solver: ["liblinear", "saga", "lbfgs"]
   - For XGBoost:
       n_estimators: [200, 400, 600] (current as middle if possible)
       learning_rate: [0.01, 0.05, 0.1] (current as middle if possible)
       max_depth: [current-2, current, current+2], min 2
   - total_combinations = 3 × 3 × 3 = 27. This MUST equal 27.
   - Output in this exact JSON structure:

   {
     "model": "<best_model_name>",
     "params": {
       "param_1": [val1, val2, val3],
       "param_2": [val1, val2, val3],
       "param_3": [val1, val2, val3]
     },
     "total_combinations": 27
   }

4. RISK MITIGATION:
   If diminishing_returns=True: state that improvements will be marginal (<2%).
   If overfit_risk=High: note that max_depth values should not increase above current.
     In that case, for max_depth use [current-4, current-2, current] instead.
   State final cv_folds and scoring to use in GridSearchCV call.

STRICT RULES:
Do NOT write any Python code.
Output exact numeric values only — no ranges or approximations in the JSON.
total_combinations must always equal exactly 27.
The JSON block must be valid and parseable.
"""


ht_strategist_agent = Agent(
    name="ht_strategist_agent",
    model="gemini-2.0-flash",
    description=(
        "Converts HT analysis into an exact GridSearchCV configuration with "
        "a 3-parameter × 3-value search grid (27 combinations), CV strategy, "
        "and scoring metric."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=[],
)


async def run_ht_strategist(
    analyzer_report: str,
    best_model_name: str,
    current_hyperparameters: dict[str, Any],
    cv_folds: int,
    primary_metric: str,
    diminishing_returns: bool,
    overfit_risk: str,
) -> dict[str, Any]:
    """
    Run the HT Strategist and return the param_grid configuration.

    Args:
        analyzer_report:         Full text from run_ht_analyzer().
        best_model_name:         Winning model name.
        current_hyperparameters: Current model hyperparameters.
        cv_folds:                3 or 5, derived from dataset N.
        primary_metric:          f1 or accuracy.
        diminishing_returns:     True if primary metric > 0.95.
        overfit_risk:            High or Low.

    Returns:
        Dict containing: best_model_name, param_grid, cv_folds,
        scoring, total_combinations, strategy_plan.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    user_message = json.dumps(
        {
            "analyzer_report": analyzer_report,
            "best_model_name": best_model_name,
            "current_hyperparameters": current_hyperparameters,
            "cv_folds": cv_folds,
            "primary_metric": primary_metric,
            "diminishing_returns": diminishing_returns,
            "overfit_risk": overfit_risk,
        },
        default=str,
    )

    runner = InMemoryRunner(agent=ht_strategist_agent)
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

    # Build deterministic param_grid from model name as a safe fallback
    # so the executor always has a valid grid even if LLM output parsing fails.
    fallback_grid = _build_fallback_grid(best_model_name, current_hyperparameters)

    return {
        "best_model_name": best_model_name,
        "cv_folds": cv_folds,
        "scoring": "f1_weighted" if primary_metric == "f1" else "accuracy",
        "param_grid": fallback_grid,      # executor will prefer LLM grid if parseable
        "total_combinations": 27,
        "strategy_plan": plan_text.strip(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic fallback grid builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_fallback_grid(
    model_name: str,
    current_hp: dict[str, Any],
) -> dict[str, list[Any]]:
    """
    Build a safe 3×3×3 param_grid from current hyperparameters.
    Used as fallback if LLM JSON cannot be parsed.
    """
    name = str(model_name).lower()

    if "random forest" in name or "rf" in name:
        n = int(current_hp.get("n_estimators", 300))
        d = int(current_hp.get("max_depth", 12))
        return {
            "n_estimators": [max(100, n - 100), n, n + 100],
            "max_depth": [max(4, d - 4), d, d + 4],
            "min_samples_split": [2, 5, 10],
        }

    if "logistic" in name or "lr" in name:
        return {
            "C": [0.1, 1.0, 10.0],
            "max_iter": [1000, 2000, 3000],
            "solver": ["liblinear", "saga", "lbfgs"],
        }

    # XGBoost default
    n = int(current_hp.get("n_estimators", 400))
    lr = float(current_hp.get("learning_rate", 0.05))
    d = int(current_hp.get("max_depth", 6))
    return {
        "n_estimators": [max(100, n - 200), n, n + 200],
        "learning_rate": [round(lr * 0.2, 4), lr, round(lr * 2, 4)],
        "max_depth": [max(2, d - 2), d, d + 2],
    }
