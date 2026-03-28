"""
app/agents/mt_orchestrator_agent.py

Model Training Orchestrator — top-level manager of training and model selection.

Architecture:
  ┌───────────────────────────────────────────────────────────────┐
  │            Model Training Orchestrator                      │
  │         4-phase pipeline (Analyze → Strategize →        ... )│
  └───────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from google.adk.agents import Agent

from app.agents.model_training_agent.analyzer_agent import (
    mt_analyzer_agent,
    run_mt_analyzer,
)
from app.agents.model_training_agent.executor_agent import (
    mt_implementor_agent,
    run_mt_implementor,
)
from app.agents.model_training_agent.strategist_agent import (
    mt_strategist_agent,
    run_mt_strategist,
)
from app.tools.executor_tools import execute_python


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_ORCHESTRATOR_PROMPT = """
You are the Model Training Orchestrator.
You manage a 3-agent pipeline to train ML models on scaled data and select
the best performing model.

You have 3 sub-agents:
mt_analyzer_agent, mt_strategist_agent, mt_implementor_agent

You will receive:
X_train and X_test features, y_train and y_test labels, target_col,
feature_names, scaling_summary.

YOUR EXACT WORKFLOW:

PHASE 1 — ANALYZE
Call mt_analyzer_agent with:
X_train and X_test shapes
y_train and y_test distributions
feature names and count
minority class percentage from y_train
Wait for complete 6-section report before Phase 2.

PHASE 2 — STRATEGIZE
Call mt_strategist_agent with:
Full analysis report from Phase 1
task_type and minority_class_percentage
feature_names
Wait for complete plan covering SMOTE decision,
3 model configs, comparison strategy, saving plan.

PHASE 3 — IMPLEMENT
Call mt_implementor_agent with:
Full training plan from Phase 2
Reminder: SMOTE before training if needed
Reminder: evaluate on X_test only
Reminder: update sandbox globals after training
Wait for confirmation that all files are saved and sandbox has
best_model, best_model_name, best_predictions, all_results, smote_applied.

PHASE 4 — REPORT
Return structured JSON:

{
  "status": "success",
  "task_type": "CLASSIFICATION",
  "smote_applied": true,
  "best_model": "XGBoost",
  "primary_metric": "f1",
  "model_comparison": {
    "Random Forest": {"accuracy": 0.91, "f1": 0.90},
    "Logistic Regression": {"accuracy": 0.87, "f1": 0.86},
    "XGBoost": {"accuracy": 0.94, "f1": 0.93}
  },
  "winner_metrics": {
    "accuracy": 0.94,
    "f1": 0.93,
    "precision": 0.94,
    "recall": 0.93
  },
  "output_files": {
    "final_model":        "outputs/final_model.pkl",
    "model_rf":           "outputs/model_rf.pkl",
    "model_lr":           "outputs/model_lr.pkl",
    "model_xgb":          "outputs/model_xgb.pkl",
    "training_results":   "outputs/training_results.json"
  },
  "sandbox_ready": {
    "best_model":       true,
    "best_model_name":  true,
    "best_predictions": true
  },
  "next_phase": "Model Evaluation"
}

STRICT RULES:
Never skip a phase
Never call Phase 3 before Phase 2 is complete
Always pass complete output between phases
SMOTE must run before training if minority < 20%
Evaluation always on X_test — never training data
All 5 output files must exist before reporting success
sandbox_ready must show all 3 keys correctly
If implementor falls back to Skip — report technique as none
Minimum 2 models must succeed for pipeline to pass
"""


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Agent definition
# ─────────────────────────────────────────────────────────────────────────────

mt_orchestrator_agent = Agent(
    name="mt_orchestrator_agent",
    model="gemini-2.0-flash",
    description=(
        "Manages the model-training 3-agent pipeline. Runs analysis, strategy, "
        "and implementation in strict order, then returns structured comparison "
        "results and next phase metadata."
    ),
    instruction=_ORCHESTRATOR_PROMPT,
    tools=[],
    sub_agents=[
        mt_analyzer_agent,
        mt_strategist_agent,
        mt_implementor_agent,
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_serializable_list(values: Any) -> list[Any]:
    """
    Convert labels or shape-like values into a stable list.
    """
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    if hasattr(values, "tolist"):
        try:
            return list(values.tolist())
        except Exception:
            pass
    return [values]


def _shape_from_input(value: Any) -> list[int]:
    """
    Normalize shape-like metadata to [rows, cols].
    """
    if hasattr(value, "shape"):
        shape = getattr(value, "shape")
        if isinstance(shape, tuple) and len(shape) >= 2:
            return [int(shape[0]), int(shape[1])]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            return [0, 0]
    return [0, 0]


def _distribution(values: list[Any]) -> dict[str, int]:
    """
    Build ordered class counts preserving first-seen order.
    """
    if not values:
        return {}
    ordered: dict[str, int] = {}
    for raw in values:
        key = str(raw)
        ordered[key] = ordered.get(key, 0) + 1
    return ordered


def _percentages(dist: dict[str, int], total: int) -> dict[str, float]:
    """
    Convert counts to rounded percentage values.
    """
    if total <= 0:
        return {k: 0.0 for k in dist}
    return {k: round((v / total) * 100, 2) for k, v in dist.items()}


def _infer_task_type(
    y_train_values: list[Any],
    target_dtype: str,
) -> str:
    """
    Infer regression/classification when task_type is not provided.
    """
    if target_dtype:
        dt = str(target_dtype).lower()
        if dt in {"object", "string", "category"}:
            return "CLASSIFICATION"
        if dt in {"float", "float64", "float32", "int64", "int32"}:
            unique_count = len(set(str(v) for v in y_train_values))
            return "REGRESSION" if unique_count > 20 else "CLASSIFICATION"

    unique_count = len(set(str(v) for v in y_train_values))
    return "CLASSIFICATION" if unique_count <= 20 else "REGRESSION"


def _read_json_file(path: str) -> dict[str, Any]:
    """
    Read a JSON artifact for pipeline summaries.
    """
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as file:
            payload = json.load(file)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _sandbox_snapshot() -> dict[str, Any]:
    """
    Snapshot sandbox globals required by model training orchestrator.
    """
    code = """
import json
snapshot = {
    "best_model_exists": "best_model" in globals(),
    "best_model_name_exists": "best_model_name" in globals(),
    "best_predictions_exists": "best_predictions" in globals(),
    "all_results_exists": "all_results" in globals(),
    "smote_applied_exists": "smote_applied" in globals(),
}
print(json.dumps(snapshot))
"""
    out = execute_python(code).strip()
    if out.startswith("ERROR:"):
        return {
            "best_model_exists": False,
            "best_model_name_exists": False,
            "best_predictions_exists": False,
            "all_results_exists": False,
            "smote_applied_exists": False,
        }
    try:
        payload = json.loads(out)
    except Exception:
        payload = {}
    return {
        "best_model_exists": bool(payload.get("best_model_exists", False)),
        "best_model_name_exists": bool(payload.get("best_model_name_exists", False)),
        "best_predictions_exists": bool(payload.get("best_predictions_exists", False)),
        "all_results_exists": bool(payload.get("all_results_exists", False)),
        "smote_applied_exists": bool(payload.get("smote_applied_exists", False)),
    }


def _normalize_model_metrics(raw_metrics: Any) -> dict[str, float]:
    """
    Keep only comparable metrics as floats.
    """
    metrics: dict[str, float] = {}
    if not isinstance(raw_metrics, dict):
        return metrics

    for key in ("accuracy", "f1", "f1_weighted", "precision", "recall"):
        if key in raw_metrics:
            try:
                metrics[key] = float(raw_metrics[key])
            except (TypeError, ValueError):
                pass
    if not metrics and "weighted_f1" in raw_metrics:
        try:
            metrics["f1"] = float(raw_metrics["weighted_f1"])
        except (TypeError, ValueError):
            pass
    return metrics


def _best_metric_summary(
    all_results: dict[str, Any],
    imbalanced: bool,
) -> tuple[str, dict[str, float]]:
    """
    Select winner name and metric dict using requested tie rules.
    """
    primary = "f1_weighted" if imbalanced else "accuracy"
    candidates: list[tuple[str, dict[str, float], bool]] = []

    for model_name, result in all_results.items():
        normalized = _normalize_model_metrics(result.get("metrics", result))
        if not normalized:
            continue
        model_success = bool(result.get("status", "").lower() == "success")
        candidates.append((model_name, normalized, model_success))

    if not candidates:
        return "", {}

    def _score(item: tuple[str, dict[str, float], bool]) -> tuple[float, float]:
        _, m, succeeded = item
        primary_score = float(m.get(primary, m.get("f1", 0.0)))
        secondary_score = float(m.get("f1", m.get("f1_weighted", 0.0)))
        return (primary_score if succeeded else -1.0, secondary_score)

    ranked = sorted(candidates, key=_score, reverse=True)
    raw_winner_name = ranked[0][0]
    winner_name = {
        "rf": "Random Forest",
        "rf_model": "Random Forest",
        "random_forest": "Random Forest",
        "lr": "Logistic Regression",
        "lr_model": "Logistic Regression",
        "logistic_regression": "Logistic Regression",
        "xgb": "XGBoost",
        "xgb_model": "XGBoost",
        "xgboost": "XGBoost",
    }.get(str(raw_winner_name).lower(), str(raw_winner_name))
    winner_scores = ranked[0][1]
    normalized_output = {
        "accuracy": winner_scores.get("accuracy", 0.0),
        "f1": winner_scores.get("f1_weighted", winner_scores.get("f1", 0.0)),
        "precision": winner_scores.get("precision", 0.0),
        "recall": winner_scores.get("recall", 0.0),
    }
    return winner_name, normalized_output


# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline runner — Model Training orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_mt_orchestrator(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    target_col: str,
    feature_names: list[str],
    scaling_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the 3-phase model-training pipeline and return a strict structured report.

    Args:
        X_train:           Scaled training features.
        X_test:            Scaled test features.
        y_train:           Training labels.
        y_test:            Test labels.
        target_col:        ML target column name.
        feature_names:     Feature column names used by train data.
        scaling_summary:   Optional scaling metadata; can include task hints.

    Returns:
        Structured JSON result for downstream model evaluation stage.
    """
    start = time.monotonic()
    errors: list[str] = []
    log_lines: list[str] = []

    def _log(msg: str) -> None:
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    # Build basic data stats for analyzer
    y_train_list = _to_serializable_list(y_train)
    y_test_list = _to_serializable_list(y_test)
    x_train_shape = _shape_from_input(X_train)
    x_test_shape = _shape_from_input(X_test)
    train_distribution = _distribution(y_train_list)
    test_distribution = _distribution(y_test_list)
    train_total = len(y_train_list)
    test_total = len(y_test_list)

    total_features = len(feature_names)
    class_counts = dict(train_distribution)
    class_percentages = _percentages(class_counts, train_total)

    if class_percentages:
        minority_class_percentage = min(class_percentages.values())
    else:
        minority_class_percentage = 100.0

    target_dtype = ""
    if isinstance(scaling_summary, dict):
        target_dtype = str(scaling_summary.get("target_dtype", "")) if scaling_summary.get("target_dtype") is not None else ""

    normalized_task_type = _infer_task_type(y_train_list, target_dtype)
    normalized_task_type = normalized_task_type.upper()
    if feature_names is None:
        feature_names = []
    if total_features <= 0 and hasattr(X_train, "columns"):
        total_features = len(getattr(X_train, "columns"))

    _log("Phase 1 start — Model Training Analyzer.")
    analyzer_report = ""
    try:
        analyzer_report = await run_mt_analyzer(
            X_train_shape=x_train_shape,
            X_test_shape=x_test_shape,
            y_train_distribution=train_distribution,
            y_test_distribution=test_distribution,
            feature_count=total_features,
            feature_names=feature_names,
            target_col=target_col,
            target_dtype=target_dtype,
            class_counts=class_counts,
            class_percentages=class_percentages,
            minority_class_percentage=minority_class_percentage,
            task_type=normalized_task_type,
        )
        if not analyzer_report:
            raise RuntimeError("Empty analyzer report.")
        _log("Phase 1 complete.")
    except Exception as exc:
        errors.append(f"Model Training Analyzer failed: {exc}")
        return {
            "status": "error",
            "task_type": normalized_task_type,
            "smote_applied": False,
            "best_model": "unknown",
            "primary_metric": "f1" if normalized_task_type == "CLASSIFICATION" else "accuracy",
            "model_comparison": {},
            "winner_metrics": {},
            "output_files": {
                "final_model": "outputs/final_model.pkl",
                "model_rf": "outputs/model_rf.pkl",
                "model_lr": "outputs/model_lr.pkl",
                "model_xgb": "outputs/model_xgb.pkl",
                "training_results": "outputs/training_results.json",
            },
            "sandbox_ready": {
                "best_model": False,
                "best_model_name": False,
                "best_predictions": False,
            },
            "next_phase": "Model Evaluation",
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }

    _log("Phase 2 start — Model Training Strategist.")
    strategist_output: dict[str, Any] | None = None
    try:
        strategist_output = await run_mt_strategist(
            analyzer_report=analyzer_report,
            target_col=target_col,
            feature_names=feature_names,
            task_type=normalized_task_type,
            minority_class_percentage=minority_class_percentage,
        )
        if not isinstance(strategist_output, dict) or not strategist_output.get("training_plan"):
            raise RuntimeError("Strategist output missing training_plan.")
        _log("Phase 2 complete.")
    except Exception as exc:
        errors.append(f"Model Training Strategist failed: {exc}")
        return {
            "status": "error",
            "task_type": normalized_task_type,
            "smote_applied": False,
            "best_model": "unknown",
            "primary_metric": "f1" if normalized_task_type == "CLASSIFICATION" else "accuracy",
            "model_comparison": {},
            "winner_metrics": {},
            "output_files": {
                "final_model": "outputs/final_model.pkl",
                "model_rf": "outputs/model_rf.pkl",
                "model_lr": "outputs/model_lr.pkl",
                "model_xgb": "outputs/model_xgb.pkl",
                "training_results": "outputs/training_results.json",
            },
            "sandbox_ready": {
                "best_model": False,
                "best_model_name": False,
                "best_predictions": False,
            },
            "next_phase": "Model Evaluation",
            "analysis": analyzer_report,
            "errors": errors,
            "pipeline_log": "\n".join(log_lines),
            "elapsed_seconds": round(time.monotonic() - start, 2),
        }

    _log("Phase 3 start — Model Training Implementor.")
    implementor_output: dict[str, Any] | None = None
    try:
        implementor_output = await run_mt_implementor(
            {
                **strategist_output,
                "task_type": normalized_task_type,
                "minority_class_percentage": minority_class_percentage,
                "class_counts": class_counts,
            }
        )
        if not isinstance(implementor_output, dict):
            raise RuntimeError("Invalid implementor output.")
        _log("Phase 3 complete.")
    except Exception as exc:
        errors.append(f"Model Training Implementor failed: {exc}")
        implementor_output = {}

    # Read execution outputs
    training_results = _read_json_file("outputs/training_results.json")
    all_results = training_results.get("all_results", {})
    if not isinstance(all_results, dict):
        all_results = {}

    success_count = 0
    for result in all_results.values():
        if isinstance(result, dict) and str(result.get("status", "")).lower() == "success":
            success_count += 1

    train_shape = training_results.get("train_shape")
    test_shape = training_results.get("test_shape")
    smote_applied = bool(training_results.get("smote_applied", implementor_output.get("smote_applied", False)))

    model_comparison: dict[str, dict[str, float]] = {}
    candidate_keys = {
        "Random Forest": ("rf", "rf_model", "random_forest", "Random Forest"),
        "Logistic Regression": ("lr", "lr_model", "logistic_regression", "Logistic Regression"),
        "XGBoost": ("xgb", "xgb_model", "xgboost", "XGBoost"),
    }

    for display_name, aliases in candidate_keys.items():
        raw = {}
        for alias in aliases:
            if alias in all_results:
                raw = all_results.get(alias, {})
                if isinstance(raw, dict):
                    break
            key_norm = str(alias).lower()
            if key_norm in all_results:
                raw = all_results.get(key_norm, {})
                if isinstance(raw, dict):
                    break
        if not isinstance(raw, dict):
            continue
        metrics = _normalize_model_metrics(raw.get("metrics", raw))
        model_metrics = {
            "accuracy": round(metrics.get("accuracy", 0.0), 6),
            "f1": round(metrics.get("f1", metrics.get("f1_weighted", 0.0)), 6),
            "precision": round(metrics.get("precision", 0.0), 6),
            "recall": round(metrics.get("recall", 0.0), 6),
        }
        model_comparison[display_name] = model_metrics

    imbalanced = minority_class_percentage < 20.0 if train_total > 0 else False
    primary_metric = "f1" if imbalanced else "accuracy"

    # Prefer strategy fallback to ensure at least deterministic summary
    if success_count < 2 or not model_comparison:
        best_model = training_results.get("best_model_name", "")
        winner_metrics = training_results.get("winner_metrics", {})
        if not winner_metrics and isinstance(model_comparison, dict):
            for model_name, metrics in model_comparison.items():
                winner_metrics = metrics
                best_model = model_name
                break
    else:
        best_model, winner_metrics = _best_metric_summary(training_results.get("all_results", all_results), imbalanced)
        if not winner_metrics:
            best_model = training_results.get("best_model_name", "")
            winner_metrics = training_results.get("winner_metrics", {})

    winner_metrics_out = {
        "accuracy": round(float(winner_metrics.get("accuracy", 0.0)), 6),
        "f1": round(
            float(winner_metrics.get("f1", winner_metrics.get("f1_weighted", 0.0))),
            6,
        ),
        "precision": round(float(winner_metrics.get("precision", 0.0)), 6),
        "recall": round(float(winner_metrics.get("recall", 0.0)), 6),
    }

    sandbox = _sandbox_snapshot()
    output_files = {
        "final_model": "outputs/final_model.pkl",
        "model_rf": "outputs/model_rf.pkl",
        "model_lr": "outputs/model_lr.pkl",
        "model_xgb": "outputs/model_xgb.pkl",
        "training_results": "outputs/training_results.json",
    }

    file_exists = all(os.path.isfile(path) for path in output_files.values())
    status = "success" if success_count >= 2 and file_exists else "error"
    if status != "success":
        if success_count < 2:
            errors.append("Less than 2 models succeeded in implementation.")
        if not file_exists:
            errors.append("One or more required output files are missing.")
    if not (
        sandbox.get("best_model_exists")
        and sandbox.get("best_model_name_exists")
        and sandbox.get("best_predictions_exists")
        and sandbox.get("all_results_exists")
        and sandbox.get("smote_applied_exists")
    ):
        status = "error"
        errors.append("Required sandbox globals missing after model implementation.")

    return {
        "status": status,
        "task_type": normalized_task_type,
        "smote_applied": bool(smote_applied),
        "best_model": best_model or ("none" if status != "success" else ""),
        "primary_metric": primary_metric,
        "model_comparison": model_comparison,
        "winner_metrics": winner_metrics_out,
        "output_files": output_files,
        "sandbox_ready": {
            "best_model": bool(sandbox.get("best_model_exists", False)),
            "best_model_name": bool(sandbox.get("best_model_name_exists", False)),
            "best_predictions": bool(sandbox.get("best_predictions_exists", False)),
        },
        "train_shape": train_shape if isinstance(train_shape, list) else x_train_shape,
        "test_shape": test_shape if isinstance(test_shape, list) else x_test_shape,
        "next_phase": "Model Evaluation",
        "analysis": analyzer_report,
        "strategy": strategist_output,
        "implementation": implementor_output,
        "training_results": training_results,
        "errors": errors,
        "pipeline_log": "\n".join(log_lines + [f"Model successes: {success_count}"]),
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }


# Backward-compatible alias
run_model_training_orchestrator = run_mt_orchestrator
