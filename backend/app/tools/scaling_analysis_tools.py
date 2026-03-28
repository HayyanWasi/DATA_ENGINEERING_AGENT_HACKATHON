"""
app/tools/scaling_analysis_tools.py

Single comprehensive feature-scaling tool consumed by the Scaling Analyzer.

Input:  Engineered data rows + optional FE metadata.
Output: A deterministic scaling profile consumed by the agent prompt.
"""

from __future__ import annotations

import math
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _detect_dtype(values: list[Any]) -> str:
    non_null = [v for v in values if v is not None and v != ""]
    if not non_null:
        return "empty"

    if all(_coerce_numeric(v) is not None for v in non_null):
        return "numeric"
    return "categorical"


def _is_numeric_scalar(stats: dict[str, Any], value: Any) -> bool:
    """
    Return True when value should be treated as a numeric range candidate.
    """
    if value is None:
        return False
    if not isinstance(value, (int, float)):
        return False
    if isinstance(value, bool):
        return False
    if not math.isfinite(float(value)):
        return False
    return True


def _percent_value_counts(values: list[float], mean: float, std: float) -> float:
    if not std or not values:
        return 0.0
    out = sum(1 for x in values if abs(x - mean) > 3 * std)
    return round(out / len(values) * 100, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main tool
# ─────────────────────────────────────────────────────────────────────────────

def analyze_for_feature_scaling(
    records: list[dict[str, Any]],
    target_col: str,
    encoded_cols: list[str] | None = None,
    transformed_cols: list[str] | None = None,
    value_ranges: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Compute a feature-scaling profile for engineered data.

    Args:
        records:         Engineered rows as plain dicts.
        target_col:      ML target column name.
        encoded_cols:    FE-encoded columns (one-hot + label encoded).
        transformed_cols: FE-transformed columns (log/sqrt).
        value_ranges:    Optional precomputed value ranges.

    Returns:
        dict with keys:
          shape            — [rows, columns]
          dtypes           — {col: "numeric|categorical|empty"}
          numeric_stats    — {col: {min, max, mean, std, skewness}}
          numeric_cols     — list of numeric column names
          categorical_cols — list of categorical column names
          target_col       — passed target name
          encoded_cols     — resolved one-hot/label encoded column names
          transformed_cols — resolved transformed column names
          value_ranges     — {col: {min, max, range}}
          outlier_summary  — {col: {outlier_count_3std, outlier_pct_3std}}
    """
    if not records:
        return {
            "shape": [0, 0],
            "dtypes": {},
            "numeric_stats": {},
            "numeric_cols": [],
            "categorical_cols": [],
            "target_col": target_col,
            "encoded_cols": encoded_cols or [],
            "transformed_cols": transformed_cols or [],
            "value_ranges": value_ranges or {},
            "outlier_summary": {},
        }

    n = len(records)
    columns = list(records[0].keys())

    col_values: dict[str, list[Any]] = {
        col: [row.get(col) for row in records] for col in columns
    }
    dtypes: dict[str, str] = {col: _detect_dtype(vs) for col, vs in col_values.items()}

    numeric_cols = [c for c, t in dtypes.items() if t == "numeric" and c != target_col]
    categorical_cols = [c for c, t in dtypes.items() if t != "numeric" and c != target_col]

    numeric_stats: dict[str, dict[str, Any]] = {}
    outlier_summary: dict[str, dict[str, Any]] = {}
    derived_ranges: dict[str, dict[str, Any]] = {}

    for col in numeric_cols:
        nums = sorted(
            v for v in (_coerce_numeric(x) for x in col_values[col]) if v is not None
        )
        if not nums:
            numeric_stats[col] = {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "skewness": None,
            }
            derived_ranges[col] = {"min": None, "max": None, "range": None}
            outlier_summary[col] = {
                "outlier_count_3std": 0,
                "outlier_pct_3std": 0.0,
            }
            continue

        count = len(nums)
        mean = sum(nums) / count
        min_val = float(min(nums))
        max_val = float(max(nums))
        variance = sum((x - mean) ** 2 for x in nums) / count
        std = math.sqrt(variance) if variance > 0 else 0.0
        rng = max_val - min_val

        skewness = 0.0
        if std > 0 and count >= 3:
            skewness = (sum((x - mean) ** 3 for x in nums) / count) / (std ** 3)

        outliers = _percent_value_counts(nums, mean, std)
        outlier_count = round(outliers / 100 * count)

        numeric_stats[col] = {
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "skewness": round(skewness, 4),
        }
        derived_ranges[col] = {
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "range": round(rng, 4),
        }
        outlier_summary[col] = {
            "outlier_count_3std": int(outlier_count),
            "outlier_pct_3std": outliers,
        }

    # If caller supplies explicit value ranges (for exact consistency), merge them
    if value_ranges:
        for col, stats in value_ranges.items():
            if not isinstance(stats, dict):
                continue
            min_v = stats.get("min")
            max_v = stats.get("max")
            if _is_numeric_scalar(stats, min_v) and _is_numeric_scalar(stats, max_v):
                derived_ranges[col] = {
                    "min": float(min_v),
                    "max": float(max_v),
                    "range": float(max_v) - float(min_v),
                }

    resolved_encoded = [c for c in (encoded_cols or []) if c in columns]
    resolved_transformed = [c for c in (transformed_cols or []) if c in columns]

    return {
        "shape": [n, len(columns)],
        "dtypes": dtypes,
        "numeric_stats": numeric_stats,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": target_col,
        "encoded_cols": resolved_encoded,
        "transformed_cols": resolved_transformed,
        "value_ranges": derived_ranges,
        "outlier_summary": outlier_summary,
    }
