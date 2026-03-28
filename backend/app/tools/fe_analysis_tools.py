"""
app/tools/fe_analysis_tools.py

Single comprehensive feature-engineering tool consumed by the Feature
Engineering Analyzer.
Computes profile statistics for cleaned rows and returns the exact dict
structure the FE Analyzer Agent expects.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
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
    bool_set = {True, False, "true", "false", "True", "False", "1", "0", 1, 0}
    if all(v in bool_set for v in non_null):
        return "boolean"
    if all(_coerce_numeric(v) is not None for v in non_null):
        return "numeric"
    dt_pat = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$")
    if all(isinstance(v, str) and dt_pat.match(v) for v in non_null):
        return "datetime"
    return "string"


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


# ─────────────────────────────────────────────────────────────────────────────
# Main tool — FE analytics profile
# ─────────────────────────────────────────────────────────────────────────────

def analyze_for_feature_engineering(
    records: list[dict],
    target_col: str,
) -> dict[str, Any]:
    """
    Compute a feature-engineering profile for cleaned data.

    Args:
        records:    Cleaned dataset rows as list of dicts (df_clean).
        target_col: Name of the ML target column.

    Returns:
        dict with keys:
          shape                — [rows, columns]
          dtypes               — {col: "numeric|string|boolean|datetime|empty"}
          numeric_stats        — {col: {min, max, mean, std, skewness, kurtosis}}
          categorical_stats    — {col: {unique_count, top_10_frequencies}}
          correlation_matrix   — {col: {col: pearson correlation}}
          variance_stats       — {col: variance}
          numeric_cols         — [col, ...]
          categorical_cols     — [col, ...]
          target_col           — target column string
          sample_rows          — first 3 rows as dicts
    """
    # Build df_clean internally as requested by contract.
    df_clean: list[dict] = [row.copy() for row in records] if records else []

    if not df_clean:
        return {
            "shape": [0, 0],
            "dtypes": {},
            "numeric_stats": {},
            "categorical_stats": {},
            "correlation_matrix": {},
            "variance_stats": {},
            "numeric_cols": [],
            "categorical_cols": [],
            "target_col": target_col,
            "sample_rows": [],
        }

    n = len(df_clean)
    columns: list[str] = list(df_clean[0].keys())

    col_values: dict[str, list] = {
        col: [row.get(col) for row in df_clean] for col in columns
    }
    dtypes: dict[str, str] = {col: _detect_dtype(vs) for col, vs in col_values.items()}

    numeric_cols = [col for col, dtype in dtypes.items() if dtype == "numeric"]
    categorical_cols = [col for col in columns if col not in numeric_cols]

    # ── numeric stats + variance + skew/kurt ────────────────────────────────
    numeric_stats: dict[str, dict] = {}
    variance_stats: dict[str, float] = {}

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
                "kurtosis": None,
            }
            variance_stats[col] = 0.0
            continue

        cnt = len(nums)
        mean = sum(nums) / cnt
        variance = sum((x - mean) ** 2 for x in nums) / cnt
        std = math.sqrt(variance) if variance > 0 else 0.0

        skewness = 0.0
        if std > 0 and cnt >= 3:
            skewness = (sum((x - mean) ** 3 for x in nums) / cnt) / (std ** 3)

        kurtosis = 0.0
        if std > 0 and cnt >= 4:
            kurtosis = (sum((x - mean) ** 4 for x in nums) / cnt) / (std ** 4) - 3.0

        numeric_stats[col] = {
            "min": round(min(nums), 4),
            "max": round(max(nums), 4),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
        }
        variance_stats[col] = round(variance, 4)

    # ── categorical stats ───────────────────────────────────────────────────
    categorical_stats: dict[str, dict] = {}
    for col in categorical_cols:
        non_null_vals = [str(v) for v in col_values[col] if v not in (None, "")]
        counter = Counter(non_null_vals)
        categorical_stats[col] = {
            "unique_count": len(counter),
            "top_10_frequencies": [
                {"value": v, "count": c} for v, c in counter.most_common(10)
            ],
        }

    # ── correlation matrix for all numeric columns ─────────────────────────
    correlation_matrix: dict[str, dict[str, float]] = {}
    for c1 in numeric_cols:
        correlation_matrix[c1] = {}
        for c2 in numeric_cols:
            if c1 == c2:
                correlation_matrix[c1][c2] = 1.0
                continue

            v1 = [_coerce_numeric(x) for x in col_values[c1]]
            v2 = [_coerce_numeric(x) for x in col_values[c2]]
            paired = [
                (a, b) for a, b in zip(v1, v2)
                if a is not None and b is not None
            ]
            if len(paired) < 2:
                correlation_matrix[c1][c2] = 0.0
                continue
            xs, ys = zip(*paired)
            correlation_matrix[c1][c2] = round(_pearson(list(xs), list(ys)), 4)

    return {
        "shape": [n, len(columns)],
        "dtypes": dtypes,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "correlation_matrix": correlation_matrix,
        "variance_stats": variance_stats,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": target_col,
        "sample_rows": df_clean[:3],
    }
