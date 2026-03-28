"""
app/tools/analysis_tools.py

Single comprehensive tool consumed by the Analyzer Agent.
Computes ALL profiling statistics in one call and returns the exact dict
structure the Analyzer Agent's system prompt expects.
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


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * pct / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


# ─────────────────────────────────────────────────────────────────────────────
# Main tool — called once by the Analyzer Agent
# ─────────────────────────────────────────────────────────────────────────────

def analyze_dataset(records: list[dict]) -> dict[str, Any]:
    """
    Compute a complete data-quality profile for a raw dataset.

    Args:
        records: List of row dicts representing the raw dataset
                 (each dict maps column-name → value).

    Returns:
        A dict with these keys:
          shape           — {"rows": int, "columns": int}
          dtypes          — {col: "numeric|string|boolean|datetime|empty"}
          null_counts     — {col: int}
          null_percentage — {col: float}
          duplicate_rows  — int
          numeric_stats   — {col: {min,max,mean,median,std,p25,p75,iqr,
                                    lower_fence,upper_fence,outlier_count}}
          categorical_stats — {col: {unique_count, top_values:[{value,count}]}}
          categorical_cols  — [col, ...]
          numeric_cols      — [col, ...]
          sample_rows       — [first 3 rows as dicts]
    """
    if not records:
        return {
            "shape": {"rows": 0, "columns": 0},
            "dtypes": {},
            "null_counts": {},
            "null_percentage": {},
            "duplicate_rows": 0,
            "numeric_stats": {},
            "categorical_stats": {},
            "categorical_cols": [],
            "numeric_cols": [],
            "sample_rows": [],
        }

    n = len(records)
    columns: list[str] = list(records[0].keys())

    # ── dtypes ───────────────────────────────────────────────────────────────
    col_values: dict[str, list] = {
        col: [row.get(col) for row in records] for col in columns
    }
    dtypes: dict[str, str] = {col: _detect_dtype(vs) for col, vs in col_values.items()}

    # ── nulls ─────────────────────────────────────────────────────────────────
    null_counts: dict[str, int] = {
        col: sum(1 for v in vs if v is None or v == "")
        for col, vs in col_values.items()
    }
    null_pct: dict[str, float] = {
        col: round(null_counts[col] / n * 100, 2) for col in columns
    }

    # ── duplicates ────────────────────────────────────────────────────────────
    fingerprints = [str(sorted(row.items())) for row in records]
    dup_count = sum(c - 1 for c in Counter(fingerprints).values() if c > 1)

    # ── column groups ─────────────────────────────────────────────────────────
    numeric_cols = [c for c, t in dtypes.items() if t == "numeric"]
    categorical_cols = [c for c, t in dtypes.items() if t in ("string", "boolean")]

    # ── numeric stats + outlier detection ─────────────────────────────────────
    numeric_stats: dict[str, dict] = {}
    for col in numeric_cols:
        nums = sorted(
            v for v in (_coerce_numeric(x) for x in col_values[col]) if v is not None
        )
        if len(nums) < 2:
            continue
        cnt = len(nums)
        mean = sum(nums) / cnt
        median = (
            nums[cnt // 2] if cnt % 2
            else (nums[cnt // 2 - 1] + nums[cnt // 2]) / 2
        )
        variance = sum((x - mean) ** 2 for x in nums) / cnt
        std = math.sqrt(variance)
        p25 = _percentile(nums, 25)
        p75 = _percentile(nums, 75)
        iqr = p75 - p25
        lower_fence = p25 - 1.5 * iqr
        upper_fence = p75 + 1.5 * iqr
        outlier_count = sum(1 for v in nums if v < lower_fence or v > upper_fence)
        numeric_stats[col] = {
            "count": cnt,
            "min": round(min(nums), 4),
            "max": round(max(nums), 4),
            "mean": round(mean, 4),
            "median": round(median, 4),
            "std": round(std, 4),
            "p25": round(p25, 4),
            "p75": round(p75, 4),
            "iqr": round(iqr, 4),
            "lower_fence": round(lower_fence, 4),
            "upper_fence": round(upper_fence, 4),
            "outlier_count": outlier_count,
            "outlier_pct": round(outlier_count / cnt * 100, 2),
        }

    # ── categorical stats ─────────────────────────────────────────────────────
    categorical_stats: dict[str, dict] = {}
    for col in categorical_cols:
        non_null_vals = [str(v) for v in col_values[col] if v not in (None, "")]
        counter = Counter(non_null_vals)
        categorical_stats[col] = {
            "unique_count": len(counter),
            "top_values": [{"value": v, "count": c} for v, c in counter.most_common(10)],
        }

    return {
        "shape": {"rows": n, "columns": len(columns)},
        "dtypes": dtypes,
        "null_counts": null_counts,
        "null_percentage": null_pct,
        "duplicate_rows": dup_count,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "sample_rows": records[:3],
    }
