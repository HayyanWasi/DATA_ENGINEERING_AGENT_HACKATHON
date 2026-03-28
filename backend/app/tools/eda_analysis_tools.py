"""
app/tools/eda_analysis_tools.py

Single comprehensive EDA tool consumed by the EDA pipeline.
Operates on CLEANED data (df_clean records) and computes all statistics
needed by the Analyzer Agent and Strategist Agent.
"""

from __future__ import annotations

import math
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


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * pct / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


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
    return round(num / (dx * dy), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main EDA tool — called by run_eda_analyzer before invoking the agent
# ─────────────────────────────────────────────────────────────────────────────

def analyze_eda(records: list[dict], target_col: str = "") -> dict[str, Any]:
    """
    Compute a complete EDA profile for cleaned data.

    The returned dict is passed directly to the EDA Analyzer Agent as its
    input message — the agent reports from these pre-computed stats without
    needing to call any tools itself.

    Args:
        records:    Cleaned dataset rows as a list of plain dicts (df_clean).
        target_col: Name of the ML target column.

    Returns:
        dict with keys:
          shape               — {"rows": int, "columns": int}
          dtypes              — {col: "numeric" | "categorical"}
          null_counts         — {col: int} (all columns, including zeros)
          duplicate_rows      — int
          numeric_cols        — list of numeric column names
          categorical_cols    — list of categorical column names
          target_col          — target column name
          target_type         — "categorical" | "numeric" | "unknown"
          target_distribution — {value: count} for categorical; stats dict for numeric
          class_balance       — {value: pct} (categorical targets only)
          is_imbalanced       — bool (True if dominant class > 70%)
          numeric_stats       — {col: {count, min, max, mean, median, std,
                                       skewness, kurtosis, p25, p75, iqr,
                                       lower_fence, upper_fence,
                                       outlier_count, outlier_pct}}
          categorical_stats   — {col: {unique_count, top_values:[{value,count}]}}
          correlation_matrix  — {col: {col: pearson_r}} for all numeric col pairs
          correlation_pairs   — top 10 pairs sorted by |r| (for strategist)
          high_corr_with_target — [{col, correlation}] sorted by |r| with target
          skewed_cols         — numeric cols with |skewness| > 1.0 (excl. target)
          outlier_cols        — numeric cols with outlier_pct > 5% (excl. target)
          sample_rows         — first 3 rows
    """
    if not records:
        return {
            "shape": {"rows": 0, "columns": 0},
            "dtypes": {}, "null_counts": {}, "duplicate_rows": 0,
            "numeric_cols": [], "categorical_cols": [],
            "target_col": target_col, "target_type": "unknown",
            "target_distribution": {}, "class_balance": {}, "is_imbalanced": False,
            "numeric_stats": {}, "categorical_stats": {},
            "correlation_matrix": {}, "correlation_pairs": [],
            "high_corr_with_target": [],
            "skewed_cols": [], "outlier_cols": [],
            "sample_rows": [],
        }

    n = len(records)
    columns: list[str] = list(records[0].keys())
    col_values: dict[str, list] = {
        col: [row.get(col) for row in records] for col in columns
    }

    # ── Detect column types ───────────────────────────────────────────────────
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    dtypes: dict[str, str] = {}
    for col, vals in col_values.items():
        non_null = [v for v in vals if v is not None and v != ""]
        if non_null and all(_coerce_numeric(v) is not None for v in non_null):
            numeric_cols.append(col)
            dtypes[col] = "numeric"
        else:
            categorical_cols.append(col)
            dtypes[col] = "categorical"

    # ── Null counts (all columns, including zeros) ────────────────────────────
    null_counts: dict[str, int] = {
        col: sum(1 for v in vs if v is None or v == "")
        for col, vs in col_values.items()
    }

    # ── Duplicate rows ────────────────────────────────────────────────────────
    fingerprints = [str(sorted(row.items())) for row in records]
    dup_count = sum(c - 1 for c in Counter(fingerprints).values() if c > 1)

    # ── Numeric stats: mean, std, skewness, kurtosis, IQR, outliers ──────────
    numeric_stats: dict[str, dict] = {}
    numeric_arrays: dict[str, list[float]] = {}

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
        std = math.sqrt(variance) if variance > 0 else 0.0

        # Pearson's moment skewness (3rd standardised moment)
        skewness = 0.0
        if std > 0 and cnt >= 3:
            skewness = (sum((x - mean) ** 3 for x in nums) / cnt) / (std ** 3)

        # Excess kurtosis (4th standardised moment − 3)
        kurtosis = 0.0
        if std > 0 and cnt >= 4:
            kurtosis = (sum((x - mean) ** 4 for x in nums) / cnt) / (std ** 4) - 3.0

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
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "p25": round(p25, 4),
            "p75": round(p75, 4),
            "iqr": round(iqr, 4),
            "lower_fence": round(lower_fence, 4),
            "upper_fence": round(upper_fence, 4),
            "outlier_count": outlier_count,
            "outlier_pct": round(outlier_count / cnt * 100, 2),
        }
        numeric_arrays[col] = nums

    # ── Categorical stats ─────────────────────────────────────────────────────
    categorical_stats: dict[str, dict] = {}
    for col in categorical_cols:
        non_null_vals = [str(v) for v in col_values[col] if v not in (None, "")]
        counter = Counter(non_null_vals)
        categorical_stats[col] = {
            "unique_count": len(counter),
            "top_values": [
                {"value": v, "count": c} for v, c in counter.most_common(10)
            ],
        }

    # ── Target variable analysis ──────────────────────────────────────────────
    target_distribution: dict[str, Any] = {}
    target_type = "unknown"
    class_balance: dict[str, float] = {}
    is_imbalanced = False

    if target_col:
        if target_col in numeric_stats:
            target_distribution = numeric_stats[target_col]
            target_type = "numeric"
        elif target_col in categorical_stats:
            counter = Counter(
                str(v) for v in col_values.get(target_col, [])
                if v not in (None, "")
            )
            target_distribution = dict(counter.most_common())
            total = sum(counter.values())
            class_balance = {
                k: round(v / total * 100, 2) for k, v in counter.items()
            }
            max_pct = max(class_balance.values()) if class_balance else 0
            is_imbalanced = max_pct > 70.0
            target_type = "categorical"

    # ── Full correlation matrix (Pearson, all numeric col pairs) ─────────────
    num_col_list = list(numeric_arrays.keys())
    correlation_matrix: dict[str, dict[str, float]] = {}
    correlation_pairs: list[dict] = []
    high_corr_with_target: list[dict] = []

    if len(num_col_list) >= 2:
        # Build full symmetric matrix
        for c1 in num_col_list:
            correlation_matrix[c1] = {}
            for c2 in num_col_list:
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
                correlation_matrix[c1][c2] = _pearson(list(xs), list(ys))

        # Top pairs list for the strategist (chart heatmap decisions)
        pairs: list[dict] = []
        for i, c1 in enumerate(num_col_list):
            for c2 in num_col_list[i + 1:]:
                corr = correlation_matrix[c1].get(c2, 0.0)
                pairs.append({
                    "col1": c1,
                    "col2": c2,
                    "correlation": corr,
                    "abs_corr": round(abs(corr), 4),
                })
        pairs.sort(key=lambda p: p["abs_corr"], reverse=True)
        correlation_pairs = [
            {k: v for k, v in p.items() if k != "abs_corr"}
            for p in pairs[:10]
        ]

        # Features most correlated with numeric target
        if target_col in num_col_list:
            target_corrs = [
                p for p in pairs
                if p["col1"] == target_col or p["col2"] == target_col
            ]
            high_corr_with_target = [
                {
                    "col": p["col2"] if p["col1"] == target_col else p["col1"],
                    "correlation": p["correlation"],
                }
                for p in target_corrs[:10]
            ]

    # ── Derived summaries (used by strategist for chart decisions) ────────────
    skewed_cols = [
        col for col, s in numeric_stats.items()
        if abs(s["skewness"]) > 1.0 and col != target_col
    ]
    outlier_cols = [
        col for col, s in numeric_stats.items()
        if s["outlier_pct"] > 5.0 and col != target_col
    ]

    return {
        "shape": {"rows": n, "columns": len(columns)},
        "dtypes": dtypes,
        "null_counts": null_counts,
        "duplicate_rows": dup_count,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": target_col,
        "target_type": target_type,
        "target_distribution": target_distribution,
        "class_balance": class_balance,
        "is_imbalanced": is_imbalanced,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "correlation_matrix": correlation_matrix,
        "correlation_pairs": correlation_pairs,
        "high_corr_with_target": high_corr_with_target,
        "skewed_cols": skewed_cols,
        "outlier_cols": outlier_cols,
        "sample_rows": records[:3],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dual-dataset stats — used by the EDA Orchestrator before calling the Analyzer
# ─────────────────────────────────────────────────────────────────────────────

def compute_eda_stats(
    raw_records: list[dict],
    clean_records: list[dict],
    target_col: str = "",
) -> dict[str, Any]:
    """
    Compute EDA statistics from BOTH the raw (pre-cleaning) and cleaned datasets.

    Returns a combined dict that the EDA Analyzer can use to compare distributions,
    null counts, and outlier counts before and after cleaning — enabling the
    6-section report to state facts like "nulls dropped from 23% to 0%".

    Also provides the column lists that init_eda_sandbox and the strategist need.

    Args:
        raw_records:   Original rows before any cleaning (df rows as list of dicts).
        clean_records: Cleaned rows (df_clean rows as list of dicts).
        target_col:    Name of the ML target column.

    Returns:
        dict with keys:
          "clean"            — full analyze_eda() result for cleaned data
          "raw"              — lightweight raw stats (shape, nulls, basic numerics)
          "target_col"       — target column name
          "numeric_cols"     — from clean stats (authoritative post-cleaning list)
          "categorical_cols" — from clean stats
    """
    # Full stats on cleaned data — primary source for the analyzer
    clean_stats = analyze_eda(clean_records, target_col=target_col)

    # Lightweight raw stats — shape + null counts + basic numeric means/stds
    # enough for the analyzer to report "before cleaning" context
    raw_n = len(raw_records)
    raw_cols: list[str] = list(raw_records[0].keys()) if raw_records else []
    raw_col_values: dict[str, list] = {
        col: [row.get(col) for row in raw_records] for col in raw_cols
    }

    raw_null_counts: dict[str, int] = {
        col: sum(1 for v in vs if v is None or v == "")
        for col, vs in raw_col_values.items()
    }
    raw_null_pct: dict[str, float] = {
        col: round(raw_null_counts[col] / raw_n * 100, 2) if raw_n else 0.0
        for col in raw_cols
    }

    # Basic numeric stats on raw (mean/std only — enough for comparison)
    raw_numeric_basic: dict[str, dict] = {}
    for col in raw_cols:
        nums = [
            v for v in (_coerce_numeric(x) for x in raw_col_values[col])
            if v is not None
        ]
        if len(nums) >= 2:
            mean = sum(nums) / len(nums)
            std_val = math.sqrt(sum((x - mean) ** 2 for x in nums) / len(nums))
            raw_numeric_basic[col] = {
                "count": len(nums),
                "mean": round(mean, 4),
                "std": round(std_val, 4),
                "min": round(min(nums), 4),
                "max": round(max(nums), 4),
            }

    raw_dup_fingerprints = [str(sorted(row.items())) for row in raw_records]
    raw_dup_count = sum(
        c - 1 for c in Counter(raw_dup_fingerprints).values() if c > 1
    )

    raw_stats: dict[str, Any] = {
        "shape": {"rows": raw_n, "columns": len(raw_cols)},
        "null_counts": raw_null_counts,
        "null_percentage": raw_null_pct,
        "duplicate_rows": raw_dup_count,
        "numeric_basic": raw_numeric_basic,
    }

    return {
        "clean": clean_stats,
        "raw": raw_stats,
        "target_col": target_col,
        # Expose column lists at top level — consumed by orchestrator + sandbox init
        "numeric_cols": clean_stats["numeric_cols"],
        "categorical_cols": clean_stats["categorical_cols"],
    }
