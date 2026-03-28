"""
app/tools/strategy_tools.py

Pure-function tools consumed by the Strategist Agent.
Each tool receives a slice of the AnalysisReport and returns a list of
strategy decision dicts so the agent can reason over them before writing
the final human-readable plan.
"""

from __future__ import annotations

from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _skewness_label(mean: float, median: float) -> str:
    """Return 'right-skewed', 'left-skewed', or 'symmetric'."""
    diff = mean - median
    rel = abs(diff) / (abs(median) + 1e-9)
    if rel < 0.05:
        return "symmetric"
    return "right-skewed" if diff > 0 else "left-skewed"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Null-handling strategy
# ─────────────────────────────────────────────────────────────────────────────

def decide_null_strategy(
    null_summary: dict[str, dict],
    dtypes: dict[str, str],
    descriptive_stats: dict[str, dict],
) -> list[dict[str, Any]]:
    """
    Decide the best strategy to handle missing values for every column that
    has at least one null.

    Decision rules:
    - null_pct > 60 % → DROP the column entirely
    - numeric column, symmetric distribution → FILL with mean
    - numeric column, skewed distribution → FILL with median
    - boolean / string column with low cardinality → FILL with mode
    - string column with high cardinality → FLAG / LEAVE as-is
    - datetime column → FILL with forward-fill (ffill)

    Args:
        null_summary:      Output of get_null_summary().
        dtypes:            Output of get_dtypes().
        descriptive_stats: Output of get_descriptive_stats().

    Returns:
        List of dicts, one per affected column:
        {column, null_count, null_pct, dtype, action, fill_value_type, reason}
    """
    decisions: list[dict[str, Any]] = []

    for col, info in null_summary.items():
        if info["null_count"] == 0:
            continue

        null_pct: float = info["null_pct"]
        dtype: str = dtypes.get(col, "string")
        stats: dict = descriptive_stats.get(col, {})

        if null_pct > 60:
            decisions.append({
                "column": col,
                "null_count": info["null_count"],
                "null_pct": null_pct,
                "dtype": dtype,
                "action": "drop_column",
                "fill_value_type": None,
                "reason": (
                    f"{null_pct}% of values are missing — dropping the column "
                    "is safer than imputing unreliable data."
                ),
            })

        elif dtype == "numeric":
            skew = _skewness_label(
                stats.get("mean", 0.0), stats.get("median", 0.0)
            )
            fill = "median" if skew != "symmetric" else "mean"
            decisions.append({
                "column": col,
                "null_count": info["null_count"],
                "null_pct": null_pct,
                "dtype": dtype,
                "action": "fill",
                "fill_value_type": fill,
                "reason": (
                    f"Numeric column with {skew} distribution — "
                    f"filling nulls with the {fill} avoids bias from outliers."
                ),
            })

        elif dtype == "datetime":
            decisions.append({
                "column": col,
                "null_count": info["null_count"],
                "null_pct": null_pct,
                "dtype": dtype,
                "action": "fill",
                "fill_value_type": "forward_fill",
                "reason": (
                    "Datetime column — forward-fill propagates the last known "
                    "timestamp, which is the most natural imputation for time series."
                ),
            })

        elif dtype in ("boolean", "string"):
            decisions.append({
                "column": col,
                "null_count": info["null_count"],
                "null_pct": null_pct,
                "dtype": dtype,
                "action": "fill",
                "fill_value_type": "mode",
                "reason": (
                    f"Categorical ({dtype}) column — filling with the most "
                    "frequent value (mode) preserves the existing distribution."
                ),
            })

        else:
            decisions.append({
                "column": col,
                "null_count": info["null_count"],
                "null_pct": null_pct,
                "dtype": dtype,
                "action": "flag",
                "fill_value_type": None,
                "reason": (
                    "Unknown type or mixed content — flagging for manual review."
                ),
            })

    return decisions


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Duplicate-handling strategy
# ─────────────────────────────────────────────────────────────────────────────

def decide_duplicate_strategy(
    duplicate_summary: dict[str, Any],
) -> dict[str, Any]:
    """
    Decide how to handle duplicate rows.

    Decision rules:
    - 0 duplicates → no action
    - ≤ 5 % duplicates → safe to drop silently
    - > 5 % → drop but log a warning (large proportion warrants attention)

    Args:
        duplicate_summary: Output of get_duplicate_summary().

    Returns:
        dict: {duplicate_count, duplicate_pct, action, reason}
    """
    count: int = duplicate_summary.get("duplicate_count", 0)
    pct: float = duplicate_summary.get("duplicate_pct", 0.0)

    if count == 0:
        return {
            "duplicate_count": 0,
            "duplicate_pct": 0.0,
            "action": "none",
            "reason": "No duplicate rows detected — no action required.",
        }

    if pct <= 5.0:
        return {
            "duplicate_count": count,
            "duplicate_pct": pct,
            "action": "drop",
            "reason": (
                f"Drop {count} exact duplicate rows ({pct}% of total). "
                "Low proportion — safe to remove silently."
            ),
        }

    return {
        "duplicate_count": count,
        "duplicate_pct": pct,
        "action": "drop_with_warning",
        "reason": (
            f"Drop {count} duplicate rows ({pct}% of total). "
            "High proportion — log a warning so the data source can be investigated."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — Outlier-handling strategy
# ─────────────────────────────────────────────────────────────────────────────

def decide_outlier_strategy(
    outlier_summary: dict[str, dict],
    descriptive_stats: dict[str, dict],
) -> list[dict[str, Any]]:
    """
    Decide the best strategy to handle outliers in each numeric column.

    Decision rules:
    - outlier_pct == 0   → no action
    - outlier_pct ≤ 1 %  → drop the outlier rows (very few, likely errors)
    - outlier_pct ≤ 10 % → cap / Winsorise at the IQR fences
    - outlier_pct > 10 % → apply log-transform or flag for review
      (large proportion suggests a real sub-population, not noise)

    Args:
        outlier_summary:   Output of get_outlier_summary().
        descriptive_stats: Output of get_descriptive_stats().

    Returns:
        List of dicts per column:
        {column, outlier_count, outlier_pct, action, cap_lower, cap_upper, reason}
    """
    decisions: list[dict[str, Any]] = []

    for col, info in outlier_summary.items():
        count: int = info.get("outlier_count", 0)
        pct: float = info.get("outlier_pct", 0.0)
        lower: float = info.get("lower_fence", 0.0)
        upper: float = info.get("upper_fence", 0.0)
        stats: dict = descriptive_stats.get(col, {})
        skew = _skewness_label(
            stats.get("mean", 0.0), stats.get("median", 0.0)
        )

        if count == 0:
            continue

        if pct <= 1.0:
            decisions.append({
                "column": col,
                "outlier_count": count,
                "outlier_pct": pct,
                "action": "drop_rows",
                "cap_lower": None,
                "cap_upper": None,
                "reason": (
                    f"Only {pct}% outliers — likely data-entry errors. "
                    "Safe to drop the affected rows."
                ),
            })

        elif pct <= 10.0:
            decisions.append({
                "column": col,
                "outlier_count": count,
                "outlier_pct": pct,
                "action": "winsorise",
                "cap_lower": round(lower, 4),
                "cap_upper": round(upper, 4),
                "reason": (
                    f"{pct}% outliers — cap values at IQR fences "
                    f"[{round(lower, 2)}, {round(upper, 2)}] (Winsorisation) "
                    "to reduce their influence without losing rows."
                ),
            })

        else:
            transform = "log_transform" if skew == "right-skewed" else "flag_for_review"
            decisions.append({
                "column": col,
                "outlier_count": count,
                "outlier_pct": pct,
                "action": transform,
                "cap_lower": None,
                "cap_upper": None,
                "reason": (
                    f"{pct}% outliers in a {skew} distribution — "
                    + (
                        "apply log-transform to compress the long tail; "
                        "this likely reflects real variation rather than errors."
                        if transform == "log_transform"
                        else
                        "flag for manual review; large outlier proportion with "
                        "non-skewed data suggests mixed populations."
                    )
                ),
            })

    return decisions


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — Type-casting strategy
# ─────────────────────────────────────────────────────────────────────────────

def decide_type_strategy(
    dtypes: dict[str, str],
    cardinality: dict[str, dict],
    null_summary: dict[str, dict],
) -> list[dict[str, Any]]:
    """
    Recommend data-type conversions and encoding strategies.

    Decision rules:
    - numeric column already → ensure stored as float/int, no cast needed
    - boolean column → cast to bool (0/1)
    - datetime column → parse to ISO datetime
    - string column, unique_count ≤ 20 → encode as category / label-encode
    - string column, unique_count > 20 but all rows unique → likely an ID column, keep as-is
    - string column, mid cardinality → one-hot encode if ≤ 10 unique values, else hash-encode

    Args:
        dtypes:      Output of get_dtypes().
        cardinality: Output of get_cardinality().
        null_summary: Output of get_null_summary() — used to skip dropped columns.

    Returns:
        List of dicts per column:
        {column, current_dtype, recommended_action, reason}
    """
    decisions: list[dict[str, Any]] = []

    for col, dtype in dtypes.items():
        card_info: dict = cardinality.get(col, {})
        unique_count: int = card_info.get("unique_count", 0)
        null_pct: float = null_summary.get(col, {}).get("null_pct", 0.0)

        # Skip columns that will be dropped due to high null %
        if null_pct > 60:
            continue

        if dtype == "numeric":
            decisions.append({
                "column": col,
                "current_dtype": dtype,
                "recommended_action": "ensure_numeric",
                "reason": "Already numeric — coerce to float64 to handle any string artefacts.",
            })

        elif dtype == "boolean":
            decisions.append({
                "column": col,
                "current_dtype": dtype,
                "recommended_action": "cast_boolean",
                "reason": "Cast to boolean (True/False or 0/1) for consistent storage.",
            })

        elif dtype == "datetime":
            decisions.append({
                "column": col,
                "current_dtype": dtype,
                "recommended_action": "parse_datetime",
                "reason": "Parse to ISO-8601 datetime; extract year/month/day features if needed.",
            })

        elif dtype == "string":
            if unique_count <= 10:
                decisions.append({
                    "column": col,
                    "current_dtype": dtype,
                    "recommended_action": "one_hot_encode",
                    "reason": (
                        f"Low cardinality ({unique_count} unique values) — "
                        "one-hot encode for ML compatibility."
                    ),
                })
            elif unique_count <= 20:
                decisions.append({
                    "column": col,
                    "current_dtype": dtype,
                    "recommended_action": "label_encode",
                    "reason": (
                        f"Medium cardinality ({unique_count} unique values) — "
                        "label-encode to reduce dimensionality."
                    ),
                })
            else:
                decisions.append({
                    "column": col,
                    "current_dtype": dtype,
                    "recommended_action": "hash_encode_or_keep",
                    "reason": (
                        f"High cardinality ({unique_count} unique values) — "
                        "consider hash-encoding or treating as a free-text / ID field."
                    ),
                })

    return decisions


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 — Overall cleaning priority queue
# ─────────────────────────────────────────────────────────────────────────────

def build_cleaning_priority(
    quality_flags: dict[str, Any],
    null_decisions: list[dict],
    duplicate_decision: dict,
    outlier_decisions: list[dict],
) -> list[dict[str, Any]]:
    """
    Build an ordered list of cleaning steps ranked by severity/impact.

    Priority order:
      1. Drop duplicate rows  (corrupts aggregates immediately)
      2. Drop high-null columns  (reduces noise early)
      3. Handle outliers that require row-drops
      4. Impute / fill nulls
      5. Winsorise / transform remaining outliers
      6. Type casts & encoding

    Args:
        quality_flags:      The quality_flags section from the AnalysisReport.
        null_decisions:     Output of decide_null_strategy().
        duplicate_decision: Output of decide_duplicate_strategy().
        outlier_decisions:  Output of decide_outlier_strategy().

    Returns:
        Ordered list of {step, priority, action_type, target, description}.
    """
    steps: list[dict[str, Any]] = []
    priority = 1

    # Step 1 — duplicates
    if quality_flags.get("has_duplicates"):
        steps.append({
            "step": priority,
            "priority": "HIGH",
            "action_type": "deduplication",
            "target": "all_rows",
            "description": duplicate_decision.get("reason", "Remove duplicate rows."),
        })
        priority += 1

    # Step 2 — drop high-null columns
    drop_cols = [d for d in null_decisions if d["action"] == "drop_column"]
    for d in drop_cols:
        steps.append({
            "step": priority,
            "priority": "HIGH",
            "action_type": "drop_column",
            "target": d["column"],
            "description": d["reason"],
        })
        priority += 1

    # Step 3 — drop-row outliers
    row_drop_outliers = [d for d in outlier_decisions if d["action"] == "drop_rows"]
    for d in row_drop_outliers:
        steps.append({
            "step": priority,
            "priority": "HIGH",
            "action_type": "drop_outlier_rows",
            "target": d["column"],
            "description": d["reason"],
        })
        priority += 1

    # Step 4 — null imputation (fill)
    fill_cols = [d for d in null_decisions if d["action"] == "fill"]
    for d in fill_cols:
        steps.append({
            "step": priority,
            "priority": "MEDIUM",
            "action_type": "impute_nulls",
            "target": d["column"],
            "description": d["reason"],
        })
        priority += 1

    # Step 5 — winsorise / transform
    cap_outliers = [d for d in outlier_decisions if d["action"] in ("winsorise", "log_transform")]
    for d in cap_outliers:
        steps.append({
            "step": priority,
            "priority": "MEDIUM",
            "action_type": d["action"],
            "target": d["column"],
            "description": d["reason"],
        })
        priority += 1

    # Step 6 — flag-only (low priority)
    flag_items = (
        [d for d in null_decisions if d["action"] == "flag"]
        + [d for d in outlier_decisions if d["action"] == "flag_for_review"]
    )
    for d in flag_items:
        steps.append({
            "step": priority,
            "priority": "LOW",
            "action_type": "flag_for_review",
            "target": d["column"],
            "description": d["reason"],
        })
        priority += 1

    return steps
