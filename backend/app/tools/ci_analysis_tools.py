"""
app/tools/ci_analysis_tools.py

Single class-imbalance analysis tool consumed by the Class Imbalance Analyzer.
Computes label counts/percentages for train and test splits and returns the full
imbalance dictionary expected by the prompt.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_list(values: Any) -> list[Any]:
    """Coerce any supported sequence-like input into a Python list."""
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    try:
        return list(values)
    except TypeError:
        return [values]


def _is_missing(value: Any) -> bool:
    """Return True when a target value should be treated as missing."""
    if value is None or value == "":
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _build_distribution(values: list[Any]) -> dict[Any, int]:
    """Build value counts from label list, excluding missing entries."""
    clean_values = [value for value in values if not _is_missing(value)]
    return dict(Counter(clean_values))


def _build_percentages(distribution: dict[Any, int], total: int) -> dict[Any, float]:
    """Build percentage dict from a count dict using total sample count."""
    if total <= 0:
        return {}
    return {
        key: round(count / total * 100, 2)
        for key, count in distribution.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main tool — class-imbalance profile
# ─────────────────────────────────────────────────────────────────────────────

def analyze_class_imbalance(
    y_train: Any,
    y_test: Any,
    target_col: str,
    target_dtype: str,
) -> dict[str, Any]:
    """
    Compute train/test target distributions and imbalance metadata.

    Args:
        y_train:      Training labels (list-like or pandas Series).
        y_test:       Test labels (list-like or pandas Series).
        target_col:   Name of the target column.
        target_dtype: Dtype hint for the target column.

    Returns:
        dict with keys:
          y_train_distribution      — value -> count from train split
          y_train_percentages       — value -> percentage from train split
          y_test_distribution       — value -> count from test split
          total_train_samples       — number of non-missing train samples
          total_test_samples        — number of non-missing test samples
          target_col                — target column name
          target_dtype              — target dtype string
          unique_classes            — ordered list of unique classes in y_train
          class_count               — number of unique classes
          minority_class            — label with the smallest class count
          minority_class_percentage — percentage of minority class
          majority_class            — label with the largest class count
          majority_class_percentage — percentage of majority class
          imbalance_ratio           — majority_count / minority_count
    """
    y_train_list = _to_list(y_train)
    y_test_list = _to_list(y_test)

    # Train / test distributions (ignore missing labels for stable class math)
    train_distribution = _build_distribution(y_train_list)
    test_distribution = _build_distribution(y_test_list)

    total_train_samples = sum(train_distribution.values())
    total_test_samples = sum(test_distribution.values())

    y_train_percentages = _build_percentages(
        train_distribution,
        total_train_samples,
    )

    unique_classes = list(train_distribution.keys())
    class_count = len(unique_classes)

    if class_count > 0:
        majority_class = max(train_distribution, key=train_distribution.get)
        minority_class = min(train_distribution, key=train_distribution.get)
        majority_count = train_distribution[majority_class]
        minority_count = train_distribution[minority_class]
        majority_pct = y_train_percentages.get(majority_class, 0.0)
        minority_pct = y_train_percentages.get(minority_class, 0.0)
        imbalance_ratio = round(majority_count / minority_count, 4) if minority_count else 0.0
    else:
        majority_class = None
        minority_class = None
        majority_pct = 0.0
        minority_pct = 0.0
        imbalance_ratio = 0.0

    return {
        "y_train_distribution": train_distribution,
        "y_train_percentages": y_train_percentages,
        "y_test_distribution": test_distribution,
        "total_train_samples": total_train_samples,
        "total_test_samples": total_test_samples,
        "target_col": target_col,
        "target_dtype": target_dtype,
        "unique_classes": unique_classes,
        "class_count": class_count,
        "minority_class": minority_class,
        "minority_class_percentage": minority_pct,
        "majority_class": majority_class,
        "majority_class_percentage": majority_pct,
        "imbalance_ratio": imbalance_ratio,
    }
