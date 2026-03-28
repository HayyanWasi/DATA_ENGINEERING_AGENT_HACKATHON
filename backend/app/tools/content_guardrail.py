"""
app/tools/content_guardrail.py

Content-safety guardrail for the ML pipeline.

Scans uploaded datasets for hate speech, slurs, discriminatory profiling
intent, and harmful themes BEFORE any agent or model training begins.

Three-layer check:
  1. Column names scanned for slurs and harmful themes.
  2. Target column checked for protected-attribute profiling.
  3. A sample of cell values scanned for hateful content.

Usage:
    from app.tools.content_guardrail import check_content_guardrail
    result = check_content_guardrail(records, target_column, column_names)
    if not result["allowed"]:
        # reject dataset — result["message"] has user-facing explanation
"""

from __future__ import annotations

import re
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Blocked-content pattern lists (all matched case-insensitively with \b)
# ─────────────────────────────────────────────────────────────────────────────

# Explicit slurs — racial, gender, homophobic, disability.
# Each entry is a raw regex fragment compiled with re.IGNORECASE.
_SLUR_PATTERNS: list[str] = [
    # Racial slurs
    r"\bni+g+[aehrs]+\b",
    r"\bk[iy]+ke\b",
    r"\bwetback\b",
    r"\bgook\b",
    r"\btowel\s*head\b",
    r"\bsand\s*ni+g+\b",
    # Misogynistic slurs
    r"\bc[u]+nt\b",
    # Homophobic slurs
    r"\bf[a]+g+[o0]+t\b",
    r"\btr[a]+nn[yie]+\b",
    # White-supremacy markers
    r"\baryan\s*supremac",
    r"\bwhite\s*power\b",
    r"\bsieg\s*heil\b",
    r"\b1488\b",
    r"\b14[\s/_-]*88\b",
    r"\bheil\s*hitler\b",
]

# Protected attributes — blocking when used as the PREDICTION TARGET
# (predicting these = discriminatory profiling).
_PROTECTED_TARGET_PATTERNS: list[str] = [
    r"\brace\b",
    r"\bracial\b",
    r"\bethnicity\b",
    r"\bethnic[_\s]?group\b",
    r"\breligion\b",
    r"\breligious[_\s]?affiliation\b",
    r"\bsexual[_\s]?orientation\b",
    r"\bskin[_\s]?colou?r\b",
    r"\bnational[_\s]?origin\b",
]

# Harmful themes — in column names or cell values
_HARMFUL_THEME_PATTERNS: list[str] = [
    r"\bhate[_\s]?score\b",
    r"\bhate[_\s]?level\b",
    r"\bradicali[sz]ation\b",
    r"\beugenics?\b",
    r"\bgenetic[_\s]?superiority\b",
    r"\bracial[_\s]?inferiority\b",
    r"\bfinal[_\s]?solution\b",
]

# ─────────────────────────────────────────────────────────────────────────────
# Compiled regexes (done once at import time)
# ─────────────────────────────────────────────────────────────────────────────

_FLAGS = re.IGNORECASE

_SLUR_RX:   list[re.Pattern[str]] = [re.compile(p, _FLAGS) for p in _SLUR_PATTERNS]
_TARGET_RX: list[re.Pattern[str]] = [re.compile(p, _FLAGS) for p in _PROTECTED_TARGET_PATTERNS]
_THEME_RX:  list[re.Pattern[str]] = [re.compile(p, _FLAGS) for p in _HARMFUL_THEME_PATTERNS]

# How many rows to sample for cell-value scanning
_VALUE_SAMPLE_SIZE = 500

_REJECTION_PREFIX = (
    "CONTENT SAFETY VIOLATION — This dataset has been rejected.\n"
    "Our AutoML agent does not train models on datasets containing "
    "hate speech, slurs, or content designed for discriminatory profiling."
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def check_content_guardrail(
    records: list[dict[str, Any]],
    target_column: str,
    column_names: list[str],
) -> dict[str, Any]:
    """
    Scan a dataset for inappropriate, hateful, or discriminatory content.

    Runs three checks in order:
      1. **Column names** scanned for slurs and harmful themes.
      2. **Target column** checked for protected-attribute profiling intent
         (e.g. predicting someone's race or religion is not permitted).
      3. **Cell values** (sampled) scanned for slurs and hate speech.

    Args:
        records:       Raw dataset rows (list of dicts).
        target_column: Column the user wants to predict.
        column_names:  All column headers in the dataset.

    Returns:
        {
            "allowed":    bool   — True if the dataset passes all checks,
            "violations": list   — specific reasons for rejection,
            "warnings":   list   — non-blocking advisories,
            "message":    str    — user-facing summary,
        }
    """
    violations: list[str] = []
    warnings:   list[str] = []

    # ── 1. Scan column names ──────────────────────────────────────────────
    # Normalise underscores/hyphens → spaces so \b works across separators
    # e.g. "wetback_score" → "wetback score" which matches \bwetback\b
    cols_text = "  ".join(column_names)
    cols_lower = re.sub(r"[_\-]", " ", cols_text).lower()

    for rx in _SLUR_RX:
        m = rx.search(cols_lower)
        if m:
            violations.append(
                f"Column names contain prohibited hate-speech term: '{m.group()}'."
            )

    for rx in _THEME_RX:
        m = rx.search(cols_lower)
        if m:
            violations.append(
                f"Column names suggest a harmful modelling theme: '{m.group()}'."
            )

    # ── 2. Check target column for discriminatory profiling ───────────────
    target_lower = re.sub(r"[_\-]", " ", target_column).lower().strip()

    for rx in _TARGET_RX:
        if rx.search(target_lower):
            violations.append(
                f"Target column '{target_column}' is a protected attribute. "
                "Training a model to predict race, ethnicity, religion, "
                "sexual orientation, or national origin constitutes "
                "discriminatory profiling and is not permitted."
            )
            break                               # one match is enough

    # ── 3. Sample cell values for slurs ───────────────────────────────────
    sample = records[:_VALUE_SAMPLE_SIZE]
    flagged: set[str] = set()

    for row in sample:
        for val in row.values():
            if not isinstance(val, str) or len(val) < 3:
                continue
            val_lower = val.lower()
            for rx in _SLUR_RX:
                m = rx.search(val_lower)
                if m and m.group() not in flagged:
                    flagged.add(m.group())
                    violations.append(
                        f"Cell values contain prohibited hate-speech term: "
                        f"'{m.group()}'."
                    )
            for rx in _THEME_RX:
                m = rx.search(val_lower)
                if m and m.group() not in flagged:
                    flagged.add(m.group())
                    violations.append(
                        f"Cell values contain harmful theme reference: "
                        f"'{m.group()}'."
                    )

    # ── Build result ──────────────────────────────────────────────────────
    allowed = len(violations) == 0

    if allowed:
        message = "Content guardrail passed. No inappropriate content detected."
    else:
        detail = "\n".join(f"  - {v}" for v in violations)
        message = (
            f"{_REJECTION_PREFIX}\n\n"
            f"Violations found ({len(violations)}):\n{detail}\n\n"
            "Please remove the offending content and re-upload."
        )

    return {
        "allowed":    allowed,
        "violations": violations,
        "warnings":   warnings,
        "message":    message,
    }
