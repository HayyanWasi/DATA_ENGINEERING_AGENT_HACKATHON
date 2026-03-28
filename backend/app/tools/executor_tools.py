"""
app/tools/executor_tools.py

Sandbox execution tool consumed by the Executor Agent.

Design:
  - A module-level _SANDBOX dict holds the shared execution namespace.
  - init_sandbox(records) initialises df + standard libraries in that namespace.
  - execute_python(code) runs arbitrary code inside the namespace and returns
    captured stdout (or an ERROR string).
  - The Executor Agent calls init_sandbox once (via run_executor), then calls
    execute_python repeatedly across its 4-step workflow.
"""

from __future__ import annotations

import io
import sys
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Module-level shared sandbox namespace
# ─────────────────────────────────────────────────────────────────────────────

_SANDBOX: dict[str, Any] = {}


def init_sandbox(
    records: list[dict[str, Any]],
    target_col: str = "",
) -> str:
    """
    Initialise the shared execution sandbox.

    Populates _SANDBOX with:
      df             — pandas DataFrame built from *records*
      pd             — pandas module
      np             — numpy module
      LabelEncoder   — sklearn.preprocessing.LabelEncoder
      StandardScaler — sklearn.preprocessing.StandardScaler
      target_col     — name of the ML target column (string)

    This must be called ONCE before any execute_python() calls for a session.

    Args:
        records:    Raw dataset rows as a list of plain dicts.
        target_col: Name of the ML target column — available inside the
                    sandbox so downstream phases (EDA, training) can use it.

    Returns:
        A confirmation string with df.shape for verification.
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import os

    df = pd.DataFrame(records)

    _SANDBOX.clear()
    _SANDBOX.update(
        {
            "df": df,
            "df_raw": df.copy(),   # preserved snapshot — used by EDA comparison charts
            "pd": pd,
            "np": np,
            "LabelEncoder": LabelEncoder,
            "StandardScaler": StandardScaler,
            "target_col": target_col,
        }
    )

    # Ensure outputs directory exists and persist raw snapshot to disk
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/raw_data.csv", index=False)

    return (
        f"Sandbox initialised. df.shape={df.shape}, "
        f"columns={list(df.columns)}, "
        f"target_col='{target_col}'"
    )


def execute_python(code: str) -> str:
    """
    Execute Python code in the shared sandbox namespace and return stdout output.

    The sandbox already contains: df, pd, np, LabelEncoder, StandardScaler,
    target_col.
    Any changes made to df or other variables persist across calls because
    exec() mutates the _SANDBOX dict in-place.

    Args:
        code: Valid Python source code to execute.

    Returns:
        Captured stdout as a string, or an ERROR line if an exception occurs.
        Format on error: "ERROR: <ExceptionType>: <message>"
    """
    if not _SANDBOX:
        return (
            "ERROR: SandboxNotInitialised: Sandbox is empty. "
            "init_sandbox must be called first."
        )

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        exec(code, _SANDBOX)  # noqa: S102
        # ✅ exec() writes variables back into _SANDBOX automatically because
        # _SANDBOX IS the globals dict — df mutations persist across all calls.
        output = buffer.getvalue()
        return output if output.strip() else "Code executed successfully (no output printed)."
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {type(exc).__name__}: {exc}"
    finally:
        sys.stdout = old_stdout


def get_sandbox_df_info() -> dict[str, Any]:
    """
    Inspect the current state of df inside the sandbox.

    Call this AFTER execute_python() steps complete to confirm that
    df was actually mutated (cleaned shape ≠ original dirty shape).

    Returns:
        dict with:
          initialised  (bool)   — is sandbox ready?
          shape        (tuple)  — current df.shape
          null_total   (int)    — total remaining nulls across all columns
          duplicate_count (int) — duplicate rows remaining
          columns      (list)   — column names after cleaning
          target_col   (str)    — target column in sandbox
    """
    if not _SANDBOX or "df" not in _SANDBOX:
        return {
            "initialised": False,
            "shape": None,
            "null_total": None,
            "duplicate_count": None,
            "columns": [],
            "target_col": "",
        }

    df = _SANDBOX["df"]
    return {
        "initialised": True,
        "shape": list(df.shape),
        "null_total": int(df.isnull().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "columns": list(df.columns),
        "target_col": _SANDBOX.get("target_col", ""),
    }


def init_eda_sandbox(
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> str:
    """
    Prepare the shared sandbox for the EDA phase.

    Loads:
      - outputs/cleaned_data.csv  → df_clean  (cleaned data)
      - outputs/raw_data.csv      → df        (raw data — pre-cleaning snapshot)

    The executor prompt uses df (raw) and df_clean (cleaned) by those exact names,
    so df is intentionally reassigned to the raw snapshot here.

    Injects EDA-specific libraries (matplotlib Agg, seaborn, os, joblib) and
    exposes numeric_cols / categorical_cols as sandbox variables so the executor
    can iterate over them directly without parsing.

    Must be called ONCE before any EDA execute_python() calls.

    Args:
        numeric_cols:    List of numeric column names from analyze_eda() stats.
                         Falls back to pandas select_dtypes if not provided.
        categorical_cols: List of categorical column names from analyze_eda() stats.
                          Falls back to pandas select_dtypes if not provided.

    Returns:
        Confirmation string summarising shapes and column counts.
    """
    import os
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe for server use
    import matplotlib.pyplot as plt
    import joblib

    cleaned_path = "outputs/cleaned_data.csv"
    raw_path = "outputs/raw_data.csv"

    if not os.path.isfile(cleaned_path):
        return (
            f"ERROR: SandboxInitFailed: {cleaned_path} not found. "
            "Run the data cleaning phase first."
        )

    df_clean = pd.read_csv(cleaned_path)

    # df = raw snapshot — falls back to df_clean if raw_data.csv wasn't saved
    raw_available = os.path.isfile(raw_path)
    df_raw = pd.read_csv(raw_path) if raw_available else df_clean.copy()

    # Resolve column lists — prefer caller-supplied (from analyze_eda stats),
    # fall back to pandas dtype inference on df_clean
    resolved_numeric = numeric_cols if numeric_cols is not None else (
        df_clean.select_dtypes(include="number").columns.tolist()
    )
    resolved_categorical = categorical_cols if categorical_cols is not None else (
        df_clean.select_dtypes(exclude="number").columns.tolist()
    )

    _SANDBOX.update(
        {
            # df = raw (per executor prompt contract: df is RAW, df_clean is CLEANED)
            "df": df_raw,
            "df_clean": df_clean,
            "plt": plt,
            "pd": pd,
            "os": os,
            "joblib": joblib,
            # Column lists available as sandbox variables for direct iteration
            "numeric_cols": resolved_numeric,
            "categorical_cols": resolved_categorical,
        }
    )

    os.makedirs("charts", exist_ok=True)

    return (
        f"EDA sandbox ready. "
        f"df.shape={df_raw.shape} (raw), "
        f"df_clean.shape={df_clean.shape} (cleaned), "
        f"numeric_cols={resolved_numeric}, "
        f"categorical_cols={resolved_categorical}, "
        f"raw_data_available={raw_available}."
    )


def verify_charts_saved(charts_dir: str = "charts") -> dict[str, Any]:
    """
    Check which PNG chart files were saved to the charts/ directory.

    Args:
        charts_dir: Directory to scan (default: 'charts').

    Returns:
        dict with:
          chart_count  (int)  — total PNG files found
          chart_files  (list) — filenames of saved charts
          charts_dir   (str)  — the scanned directory
          message      (str)  — human-readable summary
    """
    import os

    if not os.path.isdir(charts_dir):
        return {
            "chart_count": 0,
            "chart_files": [],
            "charts_dir": charts_dir,
            "message": f"❌ Directory not found: {charts_dir}",
        }

    png_files = sorted(
        f for f in os.listdir(charts_dir) if f.lower().endswith(".png")
    )
    count = len(png_files)
    msg = (
        f"✅ {count} chart(s) saved in {charts_dir}/: {png_files}"
        if count > 0
        else f"❌ No PNG charts found in {charts_dir}/"
    )
    return {
        "chart_count": count,
        "chart_files": png_files,
        "charts_dir": charts_dir,
        "message": msg,
    }


def verify_output_saved(filepath: str = "outputs/cleaned_data.csv") -> dict[str, Any]:
    """
    Assert that the cleaned CSV was actually written to disk.

    Args:
        filepath: Path to check (default: 'outputs/cleaned_data.csv').

    Returns:
        dict with:
          exists      (bool) — True if file is on disk
          filepath    (str)  — the checked path
          size_bytes  (int)  — file size in bytes (0 if not found)
          message     (str)  — human-readable result
    """
    import os

    exists = os.path.isfile(filepath)
    size = os.path.getsize(filepath) if exists else 0
    return {
        "exists": exists,
        "filepath": filepath,
        "size_bytes": size,
        "message": (
            f"✅ File saved: {filepath} ({size} bytes)"
            if exists
            else f"❌ File NOT found at: {filepath}"
        ),
    }
