"""
test_pipeline.py

Standalone test script — runs the full 3-agent pipeline on a
pre-built dirty dataset WITHOUT needing a database or a running server.

Usage:
    cd backend
    python test_pipeline.py

Optional flags:
    --strategy-only     Run Analyzer + Strategist only (skip code execution)
    --target COLUMN     Override target column (default: salary)

The script uses a built-in sample dataset with deliberate data quality issues:
  - Null values (age, salary, department)
  - Duplicate rows
  - Outlier values (salary = 9999999)
  - Mixed types (age stored as strings in some rows)
  - Categorical columns with low cardinality
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import os

# ── Load .env FIRST — must happen before any google-adk imports ──────────────
# google-adk reads GOOGLE_API_KEY from os.environ at import time.
_backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _backend_dir)

_env_file = os.path.join(_backend_dir, ".env")
if os.path.exists(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip()  # always override stale values
    print(f"[env] Loaded {_env_file}")
else:
    print(f"[env] WARNING: .env not found at {_env_file}")

if "GOOGLE_API_KEY" in os.environ:
    print(f"[env] GOOGLE_API_KEY set (...{os.environ['GOOGLE_API_KEY'][-6:]})")
else:
    print("[env] ERROR: GOOGLE_API_KEY not set — agents will fail!")

from app.agents import run_pipeline  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Built-in dirty sample dataset
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RECORDS: list[dict] = [
    # Normal rows
    {"age": 28, "salary": 55000, "department": "Engineering", "experience_years": 4, "promoted": "yes", "employee_id": "E001"},
    {"age": 34, "salary": 72000, "department": "Marketing",   "experience_years": 8, "promoted": "no",  "employee_id": "E002"},
    {"age": 22, "salary": 38000, "department": "HR",          "experience_years": 1, "promoted": "no",  "employee_id": "E003"},
    {"age": 45, "salary": 95000, "department": "Engineering", "experience_years": 20,"promoted": "yes", "employee_id": "E004"},
    {"age": 31, "salary": 61000, "department": "Marketing",   "experience_years": 6, "promoted": "no",  "employee_id": "E005"},

    # Rows with null values
    {"age": None,  "salary": 52000, "department": "HR",          "experience_years": 3,  "promoted": "yes", "employee_id": "E006"},
    {"age": 29,    "salary": None,  "department": "Engineering",  "experience_years": 5,  "promoted": "no",  "employee_id": "E007"},
    {"age": 38,    "salary": 67000, "department": None,           "experience_years": 10, "promoted": "yes", "employee_id": "E008"},
    {"age": None,  "salary": None,  "department": "Marketing",    "experience_years": 2,  "promoted": "no",  "employee_id": "E009"},

    # Outlier values
    {"age": 27,    "salary": 9999999, "department": "Engineering", "experience_years": 3,  "promoted": "no",  "employee_id": "E010"},
    {"age": 150,   "salary": 58000,   "department": "HR",          "experience_years": 7,  "promoted": "yes", "employee_id": "E011"},

    # Duplicate rows (exact copy of E001)
    {"age": 28, "salary": 55000, "department": "Engineering", "experience_years": 4, "promoted": "yes", "employee_id": "E001"},
    {"age": 28, "salary": 55000, "department": "Engineering", "experience_years": 4, "promoted": "yes", "employee_id": "E001"},

    # More clean rows to give the dataset some body
    {"age": 33, "salary": 63000, "department": "Finance",     "experience_years": 9,  "promoted": "yes", "employee_id": "E012"},
    {"age": 41, "salary": 88000, "department": "Engineering", "experience_years": 15, "promoted": "yes", "employee_id": "E013"},
    {"age": 25, "salary": 42000, "department": "HR",          "experience_years": 2,  "promoted": "no",  "employee_id": "E014"},
    {"age": 36, "salary": 71000, "department": "Finance",     "experience_years": 11, "promoted": "no",  "employee_id": "E015"},
    {"age": 29, "salary": 53000, "department": "Marketing",   "experience_years": 4,  "promoted": "no",  "employee_id": "E016"},
    {"age": 44, "salary": 91000, "department": "Engineering", "experience_years": 18, "promoted": "yes", "employee_id": "E017"},
    {"age": 23, "salary": 39000, "department": "HR",          "experience_years": 1,  "promoted": "no",  "employee_id": "E018"},
]

TARGET_COLUMN = "salary"

# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

SEP = "=" * 70

def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_result(result: dict) -> None:
    section("PIPELINE SUMMARY")
    print(f"  Dataset ID       : {result['dataset_id']}")
    print(f"  Target column    : {result['target_column']}")
    print(f"  Stages completed : {result['stages_completed']} / {result['stages_attempted']}")
    print(f"  Elapsed          : {result.get('elapsed_seconds', '?')}s")

    if result.get("errors"):
        section("⚠  ERRORS")
        for e in result["errors"]:
            print(f"  • {e}")

    # Analysis
    analysis = result.get("analysis")
    if analysis:
        section("STAGE 1 — ANALYSIS REPORT")
        stats = analysis.get("stats", {})
        shape = stats.get("shape", {})
        print(f"  Rows: {shape.get('rows', '?')}  |  Columns: {shape.get('columns', '?')}")
        print(f"  Duplicate rows: {stats.get('duplicate_rows', '?')}")
        print(f"  Numeric cols : {stats.get('numeric_cols', [])}")
        print(f"  Categorical  : {stats.get('categorical_cols', [])}")
        print()
        print(analysis.get("analysis_report", "(no report text)"))

    # Strategy
    strategy = result.get("strategy")
    if strategy:
        section("STAGE 2 — CLEANING STRATEGY")
        print(strategy.get("cleaning_strategy", "(no strategy text)"))

    # Execution
    execution = result.get("execution")
    if execution:
        section("STAGE 3 — EXECUTION LOG")
        print(f"  Sandbox init : {execution.get('sandbox_init', '?')}")
        print(f"  Output file  : {execution.get('output_file', '?')}")
        print()
        print(execution.get("execution_log", "(no log)"))
    elif result["stages_attempted"] < 3:
        section("STAGE 3 — EXECUTION SKIPPED")
        print("  (skip_executor=True — strategy-only run)")

    # Pipeline log
    section("PIPELINE LOG")
    print(result.get("pipeline_log", "(empty)"))

    print(f"\n{SEP}")
    print("  DONE")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main(target_column: str, skip_executor: bool) -> None:
    print(SEP)
    print("  DATA CLEANING PIPELINE — DIRECT TEST")
    print(SEP)
    print(f"  Records     : {len(SAMPLE_RECORDS)} rows")
    print(f"  Columns     : {list(SAMPLE_RECORDS[0].keys())}")
    print(f"  Target col  : {target_column}")
    print(f"  Skip exec   : {skip_executor}")
    print(f"  Known issues: 2 nulls in age, 2 nulls in salary, 1 null in department,")
    print(f"                2 duplicate rows, 2 outliers (salary=9999999, age=150)")
    print(SEP)

    result = await run_pipeline(
        dataset_id="test-direct-run",
        records=SAMPLE_RECORDS,
        target_column=target_column,
        skip_executor=skip_executor,
    )

    print_result(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the data cleaning pipeline directly.")
    parser.add_argument(
        "--strategy-only",
        action="store_true",
        help="Stop after strategy — skip code execution.",
    )
    parser.add_argument(
        "--target",
        default=TARGET_COLUMN,
        help=f"Target column name (default: {TARGET_COLUMN}).",
    )
    args = parser.parse_args()

    asyncio.run(main(
        target_column=args.target,
        skip_executor=args.strategy_only,
    ))
