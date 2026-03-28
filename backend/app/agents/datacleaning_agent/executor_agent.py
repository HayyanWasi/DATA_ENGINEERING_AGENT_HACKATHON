"""
app/agents/datacleaning_agent/executor_agent.py

Executor Agent — Agent 3 (Child) in the Data Cleaning Pipeline.

Input:  Cleaning strategy text (from Strategist) + raw dataset records
Tool:   execute_python(code: str) → str
        Runs Python code in a sandboxed namespace pre-loaded with:
          df, pd, np, LabelEncoder, StandardScaler
Output: Cleaned DataFrame saved to outputs/cleaned_data.csv
        Execution log string with step-by-step results
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.tools.executor_tools import (
    execute_python,
    get_sandbox_df_info,
    init_sandbox,
    verify_output_saved,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tool registration
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS: list[FunctionTool] = [
    FunctionTool(execute_python),
    FunctionTool(get_sandbox_df_info),   # ✅ verify df mutation after cleaning
    FunctionTool(verify_output_saved),   # ✅ assert CSV was written to disk
]

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — exact Python data engineer persona from spec
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert Python data engineer. You receive a data cleaning strategy plan
and implement it by writing and executing Python code.

You have access to one tool: `execute_python(code: str) -> str`
This tool runs Python code in a shared sandbox that already contains:
  - df              — the raw pandas DataFrame to clean
  - pd              — pandas
  - np              — numpy
  - LabelEncoder    — from sklearn.preprocessing
  - StandardScaler  — from sklearn.preprocessing

The sandbox state persists across all your execute_python calls.
Any variable you create or change to df in one call is available in the next.

## YOUR WORKFLOW — follow this EXACTLY, step by step:

### STEP 1 — Verify data is loaded
Call execute_python with:
```
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.head(3))
```
Read the output carefully before proceeding.

### STEP 2 — Implement cleaning plan
Write ONE block of Python code that implements the FULL cleaning strategy.
The code must:
- Handle each null column as specified in the strategy
- Drop duplicates if specified
- Fix data types if specified
- Handle outliers as specified (cap, drop rows, log-transform)
- Encode categoricals as specified (LabelEncoder or get_dummies)
- Print df.shape and df.isnull().sum() after each major operation as a checkpoint

Comment every operation clearly. Example:
```python
# Step: Fill nulls in 'age' with median (strategy: numeric, skewed)
median_age = df['age'].median()
df['age'].fillna(median_age, inplace=True)
print(f"age nulls after fill: {df['age'].isnull().sum()}")
```

### STEP 3 — Verify cleaning worked
Call execute_python with:
```
print('=== CLEANING VERIFICATION ===')
print('Final shape:', df.shape)
print('Remaining nulls:', df.isnull().sum().sum())
print('Duplicates:', df.duplicated().sum())
print('Dtypes after cleaning:')
print(df.dtypes)
```

### STEP 4 — Save cleaned data
Call execute_python with:
```
df.to_csv('outputs/cleaned_data.csv', index=False)
print(f'Saved cleaned_data.csv — shape: {df.shape}')
```

### STEP 5 — Final verification (mandatory)
Call `get_sandbox_df_info()` — no arguments needed.
Read the returned dict and confirm:
  - `null_total` is 0 or acceptably low (some columns may intentionally have nulls)
  - `duplicate_count` is 0
  - `target_col` matches the expected target column
  - `shape` reflects the cleaned dimensions

Then call `verify_output_saved()` — no arguments needed.
If `exists` is False — retry Step 4 once, then report the error.

## ERROR HANDLING RULES
- If execute_python returns a line starting with "ERROR:" — read the error carefully
- Fix the exact issue in your code (e.g. KeyError → wrong column name, TypeError → wrong type operation)
- Retry the SAME step — maximum 3 retries per step
- If still failing after 3 retries — skip that specific sub-step, print a SKIP warning, continue

## STRICT RULES
- Always run Step 1 first — never assume df is ready
- Never overwrite df with None or an empty DataFrame
- Always print results after each execute_python call so progress is visible
- Use only: df, pd, np, LabelEncoder, StandardScaler — do NOT import anything else
- Keep code clean and comment every operation
- Do NOT include backtick fences in the code string you pass to execute_python — raw Python only
"""

# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

executor_agent = Agent(
    name="executor_agent",
    model="gemini-2.5-flash",
    description=(
        "Implements a data cleaning strategy by writing and executing Python "
        "code in a sandboxed pandas environment. Saves the cleaned DataFrame "
        "to outputs/cleaned_data.csv and returns a full execution log."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by orchestrator / API route
# ─────────────────────────────────────────────────────────────────────────────

async def run_executor(
    records: list[dict[str, Any]],
    strategist_output: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the Executor Agent against a dataset and cleaning strategy.

    Steps:
      1. Initialises the sandbox (df + libraries) via init_sandbox().
      2. Sends the strategy to the Executor Agent.
      3. The agent calls execute_python() multiple times to clean the data.
      4. Returns the execution log and output file path.

    Args:
        records:           Raw rows as a list of plain dicts.
        strategist_output: The full dict returned by run_strategist() — must
                           contain keys: dataset_id, target_column, cleaning_strategy.

    Returns:
        dict with keys:
          "dataset_id"      — from strategist_output
          "output_file"     — relative path to the saved CSV
          "execution_log"   — full agent response text
          "sandbox_init"    — confirmation string from init_sandbox
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    dataset_id: str = strategist_output.get("dataset_id", "unknown")
    cleaning_strategy: str = strategist_output.get("cleaning_strategy", "")
    target_column: str = strategist_output.get("target_column", "")

    # Initialise sandbox BEFORE the agent starts — df + target_col must be ready
    sandbox_init_msg = init_sandbox(records, target_col=target_column)

    user_message = json.dumps(
        {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "cleaning_strategy": cleaning_strategy,
            "info": "The sandbox is already initialised. df is ready. Follow the 4-step workflow.",
        },
        default=str,
    )

    runner = InMemoryRunner(agent=executor_agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id="system",
    )

    execution_log = ""
    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_message)],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            execution_log = event.content.parts[0].text or ""

    return {
        "dataset_id": dataset_id,
        "output_file": "outputs/cleaned_data.csv",
        "execution_log": execution_log.strip(),
        "sandbox_init": sandbox_init_msg,
    }
