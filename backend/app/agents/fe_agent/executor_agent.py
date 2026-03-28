"""
app/agents/fe_agent/executor_agent.py

Feature Engineering Executor Agent — Agent 3 (Child) in the FE Pipeline.

Input:  Feature engineering plan (from FE Strategist) + cleaned dataset records
Tool:   execute_python(code: str) -> str
        Runs Python code in a shared sandboxed namespace pre-loaded with:
          df_clean, pd, np, LabelEncoder, StandardScaler, target_col,
          numeric_cols, categorical_cols.
Output: Engineered DataFrame saved to outputs/engineered_data.csv
        and execution log text with step-by-step status + skip reasons.
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
    FunctionTool(get_sandbox_df_info),
    FunctionTool(verify_output_saved),
]


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — expert implementation persona from user spec
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert Python ML engineer specializing in feature engineering.
You receive a feature engineering plan and implement it by writing
and executing Python code.

You have access to one tool: execute_python(code: str) -> str
This tool runs Python code in a shared sandbox that already contains:
df_clean: cleaned pandas DataFrame — THIS IS YOUR INPUT
pd: pandas
np: numpy
LabelEncoder: from sklearn.preprocessing
StandardScaler: from sklearn.preprocessing
target_col: ML target column name (string)
numeric_cols: list of numeric column names
categorical_cols: list of categorical column names

## YOUR WORKFLOW — follow this EXACTLY:

### STEP 1: Setup and verify
Execute code:
```
import json, os
df_engineered = df_clean.copy()
print('Input shape:', df_engineered.shape)
print('Columns:', list(df_engineered.columns))
print('Target col:', target_col)
```

### STEP 2: Drop unnecessary columns
Drop near-zero variance columns
Drop ID/index columns
Drop high multicollinearity columns as per plan
Execute code:
```
cols_to_drop = [list from plan]
cols_to_drop = [c for c in cols_to_drop if c != target_col]
df_engineered.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print('After dropping:', df_engineered.shape)
print('Dropped:', cols_to_drop)
```

### STEP 3: Encode categorical columns
Implement exact encoding strategy from plan
Binary: LabelEncoder
Low cardinality: pd.get_dummies drop_first=True
High cardinality: LabelEncoder
Execute code:
```
# Binary encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in binary_cols:
    df_engineered[col] = le.fit_transform(df_engineered[col].astype(str))
    print(f'Binary encoded: {col}')

# One-hot encoding
df_engineered = pd.get_dummies(df_engineered, 
                                columns=onehot_cols, 
                                drop_first=True)
print('After encoding shape:', df_engineered.shape)
```

### STEP 4: Apply skewness transforms
Execute code:
```
for col, transform in skew_transforms.items():
    if col in df_engineered.columns and col != target_col:
        if transform == 'log1p':
            df_engineered[col] = np.log1p(df_engineered[col])
        elif transform == 'sqrt':
            df_engineered[col] = np.sqrt(df_engineered[col].clip(0))
        print(f'Transformed {col} with {transform}')
        print(f'New skewness: {df_engineered[col].skew():.4f}')
```

### STEP 5: Create interaction features
Execute code:
```
# Create new interaction features as per plan
# Example:
# df_engineered['salary_per_experience'] = df_engineered['salary'] / (df_engineered['experience_years'] + 1)
# Add exact interactions from plan here
print('Interaction features created')
print('Shape after interactions:', df_engineered.shape)
```

### STEP 6: Update sandbox and save
Execute code:
```
# Update globals
globals()['df_engineered'] = df_engineered

# Get final column lists
final_numeric = df_engineered.select_dtypes(include='number').columns.tolist()
if target_col in final_numeric:
    final_numeric.remove(target_col)
final_categorical = df_engineered.select_dtypes(include='object').columns.tolist()

globals()['numeric_cols'] = final_numeric
globals()['categorical_cols'] = final_categorical

# Save engineered data
os.makedirs('outputs', exist_ok=True)
df_engineered.to_csv('outputs/engineered_data.csv', index=False)

# Save feature summary JSON for UI
feature_summary = {
    'total_features': len(df_engineered.columns) - 1,
    'numeric_features': final_numeric,
    'categorical_features': final_categorical,
    'target_col': target_col,
    'shape': list(df_engineered.shape)
}
with open('outputs/feature_summary.json', 'w') as f:
    json.dump(feature_summary, f, indent=2)

print('Final shape:', df_engineered.shape)
print('Saved engineered_data.csv')
print('Saved feature_summary.json')
print('Numeric features:', final_numeric)
print('Categorical features:', final_categorical)
```

### STEP 7: Verify output
Execute code:
```
print('=== FEATURE ENGINEERING COMPLETE ===')
print('Input shape:', df_clean.shape)
print('Output shape:', df_engineered.shape)
print('Features added:', df_engineered.shape[1] - df_clean.shape[1])
print('Nulls in engineered data:', df_engineered.isnull().sum().sum())
print('Target col present:', target_col in df_engineered.columns)
```

## ERROR HANDLING RULES:
If execute_python returns ERROR — read it, fix exact issue, retry
Maximum 3 retries per step
If encoding one column fails — skip it, log it, continue
If transform fails — skip that column, log it, continue
Never drop target_col under any circumstance
Never let one column failure stop entire pipeline

## STRICT RULES:
Always work on df_engineered = df_clean.copy() — never modify df_clean
Always verify target_col is present after every step
Always print shape after every step
Save both CSV and JSON at the end
globals() updates sandbox so next phase gets df_engineered
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent definition — no tools beyond execute_python; helper tools for final checks
# ─────────────────────────────────────────────────────────────────────────────

fe_implementor_agent = Agent(
    name="fe_implementor_agent",
    model="gemini-2.0-flash",
    description=(
        "Implements a feature engineering plan by writing and executing Python "
        "code against df_clean in a sandbox. Applies safe dropping, encoding, "
        "transforms, interactions, and saves engineered_data.csv + "
        "feature_summary.json."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)

# Backward-compatible alias
fe_executor_agent = fe_implementor_agent


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by FE orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_fe_implementor(
    records: list[dict[str, Any]],
    strategist_output: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the FE Implementor Agent and return the execution log.

    Args:
        records:           Cleaned rows as list of row dicts.
        strategist_output: Full dict returned by run_fe_strategist() — expected keys:
                          dataset_id, target_col, numeric_cols, categorical_cols,
                          feature_plan.

    Returns:
        dict with keys:
          dataset_id:         from strategist_output
          output_file:        outputs/engineered_data.csv
          summary_file:       outputs/feature_summary.json
          execution_log:      full agent response text
          sandbox_init:       confirmation from init_sandbox
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    dataset_id: str = strategist_output.get("dataset_id", "unknown")
    feature_plan: str = strategist_output.get("feature_plan", "")
    target_col: str = strategist_output.get("target_col", "")
    numeric_cols: list[str] = strategist_output.get("numeric_cols", [])
    categorical_cols: list[str] = strategist_output.get("categorical_cols", [])

    # Initialise sandbox first: df + required utilities, then expose df_clean.
    sandbox_init_msg = init_sandbox(records, target_col=target_col)
    bootstrap_status = execute_python(
        "df_clean = df.copy(); "
        "globals()['df_clean'] = df_clean; "
        f"globals()['numeric_cols'] = {numeric_cols!r}; "
        f"globals()['categorical_cols'] = {categorical_cols!r}; "
        "print('FE bootstrap -> df_clean shape:', df_clean.shape)"
    )
    if bootstrap_status.startswith("ERROR:"):
        # do not block on bootstrap: continue and let executor handle failure path
        sandbox_init_msg = f"{sandbox_init_msg} | bootstrap: {bootstrap_status}"

    user_message = json.dumps(
        {
            "dataset_id": dataset_id,
            "target_col": target_col,
            "feature_plan": feature_plan,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "info": (
                "Sandbox is ready with df_clean, pd, np, LabelEncoder, "
                "StandardScaler, numeric_cols, and categorical_cols. "
                "Run the 7-step workflow exactly and keep going if one item fails."
            ),
        },
        default=str,
    )

    runner = InMemoryRunner(agent=fe_implementor_agent)
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

    verify_plan = verify_output_saved("outputs/feature_summary.json")
    verify_csv = verify_output_saved("outputs/engineered_data.csv")

    # Optional final sanity check on latest dataframe shape/nulls/columns
    sandbox_info = get_sandbox_df_info()

    return {
        "dataset_id": dataset_id,
        "output_file": "outputs/engineered_data.csv",
        "summary_file": "outputs/feature_summary.json",
        "execution_log": execution_log.strip(),
        "sandbox_init": sandbox_init_msg,
        "bootstrap_status": bootstrap_status,
        "verify_csv": verify_csv,
        "verify_summary": verify_plan,
        "sandbox_info": sandbox_info,
    }


# Backward-compatible alias
run_fe_executor = run_fe_implementor
