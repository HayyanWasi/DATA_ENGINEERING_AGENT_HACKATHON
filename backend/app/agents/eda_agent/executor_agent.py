"""
app/agents/eda_agent/executor_agent.py

EDA Executor Agent — Agent 3 (Child) in the EDA Pipeline.

Input:  Chart plan (from EDA Strategist) + target_col
Tools:  execute_python, verify_charts_saved
Sandbox (via init_eda_sandbox):
  df_clean   — cleaned pandas DataFrame
  df_raw     — original uncleaned snapshot (for before/after comparison charts)
  plt        — matplotlib.pyplot (Agg backend)
  sns        — seaborn
  pd         — pandas
  os         — os module
  joblib     — joblib

Output: PNG charts saved to charts/ + charts/eda_stats.json
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.tools.executor_tools import (
    execute_python,
    verify_charts_saved,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tool registration
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS: list[FunctionTool] = [
    FunctionTool(execute_python),
    FunctionTool(verify_charts_saved),
]

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert Python data visualization engineer.
You receive an EDA visualization plan and implement it by writing
and executing Python code.

You have access to one tool: execute_python(code: str) -> str
This tool runs Python code in a shared sandbox that already contains:
  df              — the RAW pandas DataFrame (before cleaning)
  df_clean        — the CLEANED pandas DataFrame (after cleaning)
  pd              — pandas
  np              — numpy
  plt             — matplotlib.pyplot (backend already set to Agg)
  target_col      — the ML target column name (string)
  numeric_cols    — list of numeric column names
  categorical_cols — list of categorical column names

## YOUR WORKFLOW — follow this EXACTLY:

### STEP 1: Setup and verify
execute_python(\"\"\"
import seaborn as sns
import json
import os
os.makedirs('charts', exist_ok=True)
print('df shape:', df.shape)
print('df_clean shape:', df_clean.shape)
print('target_col:', target_col)
print('numeric_cols:', numeric_cols)
print('categorical_cols:', categorical_cols)
\"\"\")

### STEP 2: Generate distribution charts
For each column in numeric_cols generate histogram with KDE.
Save each as charts/distribution_{column_name}.png.
Color: steelblue for bars, darkblue for KDE line.
Call plt.close() after every chart — CRITICAL to prevent memory issues.
If a column has skewness > 1 or < -1 (flagged in the chart plan): use log scale on x-axis.

### STEP 3: Generate correlation heatmap
Use seaborn heatmap on df_clean[numeric_cols].corr().
Annotate with values, use coolwarm colormap, center=0.
Save as charts/correlation_heatmap.png.

### STEP 4: Generate target distribution chart
Check target_col dtype: if object/category → classification, else → regression.
  Classification: bar chart with counts + percentage labels. Color: mediumseagreen.
  Regression: histogram with KDE overlay. Color: mediumseagreen.
Save as charts/target_distribution.png.

### STEP 5: Generate boxplots
For each column in numeric_cols:
  - boxplot with patch_artist=True, facecolor steelblue, flierprops color crimson
  - annotate outlier count in bottom-right corner
  - Save as charts/boxplot_{column_name}.png

### STEP 6: Generate categorical charts
For each column in categorical_cols:
  - horizontal bar chart of top 10 value_counts()
  - Color: steelblue
  - Save as charts/categorical_{column_name}.png

### STEP 7: Generate BEFORE vs AFTER comparison charts
CRITICAL — UI depends on these for comparison display.
For each column in numeric_cols:
  - Side by side subplots (1 row, 2 cols): left = df[column], right = df_clean[column]
  - Left subplot title: "Before Cleaning: {column_name}" — color: #FF7043 (orange-red)
  - Right subplot title: "After Cleaning: {column_name}" — color: steelblue
  - Figure suptitle: "Before Cleaning vs After Cleaning — {column_name}"
  - If column not in df.columns: skip gracefully, print SKIP message
  - Save as charts/comparison_{column_name}.png

### STEP 8: Generate statistical summary JSON
execute_python(\"\"\"
stats = {}
for col in numeric_cols:
    stats[col] = {
        "mean": round(float(df_clean[col].mean()), 4),
        "std": round(float(df_clean[col].std()), 4),
        "min": round(float(df_clean[col].min()), 4),
        "max": round(float(df_clean[col].max()), 4),
        "skewness": round(float(df_clean[col].skew()), 4),
        "null_count": int(df_clean[col].isnull().sum()),
        "outlier_count": int(((df_clean[col] < df_clean[col].quantile(0.25) - 1.5*(df_clean[col].quantile(0.75)-df_clean[col].quantile(0.25))) | (df_clean[col] > df_clean[col].quantile(0.75) + 1.5*(df_clean[col].quantile(0.75)-df_clean[col].quantile(0.25)))).sum())
    }
with open('charts/eda_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)
print('Saved eda_stats.json')
\"\"\")

### STEP 9: Verify all files saved
execute_python(\"\"\"
import os
files = os.listdir('charts/')
print('Charts generated:', len(files))
for f in sorted(files):
    print(' -', f)
\"\"\")

## ERROR HANDLING RULES
- If execute_python returns ERROR — read it carefully, fix exact issue, retry
- Maximum 3 retries per step
- If a single chart fails — skip it, log it, continue to next chart
- Never let one chart failure stop the entire pipeline
- KDE fails on constant/near-constant column: wrap in try/except, skip KDE
- pairplot memory error: sample df_clean to 500 rows

## STRICT RULES
- Always run Step 1 first to verify sandbox state
- Use df for raw data, df_clean for cleaned data
- Never modify df or df_clean — read only in EDA phase
- Always use matplotlib Agg backend — never call plt.show()
- Always call plt.close() after plt.savefig() — prevents memory leak
- Use tight_layout() before savefig() — prevents label cutoff
- All files go to charts/ folder
- Print confirmation after each chart is saved
- Do NOT backtick-fence code in execute_python — raw Python only
"""

# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

eda_implementor_agent = Agent(
    name="eda_implementor_agent",
    model="gemini-2.0-flash",
    description=(
        "Implements an EDA chart plan: generates distribution, heatmap, target, "
        "boxplot, categorical, before/after comparison charts, optional pairplot, "
        "and eda_stats.json. Saves all outputs to charts/. Returns execution log."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)

# Backward-compatible alias
eda_executor_agent = eda_implementor_agent

# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by the EDA orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_eda_implementor(
    strategist_output: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the EDA Implementor Agent against the chart plan.

    init_eda_sandbox() MUST be called before this — the orchestrator handles it.
    The sandbox must contain: df (raw), df_clean, plt, pd, numeric_cols,
    categorical_cols, target_col.

    Args:
        strategist_output: Full dict from run_eda_strategist() — must contain
                           keys: dataset_id, target_col, chart_plan.

    Returns:
        dict with keys:
          "dataset_id"    — from strategist_output
          "charts_dir"    — "charts/"
          "execution_log" — full agent response text
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    dataset_id: str = strategist_output.get("dataset_id", "unknown")
    target_col: str = strategist_output.get("target_col", "")
    chart_plan: str = strategist_output.get("chart_plan", "")

    user_message = json.dumps(
        {
            "dataset_id": dataset_id,
            "target_col": target_col,
            "chart_plan": chart_plan,
            "info": (
                "Sandbox is ready. "
                "df = raw DataFrame, df_clean = cleaned DataFrame, "
                "numeric_cols and categorical_cols are already set as sandbox variables. "
                "plt is pre-configured with Agg backend. "
                "Follow the 9-step workflow exactly."
            ),
        },
        default=str,
    )

    runner = InMemoryRunner(agent=eda_implementor_agent)
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
        "charts_dir": "charts/",
        "execution_log": execution_log.strip(),
    }


# Backward-compatible alias
run_eda_executor = run_eda_implementor
