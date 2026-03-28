"""
app/agents/scaling_agent/executor_agent.py

Feature Scaling Executor Agent — Agent 3 (Child) in the Scaling Pipeline.

Input:  Scaling plan (from Scaling Strategist) + engineered dataset records
Tool:   execute_python(code: str) -> str
        Runs Python code in a shared sandbox pre-loaded with:
          df_engineered, pd, np, StandardScaler, RobustScaler,
          MinMaxScaler, train_test_split, target_col, numeric_cols.
Output: Train/test splits plus scaled feature artifacts:
        outputs/train_data.csv, outputs/test_data.csv,
        outputs/scaling_summary.json, and fitted scaler .pkl files.
"""

from __future__ import annotations

import json
import re
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
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_PATTERNS = {
    "skip": re.compile(r"^\s*1\s*[.)]\s*.*SKIP", re.IGNORECASE),
    "standard": re.compile(r"^\s*2\s*[.)]\s*.*STANDARD", re.IGNORECASE),
    "robust": re.compile(r"^\s*3\s*[.)]\s*.*ROBUST", re.IGNORECASE),
    "minmax": re.compile(r"^\s*4\s*[.)]\s*.*MINMAX", re.IGNORECASE),
}


def _norm_col_token(token: str) -> str:
    """Normalize a single token to a likely exact column name."""
    txt = token.strip().strip("-•*`)\"").strip()
    txt = re.sub(r"\s*–\s*.*", "", txt)
    txt = re.sub(r"\s*—\s*.*", "", txt)
    txt = re.sub(r"\s*->\s*.*", "", txt)
    txt = txt.strip("\u2013\u2014")
    return txt.strip()


def _coalesce_tokens(line: str) -> list[str]:
    """Extract possible column names from a section line."""
    if not line:
        return []

    # Prefer anything before bullets/description markers.
    clean = re.split(r"[;|:]", line, maxsplit=1)[0]
    # split common separators
    parts = re.split(r"[,]|\band\b", clean)
    out: list[str] = []
    for p in parts:
        token = _norm_col_token(p)
        if token:
            out.append(token)
    return out


def _extract_section_lines(plan_text: str, section: str) -> list[str]:
    """Return cleaned lines for a numbered section from planner text."""
    if not plan_text:
        return []

    lines = [ln.strip() for ln in plan_text.splitlines() if ln.strip()]
    section_key = section
    start_idx: int | None = None
    stop_idx: int | None = None

    pat = _SECTION_PATTERNS.get(section_key)
    if not pat:
        return []

    # locate section header and the next section header
    header_indices: list[int] = [i for i, line in enumerate(lines) if pat.match(line)]
    if not header_indices:
        return []

    start_idx = header_indices[0] + 1
    for j in range(start_idx, len(lines)):
        if any(regex.match(lines[j]) for regex in _SECTION_PATTERNS.values()):
            stop_idx = j
            break
    stop_idx = stop_idx if stop_idx is not None else len(lines)
    return [ln for ln in lines[start_idx:stop_idx] if ln]


def _columns_from_plan(plan_text: str, section: str) -> list[str]:
    """Best-effort column extraction from one numbered section."""
    cols: list[str] = []
    for ln in _extract_section_lines(plan_text, section):
        for token in _coalesce_tokens(ln):
            if token and token not in cols:
                cols.append(token)
    return cols


def _dedupe(items: list[str]) -> list[str]:
    """Deduplicate while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — exact implementation-oriented scaling workflow
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = '''
You are an expert Python ML engineer specializing in feature scaling.
You receive a scaling plan and implement it by writing
and executing Python code.

You have access to one tool: execute_python(code: str) -> str
This tool runs Python code in a shared sandbox that already contains:
df_engineered: feature engineered pandas DataFrame — YOUR INPUT
pd: pandas
np: numpy
StandardScaler: from sklearn.preprocessing
RobustScaler: available via sklearn.preprocessing
MinMaxScaler: available via sklearn.preprocessing
train_test_split: from sklearn.model_selection
target_col: ML target column name (string)
numeric_cols: list of numeric column names
joblib: for saving scalers

Your workflow — follow this EXACTLY:

STEP 1: Setup and verify
execute_python("""
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import joblib, os, json

os.makedirs('outputs', exist_ok=True)

df_scaled = df_engineered.copy()
print('Input shape:', df_scaled.shape)
print('Target col:', target_col)
print('Columns:', list(df_scaled.columns))
""")

STEP 2: Train/Test Split FIRST
execute_python("""
X = df_scaled.drop(columns=[target_col])
y = df_scaled[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() <= 10 else None
)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print('Train class distribution:', y_train.value_counts().to_dict() if y.nunique() <= 10 else 'regression target')
""")

STEP 3: Apply scaling — fit on train, transform both
execute_python("""
scalers = {}

# StandardScaler columns from plan
standard_cols = [list from plan — exclude target, binary, one-hot cols]
if standard_cols:
    try:
        ss = StandardScaler()
        X_train[standard_cols] = ss.fit_transform(X_train[standard_cols])
        X_test[standard_cols] = ss.transform(X_test[standard_cols])
        scalers['standard'] = ss
        print('StandardScaler applied to:', standard_cols)
    except Exception as exc:
        print(f'ERROR on StandardScaler: {exc}. Skipping these columns.')

# RobustScaler columns from plan
robust_cols = [list from plan]
if robust_cols:
    try:
        rs = RobustScaler()
        X_train[robust_cols] = rs.fit_transform(X_train[robust_cols])
        X_test[robust_cols] = rs.transform(X_test[robust_cols])
        scalers['robust'] = rs
        print('RobustScaler applied to:', robust_cols)
    except Exception as exc:
        print(f'ERROR on RobustScaler: {exc}. Falling back to StandardScaler.')
        ss = StandardScaler()
        X_train[robust_cols] = ss.fit_transform(X_train[robust_cols])
        X_test[robust_cols] = ss.transform(X_test[robust_cols])
        scalers['robust_fallback_standard'] = ss
        print('Fallback StandardScaler applied to:', robust_cols)

# MinMaxScaler columns from plan
minmax_cols = [list from plan]
if minmax_cols:
    try:
        mm = MinMaxScaler()
        X_train[minmax_cols] = mm.fit_transform(X_train[minmax_cols])
        X_test[minmax_cols] = mm.transform(X_test[minmax_cols])
        scalers['minmax'] = mm
        print('MinMaxScaler applied to:', minmax_cols)
    except Exception as exc:
        print(f'ERROR on MinMaxScaler: {exc}. Falling back to StandardScaler.')
        try:
            ss = StandardScaler()
            X_train[minmax_cols] = ss.fit_transform(X_train[minmax_cols])
            X_test[minmax_cols] = ss.transform(X_test[minmax_cols])
            scalers['minmax_fallback_standard'] = ss
            print('Fallback StandardScaler applied to:', minmax_cols)
    except Exception as exc2:
        print(f'Fallback failed for minmax columns {minmax_cols}: {exc2}.')

# Columns skipped from scaling
skip_cols = [target_col] + [binary and one-hot columns from skip section, if any]
""")

STEP 4: Save scalers
execute_python("""
for name, scaler in scalers.items():
    path = f'outputs/scaler_{name}.pkl'
    joblib.dump(scaler, path)
    print(f'Saved {path}')
""")

STEP 5: Update sandbox and save outputs
execute_python("""
# Update sandbox for Model Training phase
globals()['X_train'] = X_train
globals()['X_test'] = X_test
globals()['y_train'] = y_train
globals()['y_test'] = y_test

# Save scaled datasets
X_train_save = X_train.copy()
X_train_save[target_col] = y_train.values
X_train_save.to_csv('outputs/train_data.csv', index=False)

X_test_save = X_test.copy()
X_test_save[target_col] = y_test.values
X_test_save.to_csv('outputs/test_data.csv', index=False)

scaling_summary = {
    'standard_scaled': standard_cols,
    'robust_scaled': robust_cols,
    'minmax_scaled': minmax_cols,
    'skipped': [target_col] + skip_cols,
    'train_shape': list(X_train.shape),
    'test_shape': list(X_test.shape),
    'scalers_saved': [f'outputs/scaler_{n}.pkl' for n in scalers.keys()]
}

import json
with open('outputs/scaling_summary.json', 'w') as f:
    json.dump(scaling_summary, f, indent=2)

print('Saved train_data.csv')
print('Saved test_data.csv')
print('Saved scaling_summary.json')
""")

STEP 6: Verify
execute_python("""
print('=== SCALING COMPLETE ===')
print('X_train range check:')
print(X_train.describe().loc[['min','max']].to_string())
print('Target col in X_train:', target_col in X_train.columns)
print('y_train unique values:', y_train.nunique())
""")

ERROR HANDLING RULES:
If execute_python returns ERROR — read it, fix exact issue, retry
Maximum 3 retries per step
If one scaler fails — skip it, log it, use StandardScaler as fallback
Never scale target_col under any circumstance
Never fit scaler on full dataset — always train split only

STRICT RULES:
Always split BEFORE scaling — never scale then split
Always fit on X_train only — transform X_test with same fitted scaler
Never modify df_engineered — work on df_scaled = df_engineered.copy()
Always save scalers as .pkl for deployment
globals() updates sandbox so Model Training gets X_train, X_test, y_train, y_test
stratify split for classification, regular split for regression
'''


# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────

scaling_implementor_agent = Agent(
    name="scaling_implementor_agent",
    model="gemini-2.0-flash",
    description=(
        "Implements an exact scaling plan: train/test split first, fit scalers "
        "on train only, persist transformed splits, scaler objects, and a "
        "scaling summary for downstream model training."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)

# Backward-compatible alias
scaling_executor_agent = scaling_implementor_agent


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helper — called by a scaling orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_scaling_implementor(
    records: list[dict[str, Any]],
    strategist_output: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the Scaling Implementor and return a compact execution payload.

    Args:
        records:           Engineered dataset rows.
        strategist_output: Dict from run_scaling_strategist() containing:
                           target_col, numeric_cols, encoded_cols,
                           transformed_cols, and scaling_plan.

    Returns:
        dict with output file paths and execution report for the next training step.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    target_col: str = strategist_output.get("target_col", "")
    numeric_cols: list[str] = strategist_output.get("numeric_cols", [])
    encoded_cols: list[str] = strategist_output.get("encoded_cols", [])
    transformed_cols: list[str] = strategist_output.get("transformed_cols", [])
    scaling_plan: str = strategist_output.get("scaling_plan", "")

    # Prefer explicit lists if orchestrator injected them.
    standard_cols: list[str] = strategist_output.get("standard_cols", [])
    robust_cols: list[str] = strategist_output.get("robust_cols", [])
    minmax_cols: list[str] = strategist_output.get("minmax_cols", [])
    skip_cols: list[str] = strategist_output.get("skip_cols", [])

    if not standard_cols:
        standard_cols = _columns_from_plan(str(scaling_plan), "standard")
    if not robust_cols:
        robust_cols = _columns_from_plan(str(scaling_plan), "robust")
    if not minmax_cols:
        minmax_cols = _columns_from_plan(str(scaling_plan), "minmax")
    if not skip_cols:
        skip_cols = _columns_from_plan(str(scaling_plan), "skip")

    numeric_set = set(numeric_cols)
    # Keep target and non-numeric columns out of all scaling lists.
    standard_cols = [
        c for c in _dedupe(standard_cols) if c and c != target_col and c in numeric_set
    ]
    robust_cols = [
        c for c in _dedupe(robust_cols) if c and c != target_col and c in numeric_set
    ]
    minmax_cols = [
        c for c in _dedupe(minmax_cols) if c and c != target_col and c in numeric_set
    ]
    skip_cols = [c for c in _dedupe(skip_cols) if c and c != target_col]

    # Remove any overlap: skip columns never scaled.
    skip_set = set(skip_cols)
    standard_cols = [c for c in standard_cols if c not in skip_set]
    robust_cols = [c for c in robust_cols if c not in skip_set]
    minmax_cols = [c for c in minmax_cols if c not in skip_set]

    # Init sandbox with raw recordset and standard tools.
    sandbox_init_msg = init_sandbox(records, target_col=target_col)

    bootstrap_msg = execute_python(
        "from sklearn.preprocessing import RobustScaler, MinMaxScaler\n"
        "from sklearn.model_selection import train_test_split\n"
        "import joblib, os, json\n"
        "os.makedirs('outputs', exist_ok=True)\n"
        "globals()['df_engineered'] = df.copy()\n"
        f"globals()['numeric_cols'] = {numeric_cols!r}\n"
        f"globals()['encoded_cols'] = {encoded_cols!r}\n"
        f"globals()['transformed_cols'] = {transformed_cols!r}\n"
        "print('Scaling bootstrap -> df_engineered shape:', df_engineered.shape)\n"
        "print('Columns:', list(df_engineered.columns))"
    )

    user_message = json.dumps(
        {
            "scaling_plan": scaling_plan,
            "target_col": target_col,
            "numeric_cols": numeric_cols,
            "encoded_cols": encoded_cols,
            "transformed_cols": transformed_cols,
            "standard_cols": standard_cols,
            "robust_cols": robust_cols,
            "minmax_cols": minmax_cols,
            "skip_cols": skip_cols,
            "info": (
                "Do not invent a scaling plan. Use provided lists exactly. "
                "Split before scaling. Fit scalers only on X_train and then "
                "transform X_test. Never scale target_col, save outputs to outputs/."
            ),
        },
        default=str,
    )

    runner = InMemoryRunner(agent=scaling_implementor_agent)
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
        "status": "completed",
        "target_col": target_col,
        "train_output": "outputs/train_data.csv",
        "test_output": "outputs/test_data.csv",
        "summary_output": "outputs/scaling_summary.json",
        "scaler_dir_prefix": "outputs/scaler_",
        "execution_log": execution_log.strip(),
        "sandbox_init": sandbox_init_msg,
        "bootstrap_status": bootstrap_msg,
        "verify_train": verify_output_saved("outputs/train_data.csv"),
        "verify_test": verify_output_saved("outputs/test_data.csv"),
        "verify_summary": verify_output_saved("outputs/scaling_summary.json"),
        "sandbox_info": get_sandbox_df_info(),
        "parsed_lists": {
            "standard_cols": standard_cols,
            "robust_cols": robust_cols,
            "minmax_cols": minmax_cols,
            "skip_cols": skip_cols,
        },
    }


# Backward-compatible alias
run_scaling_executor = run_scaling_implementor
