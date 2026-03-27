# AutoML Agent — Build Steps

> Hackathon build guide. Follow in order. Each step = one working piece.

---

## Setup (Everyone — 0:00 to 0:30)

```bash
mkdir automl-agent && cd automl-agent
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install google-adk google-generativeai pandas numpy scikit-learn xgboost matplotlib plotly reportlab streamlit joblib
```

Create `.env` file:
```
GOOGLE_API_KEY=your_gemini_key_here
```

Project structure:
```
automl-agent/
├── agent.py          # P1 — 3-agent pattern + execute_python
├── pipeline.py       # P2 — all 8 pipeline steps
├── outputs.py        # P2 — save csv, pkl, pdf
├── app.py            # P3 — Streamlit UI
├── demo.csv          # P3 — sample dataset
└── .env
```

---

## P1 — Agent Framework (0:30 to 1:00)

**File: `agent.py`**

### Step 1: Shared Sandbox
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib, io, contextlib, os

# Shared state — persists across all agent code executions
sandbox = {
    "pd": pd, "np": np,
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "XGBClassifier": XGBClassifier,
    "LabelEncoder": LabelEncoder,
    "StandardScaler": StandardScaler,
    "train_test_split": train_test_split,
    "classification_report": classification_report,
    "confusion_matrix": confusion_matrix,
    "plt": plt,
    "joblib": joblib,
    "df": None,        # populated after file load
    "model": None,     # populated after training
    "results": {}      # stores step outputs
}
```

### Step 2: Execute Python Tool
```python
def execute_python(code: str) -> str:
    """Agent writes and runs Python code. Returns stdout or error."""
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, sandbox)
        result = output.getvalue()
        return result if result else "Executed successfully."
    except Exception as e:
        return f"ERROR: {str(e)}"
        # Agent reads this error and rewrites code automatically
```

### Step 3: 3-Agent Pattern Function
```python
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def run_three_agents(step_name: str, data_context: str) -> dict:
    """
    Runs 3 agents for one pipeline step.
    Returns: { analyzer, planner, code, output, explanation }
    """
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Agent 1 — Analyzer
    analyzer_prompt = f"""
    You are a data analyst. Analyze this data context and identify issues/patterns.
    Step: {step_name}
    Data Context: {data_context}
    Be specific. List exact issues found with numbers where possible.
    """
    analysis = model.generate_content(analyzer_prompt).text

    # Agent 2 — Planner
    planner_prompt = f"""
    You are a senior data scientist. Given this analysis, plan the exact steps to take.
    Step: {step_name}
    Analysis: {analysis}
    Output a numbered list of specific actions to take. Be precise.
    """
    plan = model.generate_content(planner_prompt).text

    # Agent 3 — Coder
    coder_prompt = f"""
    You are a Python expert. Write executable Python code to implement this plan.
    Step: {step_name}
    Plan: {plan}

    Rules:
    - Use variables already in sandbox: df, pd, np, sklearn classes, plt, joblib
    - Always print results so output is visible
    - Save any charts as PNG files to 'charts/' folder
    - If cleaning: save result as df (update sandbox df)
    - Keep code concise and correct
    - Return ONLY the Python code, no explanation, no markdown fences
    """
    code = model.generate_content(coder_prompt).text
    code = code.replace("```python", "").replace("```", "").strip()

    # Execute with auto-retry
    output = ""
    for attempt in range(3):
        output = execute_python(code)
        if not output.startswith("ERROR"):
            break
        # Fix code on error
        fix_prompt = f"Fix this Python code. Error: {output}\nCode:\n{code}\nReturn only fixed code."
        code = model.generate_content(fix_prompt).text.replace("```python","").replace("```","").strip()

    # Explanation
    explain_prompt = f"""
    In 2-3 sentences of plain English, explain what was done in this step and why.
    Step: {step_name}, Plan: {plan}, Output: {output}
    """
    explanation = model.generate_content(explain_prompt).text

    return {
        "step": step_name,
        "analysis": analysis,
        "plan": plan,
        "code": code,
        "output": output,
        "explanation": explanation
    }
```

---

## P2 — Pipeline Steps (1:00 to 2:00)

**File: `pipeline.py`**

```python
from agent import run_three_agents, sandbox
import pandas as pd
import os

os.makedirs("charts", exist_ok=True)

def get_data_context():
    """Get current df state as string for agents."""
    df = sandbox["df"]
    if df is None:
        return "No data loaded"
    return f"""
    Shape: {df.shape}
    Columns: {list(df.columns)}
    Dtypes: {df.dtypes.to_dict()}
    Nulls: {df.isnull().sum().to_dict()}
    Sample: {df.head(3).to_dict()}
    """

def step_load(file_path: str, target_col: str):
    code = f"""
df = pd.read_csv('{file_path}')
print("Loaded:", df.shape)
print(df.dtypes)
print(df.head())
"""
    from agent import execute_python
    output = execute_python(code)
    sandbox["target_col"] = target_col
    return {"step": "Load Data", "output": output, "explanation": f"Loaded dataset with {sandbox['df'].shape[0]} rows and {sandbox['df'].shape[1]} columns."}

def step_clean():
    context = get_data_context()
    return run_three_agents("Data Cleaning", context)

def step_eda():
    context = get_data_context()
    return run_three_agents("Exploratory Data Analysis", context)

def step_feature_engineering():
    context = get_data_context()
    return run_three_agents("Feature Engineering and Encoding", context)

def step_train():
    target = sandbox.get("target_col", "target")
    context = get_data_context() + f"\nTarget column: {target}"
    return run_three_agents("Model Training - train RF, LR, XGB and compare accuracy", context)

def step_evaluate():
    context = f"Best model trained. Target: {sandbox.get('target_col')}. Evaluate it."
    return run_three_agents("Model Evaluation - confusion matrix, classification report", context)

def run_full_pipeline(file_path: str, target_col: str):
    """Run all steps, yield results for Streamlit streaming."""
    steps = [
        lambda: step_load(file_path, target_col),
        step_clean,
        step_eda,
        step_feature_engineering,
        step_train,
        step_evaluate,
    ]
    results = []
    for step_fn in steps:
        result = step_fn()
        results.append(result)
        yield result   # stream to Streamlit live
    return results
```

**File: `outputs.py`**
```python
from agent import sandbox
import joblib, pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import glob, os

def save_cleaned_csv():
    df = sandbox["df"]
    df.to_csv("output/cleaned_data.csv", index=False)
    return "output/cleaned_data.csv"

def save_model():
    model = sandbox.get("model")
    if model:
        joblib.dump(model, "output/final_model.pkl")
    return "output/final_model.pkl"

def save_report(step_results: list):
    os.makedirs("output", exist_ok=True)
    doc = SimpleDocTemplate("output/report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AutoML Agent — Pipeline Report", styles["Title"]))
    story.append(Spacer(1, 20))

    for result in step_results:
        story.append(Paragraph(result["step"], styles["Heading2"]))
        story.append(Paragraph(result.get("explanation", ""), styles["Normal"]))
        story.append(Spacer(1, 12))

    # Add charts
    for chart_path in glob.glob("charts/*.png"):
        story.append(Image(chart_path, width=400, height=250))
        story.append(Spacer(1, 12))

    doc.build(story)
    return "output/report.pdf"
```

---

## P3 — Streamlit UI (2:00 to 2:15)

**File: `app.py`**

```python
import streamlit as st
from pipeline import run_full_pipeline
from outputs import save_cleaned_csv, save_model, save_report
import os

os.makedirs("output", exist_ok=True)

st.set_page_config(page_title="AutoML Agent", page_icon="🤖", layout="wide")
st.title("🤖 AutoML Agent")
st.caption("Upload a CSV — agent runs the full ML pipeline autonomously")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    target_col = st.text_input("Target column name", placeholder="e.g. churn, price, label")

with col2:
    st.info("**What this does:**\n\n1. Cleans your data\n2. Runs EDA\n3. Engineers features\n4. Trains 3 models\n5. Evaluates best model\n6. Gives you files to download")

if uploaded_file and target_col:
    if st.button("🚀 Run Pipeline", type="primary"):
        # Save uploaded file
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.divider()
        st.subheader("Pipeline Running...")

        all_results = []
        for result in run_full_pipeline(file_path, target_col):
            with st.expander(f"✅ {result['step']}", expanded=True):
                st.markdown(f"**What happened:** {result.get('explanation', '')}")
                if result.get("output"):
                    st.code(result["output"], language="text")
            all_results.append(result)

        # Save outputs
        st.divider()
        st.subheader("📦 Your Files are Ready")

        csv_path = save_cleaned_csv()
        pkl_path = save_model()
        pdf_path = save_report(all_results)

        col1, col2, col3 = st.columns(3)
        with col1:
            with open(csv_path, "rb") as f:
                st.download_button("⬇️ cleaned_data.csv", f, file_name="cleaned_data.csv", mime="text/csv")
        with col2:
            with open(pkl_path, "rb") as f:
                st.download_button("⬇️ final_model.pkl", f, file_name="final_model.pkl")
        with col3:
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ report.pdf", f, file_name="report.pdf", mime="application/pdf")
```

Run it:
```bash
streamlit run app.py
```

---

## Demo Dataset (P3)

Save as `demo.csv` — use Titanic or any Kaggle dataset with:
- Mix of numeric and categorical columns
- Some null values (so cleaning is visible)
- Binary target column (0/1)

Good options:
- Titanic survival (target: `Survived`)
- Bank churn (target: `Exited`)
- Heart disease (target: `target`)

Download from: https://www.kaggle.com/datasets

---

## Run Order Checklist

- [ ] `.env` file with `GOOGLE_API_KEY`
- [ ] `pip install` all dependencies
- [ ] `agent.py` — sandbox + execute_python + run_three_agents working
- [ ] Test: run one step manually in Python REPL
- [ ] `pipeline.py` — all 6 step functions working
- [ ] `outputs.py` — 3 files saving correctly
- [ ] `app.py` — UI shows steps live, downloads work
- [ ] Load `demo.csv`, run full pipeline end to end
- [ ] Time it — should be under 3 minutes
- [ ] Rehearse demo pitch (2 minutes max)

---

## Pitch Script (2 minutes)

> "Every data scientist follows the same pipeline — clean data, explore it, engineer features, train models, evaluate. It takes days.
>
> We built an agent that does it in minutes. Upload any CSV, tell it your target column — three specialized AI agents collaborate at every step: one analyzes the data, one plans the approach, one writes and runs the code.
>
> At the end you get three files: your cleaned dataset, a trained model ready to deploy, and a full report explaining every decision the agent made — like having a senior data scientist on demand.
>
> [DEMO: upload CSV, watch steps stream live, click download]"

---

*Built at Hackathon 2026 | Stack: Google ADK + Gemini 2.0 Flash + Streamlit*
