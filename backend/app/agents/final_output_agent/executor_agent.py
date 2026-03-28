"""
app/agents/final_output_agent/executor_agent.py

Final Output Executor Agent — single agent in the FO pipeline.

Input:  orchestrator_input dict containing dataset_id (and optional
        me_result, mt_result, ht_result for backward compatibility).
Tools:  execute_python, verify_output_saved
Output: outputs/final_model.joblib  +  outputs/results_manifest.json

4-step workflow:
  STEP 1 — Verify output files exist (FOUND / MISSING)
  STEP 2 — Save final model as joblib (raise on None → STOP)
  STEP 3 — Build results_manifest.json (with eda_section + model_evaluation_section)
  STEP 4 — Final verification (PASS / FAIL + size)
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from app.tools.executor_tools import execute_python, verify_output_saved


# ─────────────────────────────────────────────────────────────────────────────
# Tool registration
# ─────────────────────────────────────────────────────────────────────────────

_TOOLS: list[FunctionTool] = [
    FunctionTool(execute_python),
    FunctionTool(verify_output_saved),
]


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert Python ML engineer specializing in finalizing pipeline outputs.
You receive a JSON message containing the dataset_id and run the 4-step
final output workflow below.

You have access to execute_python(code: str) -> str which runs code in a
shared sandbox that already contains:
  final_model      — trained model object (may be None if training failed)
  final_model_name — model name string
  target_col       — ML target column name
  pd, np           — pandas and numpy
  joblib           — joblib module

YOUR WORKFLOW — follow these steps in exact order:

STEP 1: Verify output files
Write and run code to check existence of these files and print FOUND or MISSING
for each one:
  outputs/cleaned_data.csv
  outputs/evaluation_summary.json
  outputs/confusion_matrix.png
  outputs/model_comparison.png
  outputs/feature_importance.png
  outputs/roc_curve.png
  outputs/pr_curve.png
  charts/correlation_heatmap.png
  charts/target_distribution.png

Example output format:
  outputs/cleaned_data.csv          — FOUND
  outputs/confusion_matrix.png      — MISSING

Use this code pattern:
  import os
  checks = [
      "outputs/cleaned_data.csv",
      "outputs/evaluation_summary.json",
      "outputs/confusion_matrix.png",
      "outputs/model_comparison.png",
      "outputs/feature_importance.png",
      "outputs/roc_curve.png",
      "outputs/pr_curve.png",
      "charts/correlation_heatmap.png",
      "charts/target_distribution.png",
  ]
  for path in checks:
      status = "FOUND" if os.path.isfile(path) else "MISSING"
      print(f"  {path:<45} — {status}")

STEP 2: Save model as joblib
Write and run code to:
  from app.tools.fo_tools import save_model_as_joblib
  try:
      path = save_model_as_joblib()
      size = os.path.getsize(path)
      print(f"Model saved: {path}  ({size:,} bytes)")
  except Exception as exc:
      print(f"STOP: model save failed — {exc}")
      raise  # re-raise to halt pipeline if model is None

CRITICAL: If save_model_as_joblib() raises (model is None), stop here.
Do NOT proceed to Step 3 or Step 4.
Retry this step at most 3 times before stopping.

STEP 3: Build results_manifest.json
Write and run code to:

  import os, json, glob
  from app.tools.executor_tools import _SANDBOX
  from app.tools.fo_tools import get_pipeline_summary, build_results_manifest

  # Read dataset_id from the input message (inject from user JSON)
  dataset_id = "<DATASET_ID_FROM_MESSAGE>"

  # Build pipeline summary from sandbox + JSON files on disk
  pipeline_summary = get_pipeline_summary(_SANDBOX)

  # Dynamically discover charts — only include files that exist
  eda_charts = []
  for pattern in ("charts/distribution_*.png", "charts/comparison_*.png",
                  "charts/*.png"):
      for p in sorted(glob.glob(pattern)):
          if p not in eda_charts:
              eda_charts.append(p)

  eval_charts = []
  for pattern in ("outputs/distribution_*.png", "outputs/comparison_*.png",
                  "outputs/*.png"):
      for p in sorted(glob.glob(pattern)):
          if p not in eval_charts:
              eval_charts.append(p)

  # Build base manifest (writes outputs/results_manifest.json)
  manifest = build_results_manifest(
      dataset_id=dataset_id,
      pipeline_summary=pipeline_summary,
  )

  # Extend with eda_section and model_evaluation_section
  manifest["eda_section"] = {
      "charts": eda_charts,
      "chart_count": len(eda_charts),
  }
  manifest["model_evaluation_section"] = {
      "charts": eval_charts,
      "chart_count": len(eval_charts),
      "metrics": pipeline_summary.get("metrics", {}),
      "model_comparison": pipeline_summary.get("model_comparison", []),
      "performance_rating": pipeline_summary.get("model_info", {}).get("performance_rating", "Unknown"),
  }

  # Overwrite manifest with extended version
  with open("outputs/results_manifest.json", "w", encoding="utf-8") as f:
      json.dump(manifest, f, indent=2, default=str)

  print("Manifest model_used:", pipeline_summary.get("model_info", {}).get("final_model_name", "?"))
  print("Manifest accuracy:",   pipeline_summary.get("metrics", {}).get("accuracy", 0.0))
  print("EDA charts found:",    len(eda_charts))
  print("Eval charts found:",   len(eval_charts))
  print("Downloads:",           [d["file_type"] for d in manifest.get("downloads", [])])

Replace "<DATASET_ID_FROM_MESSAGE>" with the actual dataset_id value from
the JSON input message you received.

Retry this step at most 3 times if it raises an exception.
Always save outputs/results_manifest.json — the frontend depends on it.

STEP 4: Final verification
Write and run code to verify these two files and print PASS or FAIL + size:
  outputs/final_model.joblib
  outputs/results_manifest.json

Use this code pattern:
  import os
  for path in ["outputs/final_model.joblib", "outputs/results_manifest.json"]:
      if os.path.isfile(path):
          size = os.path.getsize(path)
          print(f"  PASS  {path}  ({size:,} bytes)")
      else:
          print(f"  FAIL  {path}  — NOT FOUND")

ERROR HANDLING RULES:
If save_model_as_joblib() raises (model is None) → STOP immediately, do NOT
  proceed to Steps 3 or 4.
If get_pipeline_summary or build_results_manifest raise → log the error,
  retry up to 3 times, then write a minimal manifest and continue.
Always attempt to save outputs/results_manifest.json — even if pipeline_summary
  is empty, write {"dataset_id": dataset_id, "error": "pipeline_summary_failed"}.
Never let a chart discovery failure block the manifest from being saved.

STRICT RULES:
Always run all 4 steps in order (unless STOP condition hit in Step 2).
Always print step headers so the log is readable.
Print results after every step.
Maximum 3 retries per step.
"""


fo_executor_agent = Agent(
    name="fo_executor_agent",
    model="gemini-2.0-flash",
    description=(
        "Verifies pipeline output files (FOUND/MISSING), saves the final model "
        "as joblib, builds outputs/results_manifest.json with pipeline_summary, "
        "eda_section, model_evaluation_section, and downloads, then verifies "
        "both outputs (PASS/FAIL + size)."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)


async def run_fo_executor(orchestrator_input: dict[str, Any]) -> dict[str, Any]:
    """
    Run the Final Output Executor agent end-to-end.

    Args:
        orchestrator_input: dict with keys:
            dataset_id       — UUID string
            me_result        — ME stage output (kept for backward compatibility)
            mt_result        — MT stage output (kept for backward compatibility)
            ht_result        — HT stage output (kept for backward compatibility)
            target_col       — ML target column name
            eda_charts       — list of EDA chart file paths
            elapsed_seconds  — total pipeline runtime
            dataset_stats    — {"rows": int, "features": int}

    Returns:
        {
            "status":           "completed",
            "execution_log":    str,
            "verify_manifest":  dict,
            "verify_model":     dict,
        }
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    dataset_id: str = orchestrator_input.get("dataset_id", "")

    user_message = json.dumps(
        {
            "dataset_id":      dataset_id,
            "target_col":      orchestrator_input.get("target_col", ""),
            "elapsed_seconds": orchestrator_input.get("elapsed_seconds", 0.0),
            "dataset_stats":   orchestrator_input.get("dataset_stats", {}),
            "instructions": (
                "Follow the 4-step workflow exactly. "
                f"Use dataset_id='{dataset_id}' when calling build_results_manifest "
                "and when constructing download URLs. "
                "Read all pipeline data from JSON files on disk via get_pipeline_summary(_SANDBOX). "
                "If model is None (save_model_as_joblib raises), STOP immediately. "
                "Always save outputs/results_manifest.json."
            ),
        },
        default=str,
    )

    runner = InMemoryRunner(agent=fo_executor_agent)
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
        "status":          "completed",
        "execution_log":   execution_log.strip(),
        "verify_manifest": verify_output_saved("outputs/results_manifest.json"),
        "verify_model":    verify_output_saved("outputs/final_model.joblib"),
    }
