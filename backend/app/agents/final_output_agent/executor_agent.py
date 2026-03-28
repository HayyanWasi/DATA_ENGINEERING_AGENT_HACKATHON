"""
app/agents/final_output_agent/executor_agent.py

Final Output Executor Agent — single agent in the FO pipeline.

Input:  orchestrator_input dict containing me_result, mt_result,
        ht_result, dataset_id, target_col, eda_charts,
        elapsed_seconds, dataset_stats.
Tools:  execute_python, verify_output_saved
Output: outputs/results_manifest.json  +  outputs/final_model.joblib

4-step workflow:
  STEP 1 — Verify output files exist
  STEP 2 — Save final model as joblib
  STEP 3 — Build results_manifest.json
  STEP 4 — Verify manifest
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
You receive a JSON message containing all pipeline stage results and build the
final outputs needed for the frontend.

You have access to execute_python(code: str) -> str which runs code in a shared
sandbox. The sandbox already contains final_model, pd, np, and all training data.

YOUR WORKFLOW — follow these steps in exact order:

STEP 1: Verify output files
Write and run code to:
  from app.tools.fo_tools import collect_output_files
  files = collect_output_files()
  print("Present files:", files["present"])
  print("Missing files:", files["missing"])
  print("Stages complete:", files["stages_complete"])

STEP 2: Save model as joblib
Write and run code to:
  from app.tools.fo_tools import save_model_as_joblib
  result = save_model_as_joblib()
  print("Model save result:", result)
  if not result["success"]:
      print("WARNING: model save failed:", result.get("error"))

STEP 3: Build results_manifest.json
Extract me_result, mt_result, ht_result, dataset_id, target_col,
eda_charts, elapsed_seconds, dataset_stats from the input JSON message.
Write and run code to:
  import json
  from app.tools.fo_tools import get_pipeline_summary, build_results_manifest

  # Deserialize the inputs from the message (use exact values from input JSON)
  me_result = <paste_me_result_json>
  mt_result = <paste_mt_result_json>
  ht_result = <paste_ht_result_json>

  pipeline_summary = get_pipeline_summary(me_result, mt_result, ht_result)
  manifest = build_results_manifest(
      dataset_id=<dataset_id>,
      target_col=<target_col>,
      pipeline_summary=pipeline_summary,
      eda_charts=<eda_charts>,
      elapsed_seconds=<elapsed_seconds>,
      dataset_stats=<dataset_stats>,
  )
  print("Manifest model_used:", manifest["pipeline_summary"]["model_used"])
  print("Manifest accuracy:", manifest["pipeline_summary"]["accuracy"])
  print("Downloads:", [d["file_type"] for d in manifest["downloads"]])

STEP 4: Verify manifest
Write and run code to:
  import os, json
  path = "outputs/results_manifest.json"
  if os.path.isfile(path):
      with open(path) as f:
          m = json.load(f)
      print("PASS: results_manifest.json exists")
      print("  Keys:", list(m.keys()))
      print("  Model used:", m.get("pipeline_summary", {}).get("model_used", "?"))
  else:
      print("FAIL: results_manifest.json not found")

ERROR HANDLING RULES:
If collect_output_files shows 0 stages complete, continue anyway.
If save_model_as_joblib fails, log warning but continue to Step 3.
If get_pipeline_summary returns empty model_used, use "Unknown Model".
Always complete Step 3 (build_results_manifest) — the frontend depends on it.
Maximum 2 retries per step.

STRICT RULES:
Always complete all 4 steps in order.
Always save outputs/results_manifest.json.
Print results after every step.
"""


fo_executor_agent = Agent(
    name="fo_executor_agent",
    model="gemini-2.0-flash",
    description=(
        "Verifies pipeline output files, saves the final model as joblib, "
        "and builds outputs/results_manifest.json for the Next.js frontend."
    ),
    instruction=_SYSTEM_PROMPT,
    tools=_TOOLS,
)


async def run_fo_executor(orchestrator_input: dict[str, Any]) -> dict[str, Any]:
    """
    Run the Final Output Executor agent end-to-end.

    Args:
        orchestrator_input: dict with keys:
            me_result, mt_result, ht_result — stage outputs (serializable dicts)
            dataset_id    — UUID string
            target_col    — ML target column name
            eda_charts    — list of chart file paths
            elapsed_seconds — total pipeline runtime
            dataset_stats — {"rows": int, "features": int}

    Returns:
        {
            "status": "completed",
            "execution_log": str,
            "verify_manifest": str,
        }
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    user_message = json.dumps(
        {
            "me_result":        orchestrator_input.get("me_result", {}),
            "mt_result":        orchestrator_input.get("mt_result", {}),
            "ht_result":        orchestrator_input.get("ht_result", {}),
            "dataset_id":       orchestrator_input.get("dataset_id", ""),
            "target_col":       orchestrator_input.get("target_col", ""),
            "eda_charts":       orchestrator_input.get("eda_charts", []),
            "elapsed_seconds":  orchestrator_input.get("elapsed_seconds", 0.0),
            "dataset_stats":    orchestrator_input.get("dataset_stats", {}),
            "instructions": (
                "Follow the 4-step workflow exactly. "
                "Parse me_result, mt_result, ht_result from this JSON message "
                "and use them to call get_pipeline_summary and build_results_manifest. "
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
        "status": "completed",
        "execution_log": execution_log.strip(),
        "verify_manifest": verify_output_saved("outputs/results_manifest.json"),
    }
