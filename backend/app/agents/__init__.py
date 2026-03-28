"""app.agents package — ADK agent definitions"""

from .master_orchestrator import run_full_pipeline, PipelineStageError
from .data_cleaning_orchestrator_agent import orchestrator_agent, run_pipeline
from .master_orchestrator import run_dc_orchestrator  # alias for run_pipeline
from .eda_orchestrator_agent import run_eda_pipeline
from .master_orchestrator import run_eda_orchestrator  # alias for run_eda_pipeline
from .scaling_orchestrator_agent import scaling_orchestrator_agent, run_scaling_orchestrator
from .mt_orchestrator_agent import mt_orchestrator_agent, run_mt_orchestrator, run_model_training_orchestrator
from .model_training_agent import (
    mt_analyzer_agent,
    run_mt_analyzer,
    mt_strategist_agent,
    run_mt_strategist,
    mt_implementor_agent,
    run_mt_implementor,
)
from .ms_orchestrator_agent import ms_orchestrator_agent, run_ms_orchestrator
from .model_selection_agent import (
    model_selection_analyzer_agent,
    run_model_selection_analyzer,
    model_selection_strategist_agent,
    run_model_selection_strategist,
    ms_executor_agent,
    run_ms_executor,
)
from .ht_orchestrator_agent import ht_orchestrator_agent, run_ht_orchestrator
from .hyperparameter_tuning_agent import (
    ht_analyzer_agent,
    run_ht_analyzer,
    ht_strategist_agent,
    run_ht_strategist,
    ht_executor_agent,
    run_ht_executor,
)
from .me_orchestrator_agent import me_orchestrator_agent, run_me_orchestrator
from .model_evaluation_agent import (
    me_analyzer_agent,
    run_me_analyzer,
    me_strategist_agent,
    run_me_strategist,
    me_executor_agent,
    run_me_executor,
)
from .fo_orchestrator_agent import fo_orchestrator_agent, run_fo_orchestrator
from .final_output_agent import fo_executor_agent, run_fo_executor

__all__ = [
    # Master pipeline
    "run_full_pipeline",
    "PipelineStageError",
    # Stage 1 — Data Cleaning
    "orchestrator_agent",
    "run_pipeline",
    "run_dc_orchestrator",
    # Stage 2 — EDA
    "run_eda_pipeline",
    "run_eda_orchestrator",
    # Stage 4 — Feature Scaling
    "scaling_orchestrator_agent",
    "run_scaling_orchestrator",
    # Stage 6 — Model Training
    "mt_orchestrator_agent",
    "run_mt_orchestrator",
    "run_model_training_orchestrator",
    "mt_analyzer_agent",
    "run_mt_analyzer",
    "mt_strategist_agent",
    "run_mt_strategist",
    "mt_implementor_agent",
    "run_mt_implementor",
    # Stage 7 — Model Selection
    "ms_orchestrator_agent",
    "run_ms_orchestrator",
    "model_selection_analyzer_agent",
    "run_model_selection_analyzer",
    "model_selection_strategist_agent",
    "run_model_selection_strategist",
    "ms_executor_agent",
    "run_ms_executor",
    # Stage 8 — Hyperparameter Tuning
    "ht_orchestrator_agent",
    "run_ht_orchestrator",
    "ht_analyzer_agent",
    "run_ht_analyzer",
    "ht_strategist_agent",
    "run_ht_strategist",
    "ht_executor_agent",
    "run_ht_executor",
    # Stage 9 — Model Evaluation
    "me_orchestrator_agent",
    "run_me_orchestrator",
    "me_analyzer_agent",
    "run_me_analyzer",
    "me_strategist_agent",
    "run_me_strategist",
    "me_executor_agent",
    "run_me_executor",
    # Stage 10 — Final Output
    "fo_orchestrator_agent",
    "run_fo_orchestrator",
    "fo_executor_agent",
    "run_fo_executor",
]
