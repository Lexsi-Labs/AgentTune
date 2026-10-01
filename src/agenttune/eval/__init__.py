"""
Evaluation system for AgentTune.

This module provides a unified interface for evaluating models:
1. Universal Evaluator (New): Modular, backend-agnostic evaluation for SFT/RL.
2. Legacy Evaluator (Old): Compatible with existing CLI and EvalRunner workflows.
"""

# --- New Universal Framework Exports ---
# Registry (Merged logic)
from . import registry
from .agent_eval import run_eval

# --- Legacy Framework Exports (Backward Compatibility) ---
# Assumes the old 'core.py' is preserved in the directory
from .core import (
    EvalConfig,
    EvalLogger,
    EvalRegistry,
    EvalResult,
    EvalRunner,
    EvalTask,
    EvalType,
    TaskCategory,
)
from .evaluator import BaseEvaluator
from .lm_eval_integration import (
    LMEVAL_TASKS,
    LMEvalConfig,
    LMEvalRunner,
    LMEvalTask,
    get_available_lm_eval_tasks,
    get_lm_eval_task,
    run_standard_benchmark,
)
from .metrics.base import Metric
from .metrics.code import PassAtKMetric
from .metrics.generic import AccuracyMetric, PerplexityMetric
from .metrics.math import MathAccuracyMetric
from .metrics.rl import KLDivergenceMetric, PolicyEntropyMetric, RewardAccuracyMetric
from .metrics.text import BleuMetric, RougeMetric
from .rl_evaluator import RLEvaluator

__all__ = [
    # New Universal Classes
    "BaseEvaluator",
    "RLEvaluator",
    "Metric",
    "PerplexityMetric",
    "AccuracyMetric",
    "BleuMetric",
    "RougeMetric",
    "KLDivergenceMetric",
    "RewardAccuracyMetric",
    "PolicyEntropyMetric",
    "MathAccuracyMetric",
    "PassAtKMetric",
    "run_eval",
    # Legacy Core Classes
    "EvalType",
    "TaskCategory",
    "EvalConfig",
    "EvalTask",
    "EvalResult",
    "EvalLogger",
    "EvalRegistry",
    "EvalRunner",
    # lm-eval Integration
    "LMEvalConfig",
    "LMEvalTask",
    "LMEvalRunner",
    "LMEVAL_TASKS",
    "get_available_lm_eval_tasks",
    "get_lm_eval_task",
    "run_standard_benchmark",
    # Registry
    "registry",
]
