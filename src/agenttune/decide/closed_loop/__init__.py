"""Closed-loop self-healing (Path A detect/learn + Path B decide/act).

Lazy on purpose (see ``agenttune/decide/__init__.py`` for the same pattern): most of
this package (``contracts``, ``deployment_gate``, ``retraining_trigger``, ``retrain_runner``,
``tool_isolation``) is litellm-free by design, but ``full_loop`` and ``closed_loop_runner``
pull in ``failure_classifier``/``training_example_generator`` (litellm-dependent). Eagerly
importing everything here would force litellm onto any caller reaching for a single
litellm-free name, e.g. ``agenttune.decide.closed_loop.contracts`` from ``heal_loop.py``.
``__getattr__`` (PEP 562) defers each name to first access; behavior and names are unchanged.
"""

import importlib

__all__ = [
    # contracts
    "Failure",
    "ClassifiedFailure",
    "TrainingExample",
    "BufferHealth",
    "AgenticEvalResult",
    # detection (Path A)
    "FailureDetector",
    # retraining trigger (Path B)
    "TriggerConfig",
    "TrainingBuffer",
    "RewardDriftTracker",
    "RetrainingTrigger",
    # retrain config + background runner (Path B, Week 2)
    "RetrainConfig",
    "build_retrain_config",
    "build_retrainer",
    "run_retrain",
    "examples_to_dpo_dataset",
    "examples_to_bco_dataset",
    "BackgroundRetrainer",
    "RetrainResult",
    # deployment gate (Path B)
    "DeploymentGate",
    "ABComparison",
    "GateDecision",
    # closed-loop wiring + tool isolation (Path B, Week 3)
    "ClosedLoopRunner",
    "CycleRecord",
    "IsolatedTool",
    "isolate_tools",
    # full closed loop (Path B, Week 4 — the wiring)
    "FullClosedLoop",
    "PathAConfig",
    "GateConfig",
    "LoopCycle",
]

_LAZY = {
    "Failure": (".contracts", "Failure"),
    "ClassifiedFailure": (".contracts", "ClassifiedFailure"),
    "TrainingExample": (".contracts", "TrainingExample"),
    "BufferHealth": (".contracts", "BufferHealth"),
    "AgenticEvalResult": (".contracts", "AgenticEvalResult"),
    "FailureDetector": (".failure_detector", "FailureDetector"),
    "TriggerConfig": (".retraining_trigger", "TriggerConfig"),
    "TrainingBuffer": (".retraining_trigger", "TrainingBuffer"),
    "RewardDriftTracker": (".retraining_trigger", "RewardDriftTracker"),
    "RetrainingTrigger": (".retraining_trigger", "RetrainingTrigger"),
    "RetrainConfig": (".retrain_config", "RetrainConfig"),
    "build_retrain_config": (".retrain_config", "build_retrain_config"),
    "build_retrainer": (".retrain_config", "build_retrainer"),
    "run_retrain": (".retrain_config", "run_retrain"),
    "examples_to_dpo_dataset": (".retrain_config", "examples_to_dpo_dataset"),
    "examples_to_bco_dataset": (".retrain_config", "examples_to_bco_dataset"),
    "BackgroundRetrainer": (".retrain_runner", "BackgroundRetrainer"),
    "RetrainResult": (".retrain_runner", "RetrainResult"),
    "DeploymentGate": (".deployment_gate", "DeploymentGate"),
    "ABComparison": (".deployment_gate", "ABComparison"),
    "GateDecision": (".deployment_gate", "GateDecision"),
    "ClosedLoopRunner": (".closed_loop_runner", "ClosedLoopRunner"),
    "CycleRecord": (".closed_loop_runner", "CycleRecord"),
    "IsolatedTool": (".tool_isolation", "IsolatedTool"),
    "isolate_tools": (".tool_isolation", "isolate_tools"),
    "FullClosedLoop": (".full_loop", "FullClosedLoop"),
    "PathAConfig": (".full_loop", "PathAConfig"),
    "GateConfig": (".full_loop", "GateConfig"),
    "LoopCycle": (".full_loop", "LoopCycle"),
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(importlib.import_module(module_name, __name__), attr_name)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value
