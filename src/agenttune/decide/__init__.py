"""AgentTune Decide — YAML-driven decision orchestration framework.

Lazy on purpose: some submodules (``closed_loop.contracts``, ``closed_loop.heal_loop``
consumers, etc.) are designed to be litellm/langgraph-free at import time, but Python
always executes a package's ``__init__.py`` before any of its submodules. Eagerly
importing ``GraphRunner`` etc. here would defeat that for every ``agenttune.decide.*``
import, not just the top-level names. ``__getattr__`` (PEP 562) defers each name to
first access instead, so ``import agenttune.decide.closed_loop.contracts`` stays light
while ``from agenttune.decide import GraphRunner`` still works exactly as before.
"""

import importlib

__all__ = [
    # Core
    "GraphRunner",
    "TemplateRegistry",
    "PipelineState",
    "EvalRunner",
    "CollectRunner",
    # AgentTune bridges
    "DecideToTrainerBridge",
    "train_from_audit",
    "ModelDeploymentBridge",
    "deploy_trained_model",
]

_LAZY = {
    "GraphRunner": ("agenttune.decide.graph_runner", "GraphRunner"),
    "TemplateRegistry": ("agenttune.decide.registry", "TemplateRegistry"),
    "PipelineState": ("agenttune.decide.state", "PipelineState"),
    "EvalRunner": ("agenttune.decide.eval_runner", "EvalRunner"),
    "CollectRunner": ("agenttune.decide.collect_runner", "CollectRunner"),
    "DecideToTrainerBridge": ("agenttune.decide.training_bridge", "DecideToTrainerBridge"),
    "train_from_audit": ("agenttune.decide.training_bridge", "train_from_audit"),
    "ModelDeploymentBridge": ("agenttune.decide.model_deployment", "ModelDeploymentBridge"),
    "deploy_trained_model": ("agenttune.decide.model_deployment", "deploy_trained_model"),
}


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value
