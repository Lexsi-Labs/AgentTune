"""AgentTune: a framework for training tool-using LLM agents with RL.

The stable, documented entry points live in :mod:`agenttune.api`:

    from agenttune import run_pipeline, train_agentic

The one-call training factories are also importable directly from the top
level (lazily — importing ``agenttune`` itself stays cheap; the underlying
TRL/torch training stack only loads once you actually access one of these):

    from agenttune import create_agentic_trainer, create_rag_trainer, create_distill_trainer
"""

import importlib
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agenttune")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

from agenttune.api import (  # noqa: E402
    PipelineResult,
    arun_pipeline,
    run_pipeline,
    train_agentic,
)

__all__ = [
    "__version__",
    "PipelineResult",
    "run_pipeline",
    "arun_pipeline",
    "train_agentic",
    "create_agentic_trainer",
    "create_rag_trainer",
    "create_distill_trainer",
]

# name -> (module to import, attribute on that module)
_LAZY_ATTRS = {
    "create_agentic_trainer": ("agenttune.core.backend_factory", "create_agentic_trainer"),
    "create_rag_trainer": ("agenttune.rag", "create_rag_trainer"),
    "create_distill_trainer": ("agenttune.agentic", "create_distill_trainer"),
}


def __getattr__(name: str):
    """PEP 562 lazy attribute access — defers the heavy TRL/torch import chain
    until one of these factories is actually used, not just on `import agenttune`."""
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value  # cache on the module so repeated access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
