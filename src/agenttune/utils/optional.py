"""
Optional-dependency helpers for AgentTune.

Import guards for extras that are not installed by default. All guarded code
imports lazily — this file must not import the optional packages at module
load time.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# OpenEnv availability flag  (mirrors TRL_AVAILABLE in backend_factory.py)
# ---------------------------------------------------------------------------

try:
    import openenv  # noqa: F401

    OPENENV_AVAILABLE = True
except ImportError:
    OPENENV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Guard helpers — call these at the entry point of any function that needs
# the extra; they raise ImportError with a helpful install hint.
# ---------------------------------------------------------------------------


def require_vllm() -> None:
    """Raise ImportError with pip hint if vLLM is not installed (it is an extra)."""
    import importlib.util

    if importlib.util.find_spec("vllm") is None:
        raise ImportError(
            "vLLM is not installed. It is optional (Linux + CUDA only):\n\n"
            "    pip install 'agenttune[vllm]'\n\n"
            "or run without it (e.g. use_vllm=False, the transformers backend)."
        )


def require_openenv() -> None:
    """Raise ImportError with pip hint if openenv is not installed."""
    if not OPENENV_AVAILABLE:
        raise ImportError(
            "openenv is required for remote environment tools but is not installed.\n"
            "It's a base dependency of agenttune — reinstall with:\n\n"
            "    pip install -e .\n"
        )
