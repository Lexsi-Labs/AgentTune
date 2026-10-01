"""
Shared availability checks for optional training backends.

Unsloth patches ``transformers``/``trl``/``peft`` at import time and must be
imported before those libraries to apply correctly. Checking (or requesting)
Unsloth after TRL is already loaded doesn't raise, but the optimizations
silently don't apply — so callers that only want TRL must never trigger
``import unsloth`` at all. ``_check_unsloth_available`` enforces that:
result is memoized after the first real check, and the ``*_MODE`` env vars
below short-circuit to ``False`` without ever importing the package.
"""

import inspect
import os

UNSLOTH_AVAILABLE = None  # memoized: None = not checked yet
UNSLOTH_ERROR_INFO = None


def _should_prevent_unsloth() -> bool:
    return os.environ.get("PURE_TRL_MODE", "0") == "1"


def _check_unsloth_available() -> bool:
    """Lazily check (and cache) whether Unsloth can be imported."""
    global UNSLOTH_AVAILABLE, UNSLOTH_ERROR_INFO
    if UNSLOTH_AVAILABLE is not None:
        return UNSLOTH_AVAILABLE

    if _should_prevent_unsloth():
        UNSLOTH_AVAILABLE = False
        UNSLOTH_ERROR_INFO = "Unsloth disabled via PURE_TRL_MODE"
        return False

    try:
        import unsloth  # noqa: F401  (must precede transformers/trl/peft imports)
        from unsloth import FastLanguageModel  # noqa: F401

        UNSLOTH_AVAILABLE = True
        UNSLOTH_ERROR_INFO = None
    except Exception as e:  # pragma: no cover - depends on optional install
        UNSLOTH_AVAILABLE = False
        UNSLOTH_ERROR_INFO = str(e)

    return UNSLOTH_AVAILABLE


def real_init_param_names(cls: type, fallback: "set[str]") -> "set[str]":
    """Return the real parameter names of ``cls.__init__``, working around
    Unsloth's monkey-patched TRL trainers.

    Once ``import unsloth`` has run, it wraps every TRL trainer's
    ``__init__`` (GRPOTrainer, DPOTrainer, PPOTrainer, RLOOTrainer,
    BCOTrainer, ...) in a generic ``(self, *args, **kwargs)`` shim. Backend
    code that introspects that signature to route caller kwargs between
    ``*Config`` and the trainer constructor then sees only ``{"args",
    "kwargs"}`` and silently drops every real kwarg -- including ``model``.

    When that collapse is detected, fall back to the caller-supplied set of
    known parameter names (captured from the unpatched trainer) instead of
    trusting the (useless) live signature.
    """
    try:
        params = set(inspect.signature(cls.__init__).parameters) - {"self"}
    except (TypeError, ValueError):
        params = set()
    if params <= {"args", "kwargs"}:
        return set(fallback)
    return params
