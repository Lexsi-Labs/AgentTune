"""
composite.py
Weighted combination of reward functions.
Accepts callables, string names (looked up in REWARD_REGISTRY), or a mix.
"""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def _resolve(fn) -> Callable:
    """Resolve a string name → callable via REWARD_REGISTRY, or pass through."""
    if callable(fn):
        return fn

    if isinstance(fn, str):
        from agenttune.agentic.rewards.builtin_rewards import REWARD_REGISTRY

        if fn not in REWARD_REGISTRY:
            available = list(REWARD_REGISTRY.keys())
            raise ValueError(
                f"[combine_rewards] Unknown reward function '{fn}'.\n"
                f"Available built-ins: {available}\n"
                f"Or pass a callable directly."
            )
        return REWARD_REGISTRY[fn]

    raise TypeError(f"[combine_rewards] Expected str or callable, got {type(fn).__name__}.")


def combine_rewards(reward_funcs, weights=None) -> Callable:
    """
    Combine reward functions into a single weighted callable.

    Parameters
    ----------
    reward_funcs : callable | str | list[callable | str]
        - A single callable
        - A string name from REWARD_REGISTRY
        - A list mixing callables and/or string names
    weights : list[float] | None
        Per-function weights. Normalised automatically.
        Defaults to uniform weighting.

    Returns
    -------
    Single callable compatible with GRPOTrainer reward_funcs.

    Examples
    --------
    # All string names
    combine_rewards(["correctness_reward", "structure_reward"])

    # Mixed
    combine_rewards(["correctness_reward", my_custom_fn], weights=[2.0, 1.0])

    # Single string
    combine_rewards("reward_tool_used")

    # Single callable (passthrough, still safe to call)
    combine_rewards(my_fn)
    """
    # Normalise to list
    if not isinstance(reward_funcs, list):
        reward_funcs = [reward_funcs]

    # Resolve all entries
    resolved = [_resolve(fn) for fn in reward_funcs]
    n = len(resolved)

    # Weights
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError(f"[combine_rewards] {n} functions but {len(weights)} weights.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("[combine_rewards] Weights must sum to a positive number.")
    norm_weights = [w / total for w in weights]

    fn_names = [getattr(f, "__name__", str(f)) for f in resolved]
    logger.info(
        "[combine_rewards] "
        + ", ".join(f"{name}×{w:.3f}" for name, w in zip(fn_names, norm_weights, strict=False))
    )

    def _combined(completions, **kwargs):
        totals = [0.0] * len(completions)
        for fn, w in zip(resolved, norm_weights, strict=False):
            try:
                scores = fn(completions=completions, **kwargs)
            except TypeError:
                # `fn` may use the positional `(completions, **kwargs)` convention
                # instead of the keyword one. Retry positionally; if THAT also
                # fails, let the error propagate. A reward function that raises
                # under both conventions is a real bug — silently scoring it 0.0
                # would train the policy against a dead (all-zero) signal, which
                # is far worse than failing loudly.
                scores = fn(completions, **kwargs)
            for i, s in enumerate(scores):
                totals[i] += w * float(s if s is not None else 0.0)
        return totals

    _combined.__name__ = "combined_reward(" + "+".join(fn_names) + ")"
    return _combined
