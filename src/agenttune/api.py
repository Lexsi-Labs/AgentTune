"""Public SDK facade for AgentTune.

This is the *stable, documented* surface a platform or integration should import.
Everything here is a thin wrapper over internals that are otherwise undocumented
and free to change.

Headline: :func:`run_pipeline` runs a DECIDE decision pipeline (the well-tested,
GPU-free core). :func:`train_agentic` is a thin wrapper over the agentic RL
trainers and requires a GPU plus a compatible ``trl``/``torch`` stack.

Example
-------
>>> from agenttune import api
>>> result = api.run_pipeline("bfsi/kyc_triage", "Customer: John Doe ...")
>>> result.verdict
'APPROVE'
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["PipelineResult", "run_pipeline", "arun_pipeline", "train_agentic"]

# The agentic RL algorithms exposed by core.backend_factory.create_agentic_trainer.
AGENTIC_ALGORITHMS = ("grpo", "dpo", "ppo", "rloo", "bco")


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of a single DECIDE pipeline run (inference mode)."""

    pipeline_id: str
    template_id: str
    verdict: Any
    verdict_label: Any
    confidence: Any
    reason: Any
    step_count: int
    elapsed_seconds: float
    is_complete: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _state_to_result(state: Any) -> PipelineResult:
    return PipelineResult(
        pipeline_id=state.pipeline_id,
        template_id=state.template_id,
        verdict=state.verdict,
        verdict_label=state.verdict_label,
        confidence=state.confidence,
        reason=state.reason,
        step_count=state.step_count,
        elapsed_seconds=state.elapsed_seconds,
        is_complete=state.is_complete,
        error=state.error,
    )


async def arun_pipeline(
    template: str,
    input_text: str,
    *,
    config: str | None = None,
) -> PipelineResult:
    """Async variant of :func:`run_pipeline`.

    Use this from an existing event loop (e.g. inside a FastAPI handler).

    Parameters
    ----------
    template:
        Template ID (e.g. ``"bfsi/kyc_triage"``) or a path to a template YAML.
    input_text:
        The input passed to the pipeline's first stage.
    config:
        Optional path to a global ``config.yaml``. Defaults to ``./config.yaml``.
    """
    from agenttune.decide.graph_runner import GraphRunner

    runner = GraphRunner.from_template(template, config or "./config.yaml")
    state = await runner.run(input_text)
    return _state_to_result(state)


def run_pipeline(
    template: str,
    input_text: str,
    *,
    config: str | None = None,
) -> PipelineResult:
    """Run a DECIDE decision pipeline once (inference mode) and return the result.

    This is a synchronous convenience wrapper around :func:`arun_pipeline`; it
    creates its own event loop, so do not call it from within a running loop —
    use :func:`arun_pipeline` there instead.

    See :func:`arun_pipeline` for parameter documentation.
    """
    return asyncio.run(arun_pipeline(template, input_text, config=config))


def train_agentic(algorithm: str, **kwargs: Any):
    """Create an agentic RL trainer for ``algorithm`` and return it.

    Thin wrapper over ``agenttune.core.backend_factory.create_agentic_trainer``.
    Supported algorithms: ``grpo``, ``dpo``, ``ppo``, ``rloo``, ``bco``.

    .. note::
       Agentic training requires a GPU and a compatible ``trl``/``torch`` stack.
       The returned object exposes ``.train()``. This wrapper validates the
       algorithm eagerly so a bad value fails fast without importing the heavy
       training stack.
    """
    algo = algorithm.lower()
    if algo not in AGENTIC_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. "
            f"Valid algorithms: {', '.join(AGENTIC_ALGORITHMS)}."
        )
    from agenttune.core.backend_factory import create_agentic_trainer

    return create_agentic_trainer(algo, **kwargs)
