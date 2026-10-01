"""
Agentic Backend Factory for AgentTune
======================================
Supports agentic training only:
  - Algorithm : GRPO | DPO | PPO | RLOO | BCO
  - Backend   : TRL (default, imported eagerly) | Unsloth (optional, lazy)

Unsloth patches transformers/trl/peft at import time, so its trainers are
never imported until a caller explicitly asks for backend="unsloth" — the
default TRL path is unaffected whether or not Unsloth is installed. Set
PURE_TRL_MODE=1 to guarantee `import unsloth` never runs in this process
(see ..backends._imports).

All configuration is passed as **kwargs — no config object required.
"""

import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

from ..utils.environment import patch_colab_outstream_close

patch_colab_outstream_close()

# ── TRL agentic trainer imports (eager — no global patching risk) ─────────────
try:
    from ..backends.trl.agentic.bco.agentic_bco import TrlAgenticBCO
    from ..backends.trl.agentic.dpo.agentic_dpo import TrlAgenticDPO
    from ..backends.trl.agentic.grpo.agentic_grpo import TrlAgenticGrpo
    from ..backends.trl.agentic.ppo.agentic_ppo import TrlAgenticPPO
    from ..backends.trl.agentic.rloo.agentic_rloo import TrlAgenticRloo

    TRL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TRL agentic backends not available: {e}")
    TRL_AVAILABLE = False

from ..backends._imports import _check_unsloth_available


# ── Enums ─────────────────────────────────────────────────────────────────────
class AgenticAlgorithm(Enum):
    GRPO = "grpo"
    DPO = "dpo"
    PPO = "ppo"
    RLOO = "rloo"
    BCO = "bco"


class BackendType(Enum):
    TRL = "trl"
    UNSLOTH = "unsloth"


# ── Lazy Unsloth trainer import ────────────────────────────────────────────────
def _lazy_import_unsloth_agentic_trainer(algorithm: str):
    """Import the requested Unsloth agentic trainer class on demand only.

    Never called at module load time — only from inside a placeholder's
    __new__, at the moment a caller passes backend="unsloth".
    """
    if algorithm == "grpo":
        from ..backends.unsloth.agentic.grpo.agentic_grpo import TrlAgenticGrpo as _Trainer
    elif algorithm == "dpo":
        from ..backends.unsloth.agentic.dpo.agentic_dpo import TrlAgenticDPO as _Trainer
    elif algorithm == "ppo":
        from ..backends.unsloth.agentic.ppo.agentic_ppo import TrlAgenticPPO as _Trainer
    elif algorithm == "rloo":
        from ..backends.unsloth.agentic.rloo.agentic_rloo import TrlAgenticRloo as _Trainer
    elif algorithm == "bco":
        from ..backends.unsloth.agentic.bco.agentic_bco import TrlAgenticBCO as _Trainer
    else:
        raise ValueError(f"Unknown agentic algorithm for Unsloth backend: {algorithm}")
    return _Trainer


# ── Registry ──────────────────────────────────────────────────────────────────
_REGISTRY: dict[tuple[AgenticAlgorithm, BackendType], Any] = {}


def _register():
    if TRL_AVAILABLE:
        _TRL_TRAINERS = {
            AgenticAlgorithm.GRPO: TrlAgenticGrpo,
            AgenticAlgorithm.DPO: TrlAgenticDPO,
            AgenticAlgorithm.PPO: TrlAgenticPPO,
            AgenticAlgorithm.RLOO: TrlAgenticRloo,
            AgenticAlgorithm.BCO: TrlAgenticBCO,
        }

        def _make_trl_wrapper(trainer_class):
            class _TrlWrapper:
                @classmethod
                def is_available(cls):
                    return TRL_AVAILABLE

                def __new__(cls, **kwargs):
                    return trainer_class(**kwargs)

            return _TrlWrapper

        for alg, trainer_class in _TRL_TRAINERS.items():
            _REGISTRY[(alg, BackendType.TRL)] = _make_trl_wrapper(trainer_class)

    # Unsloth placeholders always register — availability (and the actual
    # `import unsloth`) is only checked/triggered when instantiated.
    def _make_unsloth_wrapper(alg_name: str):
        class _UnslothWrapper:
            @classmethod
            def is_available(cls):
                return _check_unsloth_available()

            def __new__(cls, **kwargs):
                trainer_class = _lazy_import_unsloth_agentic_trainer(alg_name)
                return trainer_class(**kwargs)

        return _UnslothWrapper

    for alg in AgenticAlgorithm:
        _REGISTRY[(alg, BackendType.UNSLOTH)] = _make_unsloth_wrapper(alg.value)


_register()


# ── Public API ────────────────────────────────────────────────────────────────
def create_agentic_trainer(algorithm: str, backend: str = "auto", **kwargs) -> Any:
    """
    Create an agentic trainer (GRPO, DPO, PPO, RLOO, or BCO).

    All configuration is passed as **kwargs and forwarded verbatim to the
    underlying trainer class — no config object needed.

    Parameters
    ----------
    algorithm : "grpo" | "dpo" | "ppo" | "rloo" | "bco"
    backend   : "auto" (default) | "trl" | "unsloth".
                "auto" uses Unsloth if it's importable in this environment,
                otherwise silently falls back to TRL — no error either way.
                "trl" / "unsloth" pin a specific backend: if that one isn't
                available, this raises instead of silently switching.
    **kwargs  : see the matching Trl/Unsloth agentic trainer class for
                accepted keys.

    Examples
    --------
    >>> trainer = create_agentic_trainer(
    ...     algorithm    = "grpo",
    ...     model        = "Qwen/Qwen2.5-1.5B-Instruct",
    ...     reward_funcs = my_reward_fn,
    ...     tools        = [my_tool],
    ...     train_dataset = my_dataset,
    ...     output_dir   = "./runs/grpo",
    ...     max_steps    = 100,
    ... )
    >>> results = trainer.train()

    >>> trainer = create_agentic_trainer("grpo", backend="unsloth", ...)
    """
    alg = algorithm.lower()
    try:
        alg_enum = AgenticAlgorithm(alg)
    except ValueError:
        raise ValueError(  # noqa: B904
            f"Unsupported agentic algorithm '{alg}'. "
            f"Choose from: {[a.value for a in AgenticAlgorithm]}"
        )

    be = backend.lower()
    if be == "auto":
        unsloth_class = _REGISTRY.get((alg_enum, BackendType.UNSLOTH))
        if unsloth_class is not None and unsloth_class.is_available():
            backend_enum = BackendType.UNSLOTH
        else:
            backend_enum = BackendType.TRL
    else:
        try:
            backend_enum = BackendType(be)
        except ValueError:
            raise ValueError(  # noqa: B904
                f"Unsupported backend '{be}'. Choose from: "
                f"{['auto'] + [b.value for b in BackendType]}"
            )

    if backend_enum is BackendType.TRL and not TRL_AVAILABLE:
        raise RuntimeError(
            "TRL is required for the 'trl' agentic backend.\n" "Install with: pip install trl"
        )

    trainer_class = _REGISTRY.get((alg_enum, backend_enum))
    if trainer_class is None:
        raise RuntimeError(
            f"No trainer registered for algorithm '{alg}' on backend '{backend_enum.value}'."
        )

    if not trainer_class.is_available():
        if backend_enum is BackendType.UNSLOTH:
            raise RuntimeError(
                "Unsloth is not available in this environment.\n"
                "Install with: pip install unsloth\n"
                'Or use backend="trl" instead.'
            )
        raise RuntimeError(
            f"Backend '{backend_enum.value}' is not available for algorithm '{alg}'."
        )

    logger.info(f"create_agentic_trainer: algorithm={alg} backend={backend_enum.value}")
    from ..utils.environment import patch_colab_outstream_close

    patch_colab_outstream_close()
    funcs = kwargs.get("reward_funcs")
    single_callable = callable(funcs) or (
        isinstance(funcs, list) and len(funcs) == 1 and callable(funcs[0])
    )
    # A single reward function goes through as-is: combine_rewards would call it as
    # fn(completions, **kw) and break the fn(trajectory) / fn(responses[, prompts])
    # styles the rollout's reward wrapper accepts.
    if funcs is not None and not (single_callable and kwargs.get("reward_weights") is None):
        from ..agentic.rewards.composite import combine_rewards

        kwargs = dict(kwargs)
        kwargs["reward_funcs"] = combine_rewards(
            kwargs["reward_funcs"],
            weights=kwargs.pop("reward_weights", None),
        )
    trainer = trainer_class(**kwargs)
    from ..utils.hf_publish import attach_hub_push

    return attach_hub_push(trainer, algorithm=alg, backend=backend_enum.value)


def list_agentic_backends() -> dict[str, Any]:
    """Print and return agentic backend availability, keyed 'algorithm:backend'."""
    status = {
        f"{alg.value}:{backend.value}": {
            "available": cls.is_available(),
            "backend": backend.value,
            "class": cls.__name__,
        }
        for (alg, backend), cls in _REGISTRY.items()
    }

    logger.info("\n" + "=" * 50)
    logger.info("AGENTTUNE — AGENTIC BACKEND STATUS")
    logger.info("=" * 50)
    if not status:
        logger.info("  ❌ No agentic backends available (TRL not installed)")
    for key, info in status.items():
        icon = "✅" if info["available"] else "❌"
        logger.info(f"  {icon}  {key:16s}  →  {info['backend'].upper()}")
    logger.info("=" * 50 + "\n")

    return status
