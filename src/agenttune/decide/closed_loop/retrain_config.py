"""
Retrain Config — Path B, Week 2
===============================

Turns a drained batch of ``TrainingExample`` objects into an adapter-only
(LoRA) retrain set up on the existing DPO / BCO trainers.

Two responsibilities:

1. ``examples_to_dpo_dataset`` / ``examples_to_bco_dataset`` — convert the
   Path-A example contract into the column layout each trainer expects.
2. ``build_retrain_config`` / ``build_retrainer`` — assemble the kwargs for an
   adapter-only (LoRA) run and instantiate the trainer via the existing
   ``create_agentic_trainer`` factory.

Adapter-only by design: we always attach a ``peft_config`` (LoRA) so a retrain
touches a small set of adapter weights, never the full model.  This keeps each
self-healing retrain cheap and reversible.

Scope (Week 2): config assembly, dataset conversion, and a one-call
``run_retrain`` that trains end-to-end on a small dataset.  The *background*
(non-blocking) execution of this lives in ``retrain_runner.py``; full wiring
into the closed loop is Week 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agenttune.decide.closed_loop.contracts import TrainingExample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message-format helpers
# ---------------------------------------------------------------------------


def _messages_to_text(messages: list[dict[str, str]] | None) -> str:
    """Flatten a [{"role","content"}, ...] list into a single string.

    DPO/BCO accept either chat-format lists or plain strings; we normalise to
    a simple ``role: content`` join so the converter is trainer-agnostic and
    deterministic in tests.
    """
    if not messages:
        return ""
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        parts.append(f"{role}: {content}" if role else content)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dataset converters
# ---------------------------------------------------------------------------


def examples_to_dpo_dataset(examples: list[TrainingExample]) -> list[dict[str, str]]:
    """Convert examples to DPO rows: ``{prompt, chosen, rejected}``.

    Precedence (authoritative → fallback):

    1. **Generator-set pair wins.** Path A's ``TrainingExampleGenerator`` sets
       ``chosen``/``rejected`` directly (best correction vs. the *real failed
       action* recovered from the trajectory context). When present, that pair
       is used verbatim — it is the correct preference signal and is never
       overwritten here.
    2. **Bridge fallback.** Only when the generator did NOT set a pair (e.g. a
       producer that emits ``completions``/``rewards`` only) do we derive one
       via :meth:`TrainingExample.derive_preference_from_completions`
       (best-reward → chosen, worst → rejected). Kept for resilience and
       covered by tests; not used by Path A's real output.

    Examples with neither a generator pair nor >=2 distinct-reward completions
    are skipped and counted.
    """
    rows: list[dict[str, str]] = []
    skipped = 0
    for ex in examples:
        if not ex.has_preference_pair():
            # Fallback only: bridge multi-completion output into a pair.
            # (No-op when the generator already set chosen/rejected.)
            ex.derive_preference_from_completions()
        if not ex.has_preference_pair():
            skipped += 1
            continue
        rows.append(
            {
                "prompt": _messages_to_text(ex.prompt),
                "chosen": _messages_to_text(ex.chosen),
                "rejected": _messages_to_text(ex.rejected),
            }
        )
    if skipped:
        logger.info(
            "examples_to_dpo_dataset: skipped %d example(s) lacking a usable preference pair",
            skipped,
        )
    return rows


def examples_to_bco_dataset(examples: list[TrainingExample]) -> list[dict[str, Any]]:
    """Convert examples to BCO rows: ``{prompt, completion, label}``.

    BCO is binary: a ``chosen`` response is desirable (label True); a
    ``rejected`` response is undesirable (label False).

    Precedence matches :func:`examples_to_dpo_dataset`: the generator-set
    ``chosen``/``rejected`` pair is authoritative and used first (one or two
    rows). Only when neither is set do we fall back to the multi-completion
    form, labelling each completion by whether its reward is at/above the
    example's mean reward (desirable) or below (undesirable).
    """
    rows: list[dict[str, Any]] = []
    for ex in examples:
        prompt = _messages_to_text(ex.prompt)
        produced = False
        # Preference form first.
        if ex.chosen:
            rows.append(
                {"prompt": prompt, "completion": _messages_to_text(ex.chosen), "label": True}
            )
            produced = True
        if ex.rejected:
            rows.append(
                {"prompt": prompt, "completion": _messages_to_text(ex.rejected), "label": False}
            )
            produced = True
        # Multi-completion form: threshold each completion against the mean reward.
        if not produced and ex.has_completions():
            mean_r = sum(ex.rewards) / len(ex.rewards)
            for comp, r in zip(ex.completions, ex.rewards, strict=False):
                rows.append(
                    {
                        "prompt": prompt,
                        "completion": _messages_to_text(comp),
                        "label": bool(r >= mean_r),
                    }
                )
    return rows


def to_hf_dataset(rows: list[dict[str, Any]]):
    """Wrap converted rows in a HF ``Dataset`` (lazy import)."""
    from datasets import Dataset

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Adapter-only (LoRA) retrain config
# ---------------------------------------------------------------------------


@dataclass
class RetrainConfig:
    """Adapter-only retrain settings.

    ``algorithm`` is ``"dpo"`` or ``"bco"`` (the two preference trainers
    BUILD_PLAN names for Week 2). GRPO retrain is deferred. ``lora_*`` define
    the adapter; ``peft_config`` is built from them unless one is supplied
    explicitly.
    """

    model: str
    algorithm: str = "dpo"  # "dpo" | "bco"
    output_dir: str = "./output/retrain"
    # adapter (LoRA) knobs — keep small; this is a cheap incremental retrain
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None  # None → peft default per arch
    # training knobs — small by default for fast self-healing cycles
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    max_steps: int | None = None  # cap steps for quick retrains
    beta: float = 0.1
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)  # passthrough to trainer

    def peft_config_dict(self) -> dict[str, Any]:
        """LoRA config as a dict (consumed by the trainers' _resolve_peft_config)."""
        cfg: dict[str, Any] = {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "task_type": "CAUSAL_LM",
        }
        if self.lora_target_modules:
            cfg["target_modules"] = self.lora_target_modules
        return cfg


def build_retrain_config(
    examples: list[TrainingExample],
    config: RetrainConfig,
) -> dict[str, Any]:
    """Assemble the kwargs dict for ``create_agentic_trainer``.

    Converts ``examples`` into the right dataset for ``config.algorithm`` and
    attaches the LoRA ``peft_config`` so the run is adapter-only. Raises
    ``ValueError`` if conversion yields no usable rows.
    """
    alg = config.algorithm.lower()
    if alg == "dpo":
        rows = examples_to_dpo_dataset(examples)
    elif alg == "bco":
        rows = examples_to_bco_dataset(examples)
    else:
        raise ValueError(
            f"Unsupported retrain algorithm '{config.algorithm}'. Use 'dpo' or 'bco' "
            "(GRPO retrain is deferred)."
        )

    if not rows:
        raise ValueError(
            f"No usable {alg.upper()} rows from {len(examples)} example(s). "
            "DPO needs chosen+rejected pairs; check the buffer contents."
        )

    train_dataset = to_hf_dataset(rows)

    kwargs: dict[str, Any] = {
        "model": config.model,
        "train_dataset": train_dataset,
        "output_dir": config.output_dir,
        "peft_config": config.peft_config_dict(),  # ← adapter-only
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "beta": config.beta,
        "seed": config.seed,
    }
    if config.max_steps is not None:
        kwargs["max_steps"] = config.max_steps
    kwargs.update(config.extra)

    logger.info(
        "build_retrain_config: alg=%s rows=%d adapter-only(LoRA r=%d) model=%s",
        alg,
        len(rows),
        config.lora_r,
        config.model,
    )
    return kwargs


def build_retrainer(examples: list[TrainingExample], config: RetrainConfig):
    """Instantiate the adapter-only trainer via the existing factory."""
    from agenttune.core.backend_factory import create_agentic_trainer

    kwargs = build_retrain_config(examples, config)
    return create_agentic_trainer(algorithm=config.algorithm.lower(), **kwargs)


def run_retrain(examples: list[TrainingExample], config: RetrainConfig) -> dict[str, Any]:
    """End-to-end adapter-only retrain on the given examples.

    Returns the trainer's stats dict.  Synchronous — the non-blocking variant
    is ``retrain_runner.BackgroundRetrainer`` (Week 2 design) wired in Week 3.
    """
    trainer = build_retrainer(examples, config)
    return trainer.train()
