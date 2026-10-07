"""
TrlAgenticGrpo
==============
A thin, kwargs-driven wrapper around TRL's GRPOTrainer with full support for:

  - Standard GRPO (text completions as plain strings)
  - Agentic GRPO  (completions as lists of message dicts with tool calls/results)
  - External tools passed to GRPOTrainer (tool-calling loop handled by TRL)
  - Custom rollout_func / environment_factory
  - vLLM-accelerated generation
  - Hub push, trackio logging, and all GRPOConfig knobs

All configuration lives in **kwargs — nothing is hard-coded.
GRPOConfig and GRPOTrainer parameters are auto-separated via introspection.

─────────────────────────────────────────────────────────────────────────────
Standard usage
--------------
>>> trainer = TrlAgenticGrpo(
...     model="Qwen/Qwen2.5-1.5B-Instruct",
...     reward_funcs=my_reward_fn,
...     dataset_name="trl-lib/tldr",
...     output_dir="./runs/grpo",
...     num_generations=4,
...     beta=0.04,
... )
>>> results = trainer.train()

─────────────────────────────────────────────────────────────────────────────
Agentic usage (matches the BioGRID notebook pattern exactly)
------------------------------------------------------------
>>> trainer = TrlAgenticGrpo(
...     model="Qwen/Qwen3-1.7B",
...     reward_funcs=[correctness_reward, structure_reward, query_reward],
...     tools=[query_biogrid],                   # ← agentic tools
...     train_dataset=train_dataset,             # pre-formatted with chat prompts
...     output_dir="grpo_biogrid_run",
...     max_steps=100,
...     max_completion_length=1024,
...     per_device_train_batch_size=2,
...     num_generations=2,
...     chat_template_kwargs={"enable_thinking": False},
...     log_completions=True,
...     push_to_hub=True,
...     report_to="trackio",
... )
>>> results = trainer.train()
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(kwargs: dict[str, Any], *keys, default=None):
    """Return the first matching key found in *kwargs*, else *default*."""
    for k in keys:
        if k in kwargs:
            return kwargs[k]
    return default


def _split_kwargs(kwargs: dict[str, Any]) -> tuple[dict, dict]:
    """
    Split *kwargs* into (grpo_config_kwargs, grpo_trainer_kwargs).

    Uses live introspection so it automatically tracks future TRL API changes.
    Unrecognised keys (data / meta params like dataset_name) are discarded.
    """
    from agenttune.utils.environment import patch_colab_outstream_close

    patch_colab_outstream_close()
    from trl import GRPOConfig, GRPOTrainer

    config_keys = set(inspect.signature(GRPOConfig.__init__).parameters) - {"self"}
    trainer_keys = set(inspect.signature(GRPOTrainer.__init__).parameters) - {"self"}

    config_kw: dict[str, Any] = {}
    trainer_kw: dict[str, Any] = {}

    for k, v in kwargs.items():
        if k in trainer_keys:
            trainer_kw[k] = v
        elif k in config_keys:
            config_kw[k] = v
        # else: data / meta param — intentionally ignored

    return config_kw, trainer_kw


# ── Add this helper at module level (after _build_rloo_rollout_fn) ────────────


def _resolve_peft_config(raw):
    """
    Accept either:
      - None              → return None
      - a PeftConfig obj  → return as-is
      - a dict            → build LoraConfig from it
    """
    if raw is None:
        return None
    try:
        from peft import PeftConfig

        if isinstance(raw, PeftConfig):
            return raw
    except ImportError:
        raise ImportError("[AgentTune] peft is required. pip install peft")  # noqa: B904
    if isinstance(raw, dict):
        from peft import LoraConfig

        return LoraConfig(**raw)
    raise TypeError(f"[AgentTune] peft_config must be a dict or PeftConfig, got {type(raw)}")


def _build_grpo_rollout_fn(
    rollout_engine=None,
    rollout_backend=None,
    tools=None,
    max_steps: int = 20,
    system_prompt=None,
    engine_kwargs=None,
    **rollout_kwargs,
) -> Callable:
    """Build a tool-calling ``rollout_func`` from ``tools`` for agentic GRPO.

    The installed TRL only executes ``tools=`` natively on transformers>=5 (and
    can't schema a :class:`BaseTool` object), so — exactly like the RLOO backend —
    we drive the tool loop through agenttune's own rollout engine instead. The
    returned callable yields ``prompt_ids``/``completion_ids``/``logprobs`` plus
    the rollout ``trajectories`` (and ``responses``/``env_mask``/…), which the
    trainer forwards to the reward functions as extra fields — so a RAG reward can
    read back which chunks were retrieved.
    """
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

    return create_rollout_fn(
        rollout_engine=rollout_engine,
        rollout_backend=rollout_backend,
        tools=tools,
        max_steps=max_steps,
        system_prompt=system_prompt,
        engine_kwargs=engine_kwargs or {},
        **rollout_kwargs,
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class TrlAgenticGrpo:
    """
    kwargs-driven GRPO trainer supporting both standard and agentic modes.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  GRPOTrainer params  (routed automatically via introspection)       │
    │  model, reward_funcs, processing_class, peft_config,               │
    │  tools, rollout_func, environment_factory, callbacks, optimizers    │
    ├─────────────────────────────────────────────────────────────────────┤
    │  GRPOConfig params   (routed automatically via introspection)       │
    │  output_dir, beta, epsilon, num_generations, max_completion_length, │
    │  temperature, top_p, use_vllm, vllm_mode, chat_template_kwargs,    │
    │  log_completions, push_to_hub, report_to, trackio_space_id, …      │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Data params   (consumed by setup_data, never forwarded to TRL)     │
    │  dataset_name, dataset_config, split, max_samples,                  │
    │  column_mapping, system_prompt,                                      │
    │  format_fn          callable applied to each example after loading  │
    │                     signature: fn(example: dict) -> dict            │
    │                     use remove_columns to drop unwanted cols after  │
    │  format_batched     set True if format_fn operates on batches       │
    │  format_remove_columns  list[str] of columns to drop after format  │
    │  — or pass train_dataset/eval_dataset directly to skip loading —    │
    └─────────────────────────────────────────────────────────────────────┘

    Reward function signatures
    --------------------------
    Standard mode  : f(completions: list[str], **kwargs) -> list[float]
    Agentic mode   : f(completions: list[list[dict]], **kwargs) -> list[float]
        where each completion is a list of message dicts, e.g.:
        [{"role": "assistant", "tool_calls": [...]},
         {"role": "tool", "content": "..."},
         {"role": "assistant", "content": "*Yes*"}]
    TRL handles the agentic loop; reward functions just inspect the result.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, **kwargs):
        self.kwargs = kwargs

        # Runtime state
        self.trainer: Any = None
        self.train_dataset: Any = None
        self.eval_dataset: Any = kwargs.get("eval_dataset", None)
        self.training_history: list[dict] = []
        self._train_result: Any = None

        # Detect agentic mode at construction time for logging
        self._agentic_mode = bool(
            kwargs.get("tools") or kwargs.get("rollout_func") or kwargs.get("environment_factory")
        )
        if self._agentic_mode:
            logger.info(
                "[TrlAgenticGrpo] Agentic mode detected "
                f"(tools={bool(kwargs.get('tools'))}, "
                f"rollout_func={bool(kwargs.get('rollout_func'))}, "
                f"environment_factory={bool(kwargs.get('environment_factory'))})"
            )

    # ------------------------------------------------------------------ #
    #  Data setup                                                          #
    # ------------------------------------------------------------------ #

    def setup_data(self) -> None:
        """
        Load train (and optionally eval) dataset.

        Two loading paths — tried in this order:
        ─────────────────────────────────────────
        1. Pre-loaded  — caller passes ``train_dataset=<Dataset>`` directly.
                         format_fn is still applied if provided.

        2. DataManager — default for all other cases (HF Hub, local files,
                         custom splits). DataManager handles raw HF datasets,
                         chat-template injection, column mapping, thinking
                         mode, and processing_fn natively — no separate raw
                         HF path is needed.

        Common kwargs
        -------------
        dataset_name / dataset        : HF dataset id or local path
        dataset_config / config_name  : HF dataset config name
        split                         : e.g. "train" or "train[:5000]"
        max_samples                   : int — truncate training split
        column_mapping                : dict — rename cols before processing
        system_prompt                 : str  — prepended via chat template
        processing_class              : tokenizer passed to DataManager
        enable_thinking               : bool — DataManager thinking mode
        processing_fn                 : callable — DataManager processing_fn
        processing_batched            : bool     — DataManager batched flag
        format_fn                     : callable(example) -> dict,
                                        applied after DataManager output
        format_batched                : bool (default False)
        format_remove_columns         : list[str] cols to drop after format_fn
        data_manager_config           : dict — any extra kwargs forwarded
                                        verbatim to DataManager.__init__
        """

        # ── format_fn helper (applied in both paths) ─────────────────────
        format_fn = _get(self.kwargs, "format_fn", default=None)
        format_batched = _get(self.kwargs, "format_batched", default=False)
        format_remove_columns = _get(self.kwargs, "format_remove_columns", default=None)

        def _apply_format_fn(ds):
            if format_fn is None or ds is None:
                return ds
            map_kw: dict[str, Any] = {"batched": format_batched}
            if format_remove_columns:
                map_kw["remove_columns"] = format_remove_columns
            return ds.map(format_fn, **map_kw)

        # ═══════════════════════════════════════════════════════════════════
        # PATH 1 — pre-loaded datasets passed directly by the caller
        # ═══════════════════════════════════════════════════════════════════
        if "train_dataset" in self.kwargs:
            self.train_dataset = self.kwargs["train_dataset"]
            if "eval_dataset" in self.kwargs:
                self.eval_dataset = self.kwargs["eval_dataset"]

            if format_fn is not None:
                logger.info("[TrlAgenticGrpo] Applying format_fn to pre-loaded train dataset")
                self.train_dataset = _apply_format_fn(self.train_dataset)
                if self.eval_dataset is not None:
                    logger.info("[TrlAgenticGrpo] Applying format_fn to pre-loaded eval dataset")
                    self.eval_dataset = _apply_format_fn(self.eval_dataset)

            logger.info(
                f"[TrlAgenticGrpo] Using pre-loaded dataset "
                f"({len(self.train_dataset)} train examples)"
            )
            return

        # ═══════════════════════════════════════════════════════════════════
        # PATH 2 — DataManager (handles HF Hub, local, and everything else)
        # ═══════════════════════════════════════════════════════════════════
        from agenttune.data.manager import DataManager

        dataset_name = _get(self.kwargs, "dataset_name", "dataset", default="trl-lib/tldr")
        config_name = _get(self.kwargs, "dataset_config", "config_name", default=None)
        split = _get(self.kwargs, "split", default=None)
        max_samples = _get(self.kwargs, "max_samples", default=None)
        data_manager_cfg = _get(self.kwargs, "data_manager_config", default=None)

        logger.info(f"[TrlAgenticGrpo] Loading via DataManager: {dataset_name}")

        # Build DataManager kwargs — mirrors TRLGRPOTrainer.setup_data exactly
        dm_kwargs: dict[str, Any] = {
            "task_type": "grpo",
            "system_prompt": _get(self.kwargs, "system_prompt", default=None),
            "tokenizer": _get(self.kwargs, "processing_class", default=None),
            "enable_thinking": _get(self.kwargs, "enable_thinking", default=False),
            "column_mapping": _get(self.kwargs, "column_mapping", default=None),
            "processing_fn": _get(self.kwargs, "processing_fn", default=None),
            "processing_batched": _get(self.kwargs, "processing_batched", default=False),
            "max_samples": max_samples,
        }

        # data_manager_config lets callers override/extend any DataManager param
        if isinstance(data_manager_cfg, dict):
            dm_kwargs.update(data_manager_cfg)

        manager = DataManager(**dm_kwargs)

        dataset_dict = manager.load_dataset(
            dataset_name,
            config_name=config_name,
            split=split,
        )

        train_ds = dataset_dict.get("train", None)
        eval_ds = dataset_dict.get("validation", None)

        # Apply optional format_fn on top of DataManager output
        train_ds = _apply_format_fn(train_ds)
        eval_ds = _apply_format_fn(eval_ds)

        # Safety net: enforce max_samples if DataManager didn't apply it
        if train_ds is not None and max_samples and len(train_ds) > max_samples:
            train_ds = train_ds.select(range(max_samples))

        self.train_dataset = train_ds
        if self.eval_dataset is None:
            self.eval_dataset = eval_ds

        logger.info(f"[TrlAgenticGrpo] Train : {len(self.train_dataset)} examples")
        if self.eval_dataset:
            logger.info(f"[TrlAgenticGrpo] Eval  : {len(self.eval_dataset)} examples")

        # Log a sample prompt (mirrors TRLGRPOTrainer behaviour)
        if self.train_dataset and len(self.train_dataset) > 0:
            sample = self.train_dataset[0]
            prompt_col = "prompt" if "prompt" in sample else "query"
            if prompt_col in sample:
                logger.info(
                    f"[TrlAgenticGrpo] Sample prompt (first 100 chars): "
                    f"{str(sample[prompt_col])[:100]}…"
                )
            logger.info(f"[TrlAgenticGrpo] Dataset columns: {self.train_dataset.column_names}")

    # ------------------------------------------------------------------ #
    #  Trainer setup                                                       #
    # ------------------------------------------------------------------ #

    def setup_trainer(self) -> None:
        from trl import GRPOConfig, GRPOTrainer
        from trl.extras.profiling import profiling_context

        from agenttune.agentic.rewards.composite import combine_rewards
        from agenttune.agentic.trajectory.dataset import trajectory_writer

        config_kw, trainer_kw = _split_kwargs(self.kwargs)

        raw_reward_funcs = trainer_kw.pop("reward_funcs", None) or self.kwargs.get("reward_funcs")
        if raw_reward_funcs is None:
            raise ValueError(
                "[TrlAgenticGrpo] 'reward_funcs' is required. "
                "Pass reward_funcs=<callable | str | list> to the constructor."
            )
        reward_weights = self.kwargs.get("reward_weights", None)
        trainer_kw["reward_funcs"] = combine_rewards(raw_reward_funcs, weights=reward_weights)
        # reward_weights is fully consumed by combine_rewards above (folded into the single
        # combined reward function) — it must not also reach GRPOConfig, which would compare
        # its original length against the now-single reward_funcs and raise a mismatch error.
        config_kw.pop("reward_weights", None)

        # ── GRPOConfig: sensible defaults ────────────────────────────────────
        output_dir = _get(self.kwargs, "output_dir", default="./output/grpo_agentic")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        config_defaults: dict[str, Any] = {
            "output_dir": output_dir,
            "num_train_epochs": _get(self.kwargs, "num_train_epochs", "epochs", default=1),
            "per_device_train_batch_size": _get(
                self.kwargs, "per_device_train_batch_size", "batch_size", default=1
            ),
            "gradient_accumulation_steps": _get(
                self.kwargs, "gradient_accumulation_steps", default=16
            ),
            "learning_rate": _get(self.kwargs, "learning_rate", "lr", default=1e-6),
            "warmup_steps": _get(self.kwargs, "warmup_steps", default=10),
            "max_grad_norm": _get(self.kwargs, "max_grad_norm", default=1.0),
            "weight_decay": _get(self.kwargs, "weight_decay", default=0.0),
            "seed": _get(self.kwargs, "seed", default=42),
            "logging_steps": _get(self.kwargs, "logging_steps", default=10),
            "save_steps": _get(self.kwargs, "save_steps", default=100),
            "remove_unused_columns": False,
            "num_generations": _get(self.kwargs, "num_generations", default=4),
            "max_completion_length": _get(
                self.kwargs, "max_completion_length", "max_new_tokens", default=256
            ),
            "temperature": _get(self.kwargs, "temperature", default=0.7),
            "top_p": _get(self.kwargs, "top_p", default=0.95),
            "beta": _get(self.kwargs, "beta", "kl_coef", default=0.04),
        }

        for k, v in config_defaults.items():
            config_kw.setdefault(k, v)

        if config_kw.get("use_vllm"):
            from agenttune.utils.optional import require_vllm

            require_vllm()
        grpo_config = GRPOConfig(**config_kw)
        logger.info(
            f"[TrlAgenticGrpo] GRPOConfig built "
            f"(output_dir={grpo_config.output_dir}, "
            f"agentic={self._agentic_mode})"
        )

        # ── Resolve / auto-build the agentic rollout_func ────────────────────
        # The documented one-liner passes `tools=env.tools` and no rollout_func.
        # The installed TRL can't run those tools natively (native `tools=` needs
        # transformers>=5 and a json-schema-able callable, not a BaseTool object),
        # so — exactly like the RLOO backend — we build a tool-calling rollout_func
        # from the tools and route generation through it. This is what makes
        # `create_agentic_trainer("grpo", tools=env.tools, ...)` actually invoke
        # the tools during the rollout on this stack. (Deliberately consistent with
        # RLOO: with tools present, GRPO always routes through agenttune's rollout,
        # never TRL-native, even on transformers>=5.)
        rollout_func = trainer_kw.get("rollout_func") or _get(self.kwargs, "rollout_func")
        if rollout_func is None and self._agentic_mode and trainer_kw.get("tools"):
            # Rollouts are appended to <output_dir>/<trajectories_file> (None disables).
            trajectories_file = _get(self.kwargs, "trajectories_file", default="trajectories.jsonl")
            rollout_func = _build_grpo_rollout_fn(
                rollout_engine=_get(self.kwargs, "rollout_engine", default=None),
                rollout_backend=_get(self.kwargs, "rollout_backend", default=None),
                tools=trainer_kw.get("tools"),
                max_steps=_get(self.kwargs, "max_steps_per_turn", default=20),
                system_prompt=_get(self.kwargs, "system_prompt", default=None),
                engine_kwargs=_get(self.kwargs, "engine_kwargs", default=None),
                on_trajectory_end=(
                    trajectory_writer(Path(output_dir) / trajectories_file)
                    if trajectories_file
                    else None
                ),
                **{
                    k: self.kwargs[k]
                    for k in ("tools_fallback_prompt", "tool_result_format")
                    if k in self.kwargs
                },
            )
            logger.info("[TrlAgenticGrpo] Built rollout_func from tools ✓")

        # ── Patch GRPOTrainer._generate_single_turn if rollout_func is set ───
        # The installed TRL does not call rollout_func inside _generate_single_turn.
        # The patch intercepts generation and routes it through rollout_func,
        # supporting both vLLM and normal transformers generation paths. The patch
        # is installed for BOTH backends: with use_vllm it first syncs the
        # colocated vLLM weights to the trainer's current step, then calls
        # rollout_func — which inside _execute_trajectory takes the
        # trainer.use_vllm branch and generates via trainer.vllm_generation
        # (trl.experimental.openenv.generate_rollout_completions). This keeps the
        # masking-aware tool-calling loop intact under vLLM; without this guard
        # removal, use_vllm=True would fall through to TRL's native
        # _generate_single_turn and bypass the agentic rollout entirely.
        if rollout_func is not None:
            _original = GRPOTrainer._generate_single_turn

            def _patched_generate_single_turn(self_trainer, prompts, *args, **kwargs):
                # Arity-tolerant: this patch is installed at the CLASS level, so it
                # persists process-wide and is reached by *every* GRPOTrainer built
                # afterward — including standard (non-agentic) ones. TRL's non-rollout
                # path calls `_generate_single_turn(prompt_ids, images, multimodal_fields)`
                # (3 positional args), so we must accept and forward the extras to the
                # original; otherwise a plain GRPO run after an agentic one raises
                # TypeError. Agentic instances set `rollout_func` and take the branch
                # below (which needs only `prompts`); everyone else falls through.
                if getattr(self_trainer, "rollout_func", None) is not None:

                    # vLLM: sync weights first before generation
                    if self_trainer.use_vllm:
                        last_loaded_step = getattr(self_trainer, "_last_loaded_step", -1)
                        if self_trainer.state.global_step != last_loaded_step:
                            with profiling_context(self_trainer, "sync_weights"):
                                self_trainer.vllm_generation.sync_weights()
                            self_trainer._last_loaded_step = self_trainer.state.global_step

                    # Call rollout_func — handles vLLM and transformers internally
                    output = self_trainer.rollout_func(prompts, self_trainer)

                    required_keys = {"prompt_ids", "completion_ids", "logprobs"}
                    missing = required_keys - output.keys()
                    if missing:
                        raise ValueError(
                            f"[TrlAgenticGrpo] rollout_func must return keys "
                            f"{sorted(missing)} in its output dict."
                        )

                    extra_fields = {k: v for k, v in output.items() if k not in required_keys}
                    return (
                        output["prompt_ids"],
                        output["completion_ids"],
                        output["logprobs"],
                        extra_fields,
                    )

                # rollout_func not set on this instance — fall through to original,
                # forwarding whatever positional/keyword extras TRL passed (images,
                # multimodal_fields, …) so the original's real signature is honored.
                return _original(self_trainer, prompts, *args, **kwargs)

            GRPOTrainer._generate_single_turn = _patched_generate_single_turn
            logger.info(
                "[TrlAgenticGrpo] Patched GRPOTrainer._generate_single_turn "
                "to support rollout_func ✓"
            )

        # ── GRPOTrainer: inject datasets + config ────────────────────────────
        trainer_kw.setdefault("args", grpo_config)
        trainer_kw.setdefault("train_dataset", self.train_dataset)
        trainer_kw.setdefault("eval_dataset", self.eval_dataset)

        if "model" not in trainer_kw:
            raise ValueError(
                "[TrlAgenticGrpo] 'model' is required. "
                "Pass model='<hf_id_or_path_or_object>' to the constructor."
            )
        if "reward_funcs" not in trainer_kw:
            raise ValueError(
                "[TrlAgenticGrpo] 'reward_funcs' is required. "
                "Pass reward_funcs=<callable_or_list> to the constructor."
            )

        if self._agentic_mode:
            tools = trainer_kw.get("tools")
            if tools:
                tool_names = [getattr(t, "__name__", str(t)) for t in tools]
                logger.info(f"[TrlAgenticGrpo] Tools registered: {tool_names}")
            if rollout_func is not None:
                logger.info("[TrlAgenticGrpo] Agentic rollout_func active")
            if trainer_kw.get("environment_factory"):
                logger.info("[TrlAgenticGrpo] environment_factory registered")

        # When a rollout_func drives generation we run the tool loop ourselves, so
        # `tools`/`rollout_func` must NOT reach GRPOTrainer — the native `tools=`
        # path raises on transformers<5 (and can't schema a BaseTool). The rollout
        # func is re-attached to the trainer instance *after* init (below), which is
        # what the _generate_single_turn patch and TRL's vLLM path read.
        if rollout_func is not None:
            trainer_kw.pop("tools", None)
            trainer_kw.pop("rollout_func", None)

        from agenttune.agentic.rollout_engines.rollout_factory import (
            ensure_lora_targets,
            ensure_processing_class,
        )

        ensure_processing_class(trainer_kw)
        # This pop is enough — pulls it out of trainer_kw so it's not passed twice
        peft_config = _resolve_peft_config(
            trainer_kw.pop("peft_config", None) or self.kwargs.get("peft_config")
        )
        peft_config = ensure_lora_targets(peft_config, trainer_kw.get("model"))
        self.trainer = GRPOTrainer(**trainer_kw, peft_config=peft_config)
        logger.info("[TrlAgenticGrpo] GRPOTrainer ready ✓")

        # Re-attach the rollout_func to the instance (we popped it from trainer_kw
        # so GRPOTrainer wouldn't route it through its native tool loop). Without
        # this, the patch's `getattr(self_trainer, "rollout_func", None)` is None
        # and generation silently falls back to a tool-free single turn.
        if rollout_func is not None:
            self.trainer.rollout_func = rollout_func

        if (
            _get(self.kwargs, "use_vllm", default=False)
            and _get(self.kwargs, "vllm_mode", default="colocate") == "colocate"
            and hasattr(self.trainer, "vllm_generation")
            and hasattr(self.trainer.vllm_generation, "llm")
            and not hasattr(self.trainer, "llm")
        ):
            self.trainer.llm = self.trainer.vllm_generation.llm
            logger.info(
                "[TrlAgenticGrpo] Applied vLLM colocate patch: "
                "trainer.llm → trainer.vllm_generation.llm"
            )

    # ------------------------------------------------------------------ #
    #  Training entry point                                                #
    # ------------------------------------------------------------------ #

    def train(self) -> dict[str, Any]:
        """Full pipeline: setup_data → setup_trainer → train → save."""
        self.setup_data()
        self.setup_trainer()

        logger.info(
            f"[TrlAgenticGrpo] ▶ Starting {'agentic ' if self._agentic_mode else ''}GRPO training …"
        )
        t0 = time.time()
        resume = _get(self.kwargs, "resume_from_checkpoint", default=False)
        self._train_result = self.trainer.train(resume_from_checkpoint=resume)
        elapsed = time.time() - t0
        logger.info(f"[TrlAgenticGrpo] ✓ Training finished in {elapsed:.1f}s")

        output_dir = _get(self.kwargs, "output_dir", default="./output/grpo_agentic")
        self.save_model(output_dir)

        return self.get_training_stats()

    # ------------------------------------------------------------------ #
    #  Save / load                                                         #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str | None = None, push_to_hub: bool = False, **extra_meta) -> str:
        """
        Save model, tokenizer, GRPOConfig (YAML), and training stats (JSON).

        Parameters
        ----------
        path        : target directory; falls back to ``output_dir`` kwarg.
        push_to_hub : push to HF Hub after saving (overrides kwarg if True).
        **extra_meta: extra key/value pairs merged into training_stats.json.

        Returns
        -------
        str : the resolved save path.
        """
        save_path = path or _get(self.kwargs, "output_dir", default="./output/grpo_agentic")
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # ── model + tokenizer via trainer (handles PEFT saving correctly) ─
        if self.trainer is not None:
            self.trainer.save_model(save_path)
            logger.info(f"[TrlAgenticGrpo] Model saved → {save_path}")

            proc = getattr(self.trainer, "processing_class", None)
            if proc is not None and hasattr(proc, "save_pretrained"):
                proc.save_pretrained(save_path)
                logger.info(f"[TrlAgenticGrpo] Tokenizer saved → {save_path}")

        # ── GRPOConfig as YAML ──────────────────────────────────────────
        config_path = Path(save_path) / "grpo_training_config.yaml"
        if self.trainer is not None and hasattr(self.trainer, "args"):
            cfg_src = self.trainer.args
            cfg_dict = cfg_src.to_dict() if hasattr(cfg_src, "to_dict") else vars(cfg_src)
        else:
            cfg_dict = self._serialisable_kwargs()

        with open(config_path, "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)
        logger.info(f"[TrlAgenticGrpo] Config saved → {config_path}")

        # ── Training stats as JSON ──────────────────────────────────────
        stats = self.get_training_stats()
        stats.update(extra_meta)
        stats_path = Path(save_path) / "training_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"[TrlAgenticGrpo] Stats saved  → {stats_path}")

        from agenttune.utils.provenance import write_provenance

        write_provenance(
            save_path,
            method="agentic.grpo",
            base_model=_get(self.kwargs, "model"),
            dataset=_get(self.kwargs, "dataset_name", "dataset"),
            dataset_config=_get(self.kwargs, "dataset_config", "config_name"),
        )

        # ── Optional Hub push ───────────────────────────────────────────
        should_push = push_to_hub or bool(_get(self.kwargs, "push_to_hub", default=False))
        if should_push and self.trainer is not None:
            logger.info("[TrlAgenticGrpo] Pushing model to Hugging Face Hub …")
            self.trainer.push_to_hub()

        return str(Path(save_path).resolve())

    def load_model(self, path: str, **kwargs) -> None:
        """
        Reload a saved model + tokenizer.

        Parameters
        ----------
        path     : directory created by :meth:`save_model`.
        **kwargs : forwarded to ``from_pretrained``
                   (e.g. device_map="auto", torch_dtype=torch.bfloat16).
        """
        from transformers import AutoTokenizer

        from agenttune.utils.model_class_resolver import resolve_model_class

        device_map = kwargs.pop("device_map", _get(self.kwargs, "device_map", default="auto"))
        trust_remote_code = kwargs.pop(
            "trust_remote_code", _get(self.kwargs, "trust_remote_code", default=False)
        )

        logger.info(f"[TrlAgenticGrpo] Loading model from: {path}")

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = resolve_model_class(path, trust_remote_code=trust_remote_code).from_pretrained(
            path,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        if self.trainer is not None:
            self.trainer.model = model
            if hasattr(self.trainer, "processing_class"):
                self.trainer.processing_class = tokenizer
        else:
            # Stash so setup_trainer can pick them up later
            self.kwargs["model"] = model
            self.kwargs["processing_class"] = tokenizer

        logger.info("[TrlAgenticGrpo] Model loaded ✓")

    # ------------------------------------------------------------------ #
    #  Stats & introspection                                               #
    # ------------------------------------------------------------------ #

    def get_training_stats(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the run."""
        tr = self._train_result
        metrics = getattr(tr, "metrics", {}) if tr else {}

        # Collect tool names for agentic runs
        tools = _get(self.kwargs, "tools")
        tool_names = [getattr(t, "__name__", str(t)) for t in tools] if tools else []

        return {
            "model": str(_get(self.kwargs, "model", default="unknown")),
            "dataset_name": _get(self.kwargs, "dataset_name", "dataset", default="unknown"),
            "output_dir": _get(self.kwargs, "output_dir", default="./output/grpo_agentic"),
            "agentic_mode": self._agentic_mode,
            "tools": tool_names,
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "eval_size": len(self.eval_dataset) if self.eval_dataset else 0,
            "final_loss": getattr(tr, "training_loss", metrics.get("train_loss")),
            "total_steps": getattr(tr, "global_step", None),
            "training_history": self.training_history,
            "metrics": metrics,
            "config_kwargs": self._serialisable_kwargs(),
        }

    def _serialisable_kwargs(self) -> dict[str, Any]:
        """Return self.kwargs with non-serialisable values replaced by their repr."""
        out: dict[str, Any] = {}
        for k, v in self.kwargs.items():
            if callable(v):
                out[k] = f"<callable: {getattr(v, '__name__', type(v).__name__)}>"
            elif isinstance(v, str | int | float | bool | list | dict | type(None)):
                out[k] = v
            else:
                out[k] = str(v)
        return out
