"""
TrlAgenticRloo
==============
Fixed version: _generate_single_turn patch handles both TRL API variants:

  Old API (pre-~0.19):  _generate_single_turn(self, prompt_ids, images, multimodal_fields)
  New API (>=~0.19):    _generate_single_turn(self, prompts)   <- single arg, raw strings

The TypeError you saw:
  TypeError: _agentic_generate_single_turn() missing 2 required positional
             arguments: 'images' and 'multimodal_fields'

was caused by the installed TRL calling  self._generate_single_turn(prompts)
(one positional arg) while the patch expected three. The fix makes all args
after the first optional and auto-detects whether it received decoded strings
or token-id lists.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def _get(kwargs: dict[str, Any], *keys, default=None):
    for k in keys:
        if k in kwargs:
            return kwargs[k]
    return default


def _split_kwargs_for_rloo(kwargs: dict[str, Any]) -> tuple[dict, dict]:
    from agenttune.utils.environment import patch_colab_outstream_close

    patch_colab_outstream_close()
    from trl import RLOOConfig, RLOOTrainer

    config_keys = set(inspect.signature(RLOOConfig.__init__).parameters) - {"self"}
    trainer_keys = set(inspect.signature(RLOOTrainer.__init__).parameters) - {"self"}
    config_kw: dict[str, Any] = {}
    trainer_kw: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in trainer_keys:
            trainer_kw[k] = v
        elif k in config_keys:
            config_kw[k] = v
    return config_kw, trainer_kw


# ─────────────────────────────────────────────────────────────────────────────
# Rollout builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_rloo_rollout_fn(
    rollout_engine=None,
    tools=None,
    max_steps: int = 20,
    system_prompt=None,
    engine_kwargs=None,
    rollout_backend=None,
    on_trajectory_end=None,
    **rollout_kwargs,
) -> Callable:
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

    _base_rollout = create_rollout_fn(
        rollout_engine=rollout_engine,
        rollout_backend=rollout_backend,
        tools=tools,
        max_steps=max_steps,
        system_prompt=system_prompt,
        engine_kwargs=engine_kwargs or {},
        on_trajectory_end=on_trajectory_end,
        **rollout_kwargs,
    )

    def rloo_rollout_fn(prompts: list[str], trainer=None) -> dict[str, Any]:
        batch = _base_rollout(prompts, trainer=trainer)
        return {
            "prompt_ids": batch["prompt_ids"],
            "completion_ids": batch["completion_ids"],
            "logprobs": batch.get("logprobs", []),
            "responses": batch.get("responses", []),
            "trajectories": batch.get("trajectories", []),
        }

    return rloo_rollout_fn


# ─────────────────────────────────────────────────────────────────────────────
# Core patch
# ─────────────────────────────────────────────────────────────────────────────


def _patch_generate_single_turn(trainer_instance, rollout_func: Callable) -> None:
    """
    Bind a new _generate_single_turn to trainer_instance that routes
    generation through rollout_func.

    Handles two TRL calling conventions so the patch is version-agnostic:

      Old TRL: _generate_single_turn(self, prompt_ids, images, multimodal_fields)
               where prompt_ids is list[list[int]]

      New TRL: _generate_single_turn(self, prompts)
               where prompts is list[str]  (already decoded)

    We make 'images' and 'multimodal_fields' optional (default None) and
    auto-detect the input type, so the same patched method works for both.
    """

    def _agentic_generate_single_turn(
        self_trainer,
        prompt_ids_or_prompts,  # list[list[int]] (old TRL) OR list[str] (new TRL)
        images=None,  # old TRL only — ignored by rollout_func
        multimodal_fields=None,  # old TRL only — ignored by rollout_func
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Version-agnostic replacement for RLOOTrainer._generate_single_turn.

        Normalises the input to plain strings, calls rollout_func, and returns
        (prompt_ids, completion_ids) — both list[list[int]].
        """
        processing_class = self_trainer.processing_class

        if not prompt_ids_or_prompts:
            return []

        first = prompt_ids_or_prompts[0]

        # ── Detect input type ─────────────────────────────────────────────
        if isinstance(first, str):
            # New TRL: already decoded strings
            raw_prompts: list[str] = list(prompt_ids_or_prompts)
            # Re-tokenise to get prompt_ids (needed for the return value)
            processing_class(text=raw_prompts)["input_ids"]

        elif isinstance(first, list | tuple) and first and isinstance(first[0], int):
            # Old TRL: list[list[int]] — decode for rollout_func
            raw_prompts = processing_class.batch_decode(
                prompt_ids_or_prompts, skip_special_tokens=True
            )
            list(prompt_ids_or_prompts)

        else:
            # Fallback — treat as token ids
            raw_prompts = processing_class.batch_decode(
                prompt_ids_or_prompts, skip_special_tokens=True
            )
            list(prompt_ids_or_prompts)

        # ── Sync vLLM weights if needed ───────────────────────────────────
        if getattr(self_trainer, "use_vllm", False):
            last_loaded = getattr(self_trainer, "_last_loaded_step", -1)
            if self_trainer.state.global_step != last_loaded:
                from trl.extras.profiling import profiling_context

                with profiling_context(self_trainer, "sync_weights"):
                    self_trainer.vllm_generation.sync_weights()
                self_trainer._last_loaded_step = self_trainer.state.global_step

        # ── Call rollout_func ─────────────────────────────────────────────
        output: dict[str, Any] = rollout_func(raw_prompts, trainer=self_trainer)

        # ── Validate output ───────────────────────────────────────────────
        required = {"prompt_ids", "completion_ids"}
        missing = required - output.keys()
        if missing:
            raise ValueError(
                f"[TrlAgenticRloo] rollout_func must return keys "
                f"{sorted(missing)} but they are missing from its output."
            )

        return output["completion_ids"]

    trainer_instance._generate_single_turn = types.MethodType(
        _agentic_generate_single_turn, trainer_instance
    )
    logger.info(
        "[TrlAgenticRloo] Patched RLOOTrainer._generate_single_turn "
        "to use rollout_func (version-agnostic) ✓"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class TrlAgenticRloo:
    """
    kwargs-driven RLOO trainer supporting both standard and agentic rollouts.
    See module docstring for full parameter documentation.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.trainer: Any = None
        self.train_dataset = None
        self.eval_dataset = kwargs.get("eval_dataset", None)
        self._train_result = None
        self.training_history: list[dict] = []

        self._agentic_mode = bool(
            kwargs.get("tools") or kwargs.get("rollout_engine") or kwargs.get("rollout_func")
        )
        logger.info(f"[TrlAgenticRloo] Initialized (agentic={self._agentic_mode})")

    # ── Data ──────────────────────────────────────────────────────────────────

    def setup_data(self) -> None:
        format_fn = _get(self.kwargs, "format_fn", default=None)
        format_batched = _get(self.kwargs, "format_batched", default=False)
        format_remove_columns = _get(self.kwargs, "format_remove_columns", default=None)

        def _apply_fmt(ds):
            if format_fn is None or ds is None:
                return ds
            kw: dict[str, Any] = {"batched": format_batched}
            if format_remove_columns:
                kw["remove_columns"] = format_remove_columns
            return ds.map(format_fn, **kw)

        if "train_dataset" in self.kwargs:
            self.train_dataset = _apply_fmt(self.kwargs["train_dataset"])
            if "eval_dataset" in self.kwargs:
                self.eval_dataset = _apply_fmt(self.kwargs["eval_dataset"])
            logger.info(
                f"[TrlAgenticRloo] Using pre-loaded dataset "
                f"({len(self.train_dataset)} examples)"
            )
            return

        from agenttune.data.manager import DataManager

        dataset_name = _get(self.kwargs, "dataset_name", "dataset", default="trl-lib/tldr")
        config_name = _get(self.kwargs, "dataset_config", "config_name", default=None)
        split = _get(self.kwargs, "split", default=None)
        max_samples = _get(self.kwargs, "max_samples", default=None)
        dm_cfg = _get(self.kwargs, "data_manager_config", default=None)

        dm_kwargs: dict[str, Any] = {
            "task_type": "rloo",
            "system_prompt": _get(self.kwargs, "system_prompt", default=None),
            "tokenizer": _get(self.kwargs, "processing_class", default=None),
            "enable_thinking": _get(self.kwargs, "enable_thinking", default=False),
            "column_mapping": _get(self.kwargs, "column_mapping", default=None),
            "processing_fn": _get(self.kwargs, "processing_fn", default=None),
            "processing_batched": _get(self.kwargs, "processing_batched", default=False),
            "max_samples": max_samples,
        }
        if isinstance(dm_cfg, dict):
            dm_kwargs.update(dm_cfg)

        manager = DataManager(**dm_kwargs)
        dataset_dict = manager.load_dataset(dataset_name, config_name=config_name, split=split)

        train_ds = _apply_fmt(dataset_dict.get("train", None))
        eval_ds = _apply_fmt(dataset_dict.get("validation", None))

        if train_ds and max_samples and len(train_ds) > max_samples:
            train_ds = train_ds.select(range(max_samples))

        self.train_dataset = train_ds
        if self.eval_dataset is None:
            self.eval_dataset = eval_ds

        logger.info(f"[TrlAgenticRloo] Train: {len(self.train_dataset)} examples")

    # ── Trainer setup ─────────────────────────────────────────────────────────

    def setup_trainer(self) -> None:
        from trl import RLOOConfig, RLOOTrainer

        config_kw, trainer_kw = _split_kwargs_for_rloo(self.kwargs)

        output_dir = _get(self.kwargs, "output_dir", default="./output/rloo_agentic")
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
            "seed": _get(self.kwargs, "seed", default=42),
            "logging_steps": _get(self.kwargs, "logging_steps", default=10),
            "save_steps": _get(self.kwargs, "save_steps", default=100),
            "num_generations": _get(self.kwargs, "num_generations", default=4),
            "max_completion_length": _get(
                self.kwargs, "max_completion_length", "max_new_tokens", default=256
            ),
            "temperature": _get(self.kwargs, "temperature", default=0.7),
            "top_p": _get(self.kwargs, "top_p", default=0.95),
            "beta": _get(self.kwargs, "beta", "kl_coef", default=0.05),
        }
        for k, v in config_defaults.items():
            config_kw.setdefault(k, v)

        if config_kw.get("use_vllm"):
            from agenttune.utils.optional import require_vllm

            require_vllm()
        rloo_config = RLOOConfig(**config_kw)
        logger.info(
            f"[TrlAgenticRloo] RLOOConfig built "
            f"(output_dir={rloo_config.output_dir}, agentic={self._agentic_mode})"
        )

        if "model" not in trainer_kw and "model" not in self.kwargs:
            raise ValueError("[TrlAgenticRloo] 'model' is required.")
        if "reward_funcs" not in trainer_kw and "reward_funcs" not in self.kwargs:
            raise ValueError("[TrlAgenticRloo] 'reward_funcs' is required.")

        rollout_func: Callable | None = _get(self.kwargs, "rollout_func", default=None)

        if rollout_func is None and self._agentic_mode:
            from agenttune.agentic.trajectory.dataset import trajectory_writer

            # Rollouts are appended to <output_dir>/<trajectories_file> (None disables).
            trajectories_file = _get(self.kwargs, "trajectories_file", default="trajectories.jsonl")
            rollout_func = _build_rloo_rollout_fn(
                rollout_engine=_get(self.kwargs, "rollout_engine", default=None),
                rollout_backend=_get(self.kwargs, "rollout_backend", default=None),
                tools=_get(self.kwargs, "tools", default=None),
                max_steps=_get(self.kwargs, "max_steps_per_turn", "max_steps", default=20),
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
            logger.info("[TrlAgenticRloo] Built rollout_func from tools/engine ✓")

        trainer_kw.setdefault("args", rloo_config)
        trainer_kw.setdefault("train_dataset", self.train_dataset)
        trainer_kw.setdefault("eval_dataset", self.eval_dataset)

        for agentic_key in (
            "tools",
            "rollout_func",
            "rollout_engine",
            "rollout_backend",
            "max_steps_per_turn",
            "max_steps",
            "engine_kwargs",
        ):
            trainer_kw.pop(agentic_key, None)

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

        self.trainer = RLOOTrainer(
            **trainer_kw,
            peft_config=peft_config,
        )
        logger.info("[TrlAgenticRloo] RLOOTrainer instantiated ✓")

        if rollout_func is not None:
            self.trainer.rollout_func = rollout_func
            _patch_generate_single_turn(self.trainer, rollout_func)

            tool_names = [
                getattr(t, "__name__", str(t)) for t in (_get(self.kwargs, "tools") or [])
            ]
            logger.info(
                f"[TrlAgenticRloo] Agentic patch applied. "
                f"Tools: {tool_names or '(none — custom rollout_func)'}"
            )

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self) -> dict[str, Any]:
        self.setup_data()
        self.setup_trainer()
        logger.info(
            f"[TrlAgenticRloo] Starting "
            f"{'agentic ' if self._agentic_mode else ''}RLOO training ..."
        )
        t0 = time.time()
        self._train_result = self.trainer.train()
        logger.info(f"[TrlAgenticRloo] Done in {time.time()-t0:.1f}s")
        self.save_model(_get(self.kwargs, "output_dir", default="./output/rloo_agentic"))
        return self.get_training_stats()

    # ── Save / Load / Stats ───────────────────────────────────────────────────

    def save_model(self, path=None, push_to_hub=False, **extra_meta) -> str:
        save_path = path or _get(self.kwargs, "output_dir", default="./output/rloo_agentic")
        Path(save_path).mkdir(parents=True, exist_ok=True)

        if self.trainer is not None:
            self.trainer.save_model(save_path)
            proc = getattr(self.trainer, "processing_class", None)
            if proc and hasattr(proc, "save_pretrained"):
                proc.save_pretrained(save_path)

        cfg_dict = (
            self.trainer.args.to_dict()
            if self.trainer and hasattr(self.trainer.args, "to_dict")
            else self._serialisable_kwargs()
        )
        with open(Path(save_path) / "rloo_config.yaml", "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)

        stats = {**self.get_training_stats(), **extra_meta}
        with open(Path(save_path) / "training_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

        from agenttune.utils.provenance import write_provenance

        write_provenance(
            save_path,
            method="agentic.rloo",
            base_model=_get(self.kwargs, "model"),
            dataset=_get(self.kwargs, "dataset_name", "dataset"),
            dataset_config=_get(self.kwargs, "dataset_config", "config_name"),
        )

        if push_to_hub or bool(_get(self.kwargs, "push_to_hub", default=False)):
            self.trainer.push_to_hub()

        return str(Path(save_path).resolve())

    def load_model(self, path: str, **kwargs) -> None:
        from transformers import AutoTokenizer

        from agenttune.utils.model_class_resolver import resolve_model_class

        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = resolve_model_class(
            path, trust_remote_code=kwargs.get("trust_remote_code", False)
        ).from_pretrained(path, device_map=kwargs.pop("device_map", "auto"), **kwargs)
        if self.trainer:
            self.trainer.model = model
            self.trainer.processing_class = tokenizer
        else:
            self.kwargs["model"] = model
            self.kwargs["processing_class"] = tokenizer

    def get_training_stats(self) -> dict[str, Any]:
        tr = self._train_result
        metrics = getattr(tr, "metrics", {}) if tr else {}
        tools = _get(self.kwargs, "tools")
        return {
            "model": str(_get(self.kwargs, "model", default="unknown")),
            "output_dir": _get(self.kwargs, "output_dir", default="./output/rloo_agentic"),
            "agentic_mode": self._agentic_mode,
            "tools": [getattr(t, "__name__", str(t)) for t in tools] if tools else [],
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "final_loss": getattr(tr, "training_loss", metrics.get("train_loss")),
            "total_steps": getattr(tr, "global_step", None),
            "training_history": self.training_history,
            "metrics": metrics,
            "config_kwargs": self._serialisable_kwargs(),
        }

    def _serialisable_kwargs(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self.kwargs.items():
            if callable(v):
                out[k] = f"<callable: {getattr(v, '__name__', type(v).__name__)}>"
            elif isinstance(v, str | int | float | bool | list | dict | type(None)):
                out[k] = v
            else:
                out[k] = str(v)
        return out
