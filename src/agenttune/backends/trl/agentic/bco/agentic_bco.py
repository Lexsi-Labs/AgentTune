# agenttune/agentic/trainers/agentic_bco.py
"""
TrlAgenticBCO
=============
A kwargs-driven wrapper around TRL's BCOTrainer with full support for
agentic (multi-turn tool-calling) rollouts in ONLINE mode.

Uses BCOTrainer from trl.experimental.bco — NOT DPOTrainer.
Fresh desirable/undesirable completions are generated every epoch via a
rollout_fn, delegated to create_rollout_fn from rollout_factory.

Three operating modes
─────────────────────
MODE 1 — Standard offline BCO (complete dataset, no live generation):
    Dataset must have ``prompt``, ``completion``, ``label`` columns.
    reward_funcs is NOT required. If passed it is silently ignored.

    TrlAgenticBCO(model=..., train_dataset=full_ds, beta=0.1)

MODE 2 — Online rollout BCO (use_rollouts=True, no tools):
    Dataset needs only a ``prompt`` column (or a pool can be passed via
    prompt_pool). Completions are generated every epoch; a reward_fn
    thresholds scores into desirable (True) / undesirable (False) labels.

    TrlAgenticBCO(
        model=..., reward_funcs=my_reward,
        train_dataset=prompt_ds,
        use_rollouts=True,
        score_threshold=0.5,
    )

    Also auto-triggered when ``tools`` or ``rollout_engine`` are passed.

MODE 3 — Agentic tool-calling rollout BCO:
    Same as Mode 2 but generation goes through multi-turn tool calls.

    TrlAgenticBCO(
        model=..., reward_funcs=my_reward,
        tools=[calculator, web_search],
        train_dataset=prompt_ds,
        score_threshold=0.5,
    )

NOTE — passing reward_funcs with a complete dataset:
    reward_funcs alone does NOT enable live generation. Your static
    prompt/completion/label data is used as-is, exactly like standard BCO.

    TrlAgenticBCO(
        model=..., reward_funcs=eval_reward,   # safe — no rollouts triggered
        train_dataset=full_ds,
    )
"""

from __future__ import annotations

import inspect
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get(kwargs: dict[str, Any], *keys, default=None):
    """Return the first matching key found in kwargs, else default."""
    for k in keys:
        if k in kwargs:
            return kwargs[k]
    return default


def _split_kwargs_for_bco(kwargs: dict[str, Any]) -> tuple[dict, dict]:
    """Split kwargs into (bco_config_kwargs, bco_trainer_kwargs)."""
    from agenttune.utils.environment import patch_colab_outstream_close

    patch_colab_outstream_close()
    from trl.experimental.bco.bco_config import BCOConfig
    from trl.experimental.bco.bco_trainer import BCOTrainer

    config_keys = set(inspect.signature(BCOConfig.__init__).parameters) - {"self"}
    trainer_keys = set(inspect.signature(BCOTrainer.__init__).parameters) - {"self"}

    config_kw: dict[str, Any] = {}
    trainer_kw: dict[str, Any] = {}

    for k, v in kwargs.items():
        if k in trainer_keys:
            trainer_kw[k] = v
        elif k in config_keys:
            config_kw[k] = v
        # else: agentic/meta param — intentionally ignored

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


# ─────────────────────────────────────────────────────────────────────────────
# OnlineBCOTrainer — BCOTrainer subclass with per-epoch rollout regeneration
# ─────────────────────────────────────────────────────────────────────────────


class _OnlineBCOTrainer:
    """
    Mixin/factory that builds a BCOTrainer subclass with online rollout support.

    We cannot import BCOTrainer at module load time (optional dependency), so
    the actual subclass is created lazily inside build().
    """

    @staticmethod
    def build(rollout_fn, prompt_pool, prompts_per_epoch, score_threshold, num_generations):
        """
        Return a BCOTrainer subclass configured for online generation.

        Parameters mirror OnlineBCOTrainer.__init__ from the reference
        implementation and are baked in at class-creation time so the
        resulting class is a plain BCOTrainer drop-in.
        """
        from trl.experimental.bco.bco_trainer import BCOTrainer

        class OnlineBCOTrainer(BCOTrainer):
            """BCOTrainer that regenerates its dataset before every epoch."""

            def __init__(self, *args, **kwargs):
                # Stash online params before super().__init__ touches them
                self._rollout_fn = rollout_fn
                self._prompt_pool = prompt_pool
                self._prompts_per_epoch = prompts_per_epoch
                self._score_threshold = score_threshold
                self._num_generations = num_generations
                self._epoch_counter = 0
                # Guard: True while _generate_bco_dataset() is running so that
                # the get_train_dataloader() call inside super().__init__ does
                # NOT trigger generation before the model is ready.
                self._generating = False
                self._mixed = None
                self._skip = False

                super().__init__(*args, **kwargs)

            # ── Generation ────────────────────────────────────────────────
            def _generate_bco_dataset(self) -> Dataset:
                self._epoch_counter += 1
                logger.info(f"[OnlineBCO] Generating data for epoch {self._epoch_counter}…")

                # Switch to eval mode for generation (no dropout / no grad)
                was_training = self.model.training
                self.model.eval()
                if hasattr(self, "model_wrapped") and self.model_wrapped is not self.model:
                    self.model_wrapped.eval()

                try:
                    # Sample prompts and expand for num_generations rollouts each
                    sampled = random.choices(self._prompt_pool, k=self._prompts_per_epoch)
                    expanded_prompts = [p for p in sampled for _ in range(self._num_generations)]

                    # Run rollout — passes trainer=self so _gen() uses this
                    # model's weights, tokenizer, and device
                    rollout_output = self._rollout_fn(
                        prompts=expanded_prompts,
                        trainer=self,
                    )

                    responses = rollout_output["responses"]  # list[str]
                    rewards = rollout_output["rewards"]  # list[float]
                    if rewards:
                        logger.info(
                            f"[OnlineBCO] reward min={min(rewards):.3f} " f"max={max(rewards):.3f}"
                        )

                    def _prompt_text(p):
                        return p["prompt"] if isinstance(p, dict) else p

                    # Threshold rewards → BCO labels (True = desirable)
                    raw_data = [
                        {
                            "prompt": _prompt_text(prompt),
                            "completion": response,
                            "label": bool(score >= self._score_threshold),
                        }
                        for prompt, response, score in zip(
                            expanded_prompts, responses, rewards, strict=False
                        )
                    ]

                    n_desirable = sum(1 for d in raw_data if d["label"])
                    n_undesirable = sum(1 for d in raw_data if not d["label"])
                    logger.info(
                        f"[OnlineBCO] {len(raw_data)} examples: "
                        f"{n_desirable} desirable / {n_undesirable} undesirable"
                    )

                    if n_desirable == 0 or n_undesirable == 0:
                        logger.warning(
                            "[OnlineBCO] all labels the same; skipping "
                            "(not inventing a rank split)."
                        )
                        return Dataset.from_list([])

                    return Dataset.from_list(raw_data)

                finally:
                    # Always restore training mode before returning
                    if was_training:
                        self.model.train()
                        if hasattr(self, "model_wrapped") and self.model_wrapped is not self.model:
                            self.model_wrapped.train()

            # ── Tokenisation pipeline (mirrors BCOTrainer.__init__ flow) ──
            def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
                from accelerate import PartialState
                from trl.data_utils import (
                    maybe_apply_chat_template,
                    maybe_extract_prompt,
                    maybe_unpair_preference_dataset,
                )
                from trl.experimental.bco.bco_trainer import _process_tokens, _tokenize

                args = self.args
                with PartialState().main_process_first():
                    dataset = dataset.map(
                        maybe_extract_prompt,
                        num_proc=args.dataset_num_proc,
                        desc="Extracting prompt",
                    )
                    dataset = maybe_unpair_preference_dataset(
                        dataset, args.dataset_num_proc, desc="Unpairing"
                    )
                    dataset = dataset.map(
                        maybe_apply_chat_template,
                        fn_kwargs={"processing_class": self.processing_class},
                        num_proc=args.dataset_num_proc,
                    )
                    dataset = dataset.map(
                        _tokenize,
                        batched=True,
                        fn_kwargs={
                            "tokenizer": self.processing_class,
                            "embedding_tokenizer": self.embedding_tokenizer,
                        },
                        num_proc=args.dataset_num_proc,
                        desc="Tokenizing",
                    )
                    dataset = dataset.map(
                        _process_tokens,
                        fn_kwargs={
                            "prefix": "",
                            "is_encoder_decoder": self.is_encoder_decoder,
                            "tokenizer": self.processing_class,
                            "max_length": self.max_length,
                            "truncation_mode": getattr(self, "truncation_mode", "keep_end"),
                            "max_completion_length": self.max_completion_length,
                        },
                        num_proc=args.dataset_num_proc,
                        desc="Processing tokens",
                    )
                return dataset

            # ── get_train_dataloader: regenerate on every call except init ─
            def get_train_dataloader(self):
                # During __init__ the model/accelerator aren't ready yet —
                # fall through to parent so it can build from the seed dataset.
                if self._generating or not hasattr(self, "accelerator"):
                    return super().get_train_dataloader()

                self._generating = True
                try:
                    raw_dataset = self._generate_bco_dataset()
                    if len(raw_dataset) == 0:
                        logger.warning("[OnlineBCO] no mixed labels this epoch; skipping")
                        if self._mixed is not None:
                            self.train_dataset = self._mixed
                            self._skip = False
                        else:
                            self._skip = True
                        return super().get_train_dataloader()
                    self.train_dataset = self._preprocess_dataset(raw_dataset)
                    self._mixed = self.train_dataset
                    self._skip = False
                    logger.info(f"[OnlineBCO] Dataloader ready: {len(self.train_dataset)} examples")
                    return super().get_train_dataloader()
                finally:
                    self._generating = False

            # ── _run_epoch: regenerate data before epoch 1+ ───────────────
            def _run_epoch(
                self,
                model,
                epoch,
                train_dataloader,
                steps_in_epoch,
                num_update_steps_per_epoch,
                trial,
                ignore_keys_for_eval,
                start_time,
                resume_from_checkpoint,
                epochs_trained,
                steps_trained_in_current_epoch,
            ):
                # epoch 0 already received fresh data from get_train_dataloader()
                # called inside _inner_training_loop; epoch 1+ regenerate.
                if epoch > epochs_trained:
                    logger.info(f"[OnlineBCO] Epoch {epoch}: regenerating with updated model…")
                    train_dataloader = self.get_train_dataloader()
                    steps_in_epoch = len(train_dataloader)
                    num_update_steps_per_epoch = max(
                        steps_in_epoch // self.args.gradient_accumulation_steps, 1
                    )

                super()._run_epoch(
                    model=model,
                    epoch=epoch,
                    train_dataloader=train_dataloader,
                    steps_in_epoch=steps_in_epoch,
                    num_update_steps_per_epoch=num_update_steps_per_epoch,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                    start_time=start_time,
                    resume_from_checkpoint=resume_from_checkpoint,
                    epochs_trained=epochs_trained,
                    steps_trained_in_current_epoch=steps_trained_in_current_epoch,
                )

            def training_step(self, *args, **kwargs):
                if self._skip:
                    import torch

                    logger.warning("[OnlineBCO] all labels the same; skipping step")
                    model = args[0] if args else kwargs.get("model")
                    loss = torch.zeros(
                        (), device=next(model.parameters()).device, requires_grad=True
                    )
                    self.accelerator.backward(loss)
                    return loss.detach() / self.args.gradient_accumulation_steps
                return super().training_step(*args, **kwargs)

        return OnlineBCOTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class TrlAgenticBCO:
    """
    Online/offline BCO trainer with optional agentic rollouts.

    See module docstring for the three operating modes.

    Key parameters
    --------------
    model               : str or PreTrainedModel
    ref_model           : str or PreTrainedModel (optional; BCO reference model)
    reward_funcs        : callable(prompts, responses) -> list[float]
                          Required when use_rollouts=True.
                          Safe to pass with a complete dataset when
                          use_rollouts=False — will NOT trigger rollouts.
    train_dataset       : HF Dataset
                          Complete mode : needs prompt / completion / label.
                          Rollout mode  : needs only prompt (or use prompt_pool).
    eval_dataset        : optional HF Dataset
    prompt_pool         : list[str] — explicit prompt pool for rollout sampling.
                          Defaults to unique prompts extracted from train_dataset.
    use_rollouts        : bool — explicit flag to enable live generation.
                          Default: auto-inferred. True only when tools or
                          rollout_engine are present, NOT merely because
                          reward_funcs was supplied.
    num_generations     : int >= 1 (rollouts per prompt per epoch, default 2)
    prompts_per_epoch   : int (prompts sampled per epoch, default 16)
    score_threshold     : float (reward cutoff for desirable label, default 0.5)
    tools               : list of tool callables (auto-enables rollouts)
    rollout_engine      : pre-built external RolloutEngine
    max_steps_per_turn  : int (max tool-call steps per turn, agentic only)
    system_prompt       : str (prepended to every prompt)
    beta                : float (BCO beta, default 0.1)
    max_length          : int (max prompt + completion tokens, default 512)
    output_dir          : str
    embedding_func      : optional callable for UDM density estimation
    embedding_tokenizer : optional tokenizer for embedding_func
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = kwargs.get("eval_dataset", None)
        self._train_result = None
        self.training_history: list[dict] = []

        # ── Resolve use_rollouts ──────────────────────────────────────────
        # Priority:
        #   1. Explicit use_rollouts=True/False → always honoured.
        #   2. tools / rollout_engine present   → always rollouts.
        #   3. reward_funcs present             → tentatively True;
        #      setup_data() will flip it if the dataset already has all
        #      three required BCO columns (complete dataset case).
        #   4. None of the above               → False (standard BCO).
        explicit = kwargs.get("use_rollouts", None)
        if explicit is not None:
            self._use_rollouts = bool(explicit)
            self._use_rollouts_explicit = True
        else:
            self._use_rollouts_explicit = False
            self._use_rollouts = bool(
                kwargs.get("tools")
                or kwargs.get("rollout_engine")
                or kwargs.get("reward_funcs")  # tentative — refined in setup_data
            )

        self._agentic_mode = self._use_rollouts

        logger.info(
            f"[TrlAgenticBCO] Initialized "
            f"(use_rollouts={self._use_rollouts} [tentative until setup_data], "
            f"explicit={self._use_rollouts_explicit})"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────────────────────────────

    def setup_data(self) -> None:
        """
        Load and prepare training data.

        PATH 1 — train_dataset passed directly.
        PATH 2 — DataManager loads from HF Hub / local path.

        Dummy completion/label columns are injected ONLY when
        use_rollouts=True AND columns are missing, so BCOTrainer.__init__
        can tokenise without crashing. They are replaced before epoch 0
        by the first call to get_train_dataloader().
        """
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

        # ── PATH 1: pre-loaded ────────────────────────────────────────────
        if "train_dataset" in self.kwargs:
            self.train_dataset = _apply_fmt(self.kwargs["train_dataset"])
            if "eval_dataset" in self.kwargs:
                self.eval_dataset = _apply_fmt(self.kwargs["eval_dataset"])
            logger.info(
                f"[TrlAgenticBCO] Using pre-loaded dataset " f"({len(self.train_dataset)} examples)"
            )
        else:
            # ── PATH 2: DataManager ───────────────────────────────────────
            try:
                from agenttune.data.manager import DataManager
            except ImportError:
                raise ImportError(  # noqa: B904
                    "agenttune DataManager not found. " "Pass train_dataset directly instead."
                )

            dataset_name = _get(self.kwargs, "dataset_name", "dataset", default="trl-lib/tldr")
            config_name = _get(self.kwargs, "dataset_config", "config_name", default=None)
            split = _get(self.kwargs, "split", default=None)
            max_samples = _get(self.kwargs, "max_samples", default=None)
            dm_cfg = _get(self.kwargs, "data_manager_config", default=None)

            dm_kwargs = {
                "task_type": "bco",
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

            logger.info(f"[TrlAgenticBCO] Train: {len(self.train_dataset)} examples")
            if self.eval_dataset:
                logger.info(f"[TrlAgenticBCO] Eval : {len(self.eval_dataset)} examples")

        # ── Refine _use_rollouts now that we have the actual dataset ──────
        # If use_rollouts was NOT set explicitly and the dataset already has
        # all three BCO columns, treat it as a complete dataset and disable
        # rollouts — even if reward_funcs was passed.
        if not self._use_rollouts_explicit:
            ds_cols = set(self.train_dataset.column_names)
            has_complete = {"prompt", "completion", "label"}.issubset(ds_cols)
            has_agentic = bool(self.kwargs.get("tools") or self.kwargs.get("rollout_engine"))
            if has_complete and not has_agentic:
                if self._use_rollouts:
                    logger.info(
                        "[TrlAgenticBCO] Dataset has 'prompt'/'completion'/'label' columns — "
                        "disabling rollouts (use use_rollouts=True to override)."
                    )
                self._use_rollouts = False
                self._agentic_mode = False

        logger.info(f"[TrlAgenticBCO] use_rollouts finalised → {self._use_rollouts}")

        # ── Build prompt pool for rollout mode ────────────────────────────
        if self._use_rollouts:
            explicit_pool = _get(self.kwargs, "prompt_pool", default=None)
            if explicit_pool:
                self._prompt_pool = list(explicit_pool)
                logger.info(
                    f"[TrlAgenticBCO] Using explicit prompt_pool "
                    f"({len(self._prompt_pool)} prompts)"
                )
            else:
                # Any dataset column beyond "prompt" (not just "answer") -- e.g. a
                # custom gold_answer/gold_chunk_ids a reward_fn depends on. Sampling
                # bare prompt strings (the old else branch) silently dropped every
                # other column, starving reward_fn of anything but the prompt text.
                extra_cols = [c for c in self.train_dataset.column_names if c != "prompt"]
                if extra_cols:
                    seen = set()
                    pool = []
                    for row in self.train_dataset:
                        p = row["prompt"]
                        if p not in seen:
                            seen.add(p)
                            pool.append({"prompt": p, **{c: row[c] for c in extra_cols}})
                    self._prompt_pool = pool
                else:
                    self._prompt_pool = list(set(self.train_dataset["prompt"]))
                logger.info(
                    f"[TrlAgenticBCO] Extracted prompt_pool from dataset "
                    f"({len(self._prompt_pool)} unique prompts)"
                )

            # TRL BCOTrainer tokenises completion/label at init. Live epochs
            # replace the dataset. Do not add rows or invent an opposite label.
            n = len(self.train_dataset)
            if "completion" not in self.train_dataset.column_names:
                # "" tokenises to empty answer_input_ids; TRL _process_tokens indexes [-1].
                self.train_dataset = self.train_dataset.add_column(
                    "completion", list(self.train_dataset["prompt"])
                )
            if "label" not in self.train_dataset.column_names:
                self.train_dataset = self.train_dataset.add_column("label", [False] * n)
            if self.eval_dataset is not None:
                m = len(self.eval_dataset)
                if "completion" not in self.eval_dataset.column_names:
                    self.eval_dataset = self.eval_dataset.add_column(
                        "completion", list(self.eval_dataset["prompt"])
                    )
                if "label" not in self.eval_dataset.column_names:
                    self.eval_dataset = self.eval_dataset.add_column("label", [False] * m)

    # ─────────────────────────────────────────────────────────────────────
    # Trainer setup
    # ─────────────────────────────────────────────────────────────────────

    def setup_trainer(self) -> None:
        from trl.experimental.bco.bco_config import BCOConfig
        from trl.experimental.bco.bco_trainer import BCOTrainer

        from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

        config_kw, trainer_kw = _split_kwargs_for_bco(self.kwargs)
        peft_config = _resolve_peft_config(
            trainer_kw.pop("peft_config", None) or self.kwargs.get("peft_config")
        )

        # ── Config defaults ───────────────────────────────────────────────
        output_dir = _get(self.kwargs, "output_dir", default="./output/bco_agentic")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        config_defaults: dict[str, Any] = {
            "output_dir": output_dir,
            "num_train_epochs": _get(self.kwargs, "num_train_epochs", "epochs", default=3),
            "per_device_train_batch_size": _get(
                self.kwargs, "per_device_train_batch_size", "batch_size", default=2
            ),
            "per_device_eval_batch_size": _get(
                self.kwargs, "per_device_eval_batch_size", default=2
            ),
            "gradient_accumulation_steps": _get(
                self.kwargs, "gradient_accumulation_steps", default=1
            ),
            "learning_rate": _get(self.kwargs, "learning_rate", "lr", default=1e-5),
            "warmup_steps": _get(self.kwargs, "warmup_steps", default=0),
            "max_grad_norm": _get(self.kwargs, "max_grad_norm", default=1.0),
            "seed": _get(self.kwargs, "seed", default=42),
            "logging_steps": _get(self.kwargs, "logging_steps", default=10),
            "save_strategy": _get(self.kwargs, "save_strategy", default="epoch"),
            # Default to "epoch" only when an eval_dataset is actually available — otherwise
            # transformers' Trainer._validate_args() raises ("eval_strategy set but no
            # eval_dataset"), which used to happen for any offline-mode call that doesn't
            # pass eval_dataset explicitly (the documented Quick Start usage).
            "eval_strategy": _get(
                self.kwargs, "eval_strategy", default=("epoch" if self.eval_dataset else "no")
            ),
            "remove_unused_columns": False,
            "beta": _get(self.kwargs, "beta", default=0.1),
            "max_length": _get(self.kwargs, "max_length", default=512),
            "truncation_mode": _get(self.kwargs, "truncation_mode", default="keep_end"),
            "disable_dropout": _get(self.kwargs, "disable_dropout", default=True),
            "report_to": _get(self.kwargs, "report_to", default="none"),
            "dataset_num_proc": _get(self.kwargs, "dataset_num_proc", default=None),
            "precompute_ref_log_probs": _get(
                self.kwargs, "precompute_ref_log_probs", default=False
            ),
            "generate_during_eval": _get(self.kwargs, "generate_during_eval", default=False),
            # UDM density-ratio params (only relevant with embedding_func)
            "prompt_sample_size": _get(self.kwargs, "prompt_sample_size", default=1024),
            "min_density_ratio": _get(self.kwargs, "min_density_ratio", default=0.5),
            "max_density_ratio": _get(self.kwargs, "max_density_ratio", default=10.0),
        }

        if _get(self.kwargs, "max_steps") is not None:
            config_defaults["max_steps"] = _get(self.kwargs, "max_steps")

        # max_completion_length only for encoder-decoder models
        if _get(self.kwargs, "max_completion_length") is not None:
            config_defaults["max_completion_length"] = _get(self.kwargs, "max_completion_length")

        for k, v in config_defaults.items():
            config_kw.setdefault(k, v)

        # Drop any keys the installed BCOConfig version doesn't accept (TRL
        # version drift, e.g. truncation_mode was removed/renamed). Keeps the
        # wrapper resilient across TRL releases instead of hard-crashing.
        import inspect as _inspect

        _bco_cfg_keys = set(_inspect.signature(BCOConfig.__init__).parameters) - {"self"}
        _dropped = [k for k in list(config_kw) if k not in _bco_cfg_keys]
        if _dropped:
            logger.warning(
                "[TrlAgenticBCO] Dropping BCOConfig kwargs not supported by installed "
                "TRL version: %s",
                _dropped,
            )
            for k in _dropped:
                config_kw.pop(k, None)

        bco_config = BCOConfig(**config_kw)
        logger.info(
            f"[TrlAgenticBCO] BCOConfig built "
            f"(output_dir={bco_config.output_dir}, use_rollouts={self._use_rollouts})"
        )

        # ── Validate required args ────────────────────────────────────────
        if "model" not in trainer_kw and "model" not in self.kwargs:
            raise ValueError("[TrlAgenticBCO] 'model' is required.")

        # ── Resolve model + processing_class from string if needed ────────
        # BCOTrainer requires a PreTrainedModel and a processing_class
        # (tokenizer). When the caller passes model="some/hf-id" we load
        # both here so they are concrete objects before __init__ is called.
        # This mirrors what the raw OnlineBCOTrainer script does explicitly.
        raw_model = trainer_kw.get("model") or self.kwargs.get("model")
        if isinstance(raw_model, str):
            from transformers import AutoTokenizer

            from agenttune.utils.model_class_resolver import resolve_model_class

            model_name = raw_model
            trust_remote_code = _get(self.kwargs, "trust_remote_code", default=False)
            device_map = _get(self.kwargs, "device_map", default=None)

            logger.info(f"[TrlAgenticBCO] Loading model from '{model_name}' …")
            load_kw: dict[str, Any] = {"trust_remote_code": trust_remote_code}
            if device_map is not None:
                load_kw["device_map"] = device_map

            loaded_model = resolve_model_class(
                model_name, trust_remote_code=trust_remote_code
            ).from_pretrained(model_name, **load_kw)
            trainer_kw["model"] = loaded_model

            # Load ref_model from string if provided as one
            raw_ref = trainer_kw.get("ref_model") or self.kwargs.get("ref_model")
            if isinstance(raw_ref, str):
                logger.info(f"[TrlAgenticBCO] Loading ref_model from '{raw_ref}' …")
                trainer_kw["ref_model"] = resolve_model_class(
                    raw_ref, trust_remote_code=trust_remote_code
                ).from_pretrained(raw_ref, **load_kw)
            elif raw_ref is not None:
                trainer_kw["ref_model"] = raw_ref

            # Build tokenizer only if processing_class was not supplied
            if "processing_class" not in trainer_kw:
                logger.info(f"[TrlAgenticBCO] Loading tokenizer from '{model_name}' …")
                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                trainer_kw["processing_class"] = tok
                logger.info("[TrlAgenticBCO] Tokenizer loaded and set as processing_class ✓")

        # Guard: processing_class must be present — BCOTrainer raises
        # "max_length or a processing_class must be specified" without it.
        if "processing_class" not in trainer_kw:
            raise ValueError(
                "[TrlAgenticBCO] 'processing_class' (tokenizer) is required. "
                "Pass it explicitly or let the trainer load it by passing model as a string."
            )

        # ── Build rollout_fn and choose trainer class ─────────────────────
        if self._use_rollouts:
            raw_reward = trainer_kw.pop("reward_funcs", None) or self.kwargs.get("reward_funcs")
            if raw_reward is None:
                raise ValueError(
                    "[TrlAgenticBCO] 'reward_funcs' is required when use_rollouts=True "
                    "to score completions into desirable/undesirable labels."
                )
            from agenttune.agentic.rewards.composite import combine_rewards

            primary_reward = combine_rewards(
                raw_reward,
                weights=_get(self.kwargs, "reward_weights", default=None),
            )

            num_generations = int(_get(self.kwargs, "num_generations", default=2))
            prompts_per_epoch = int(_get(self.kwargs, "prompts_per_epoch", default=16))
            score_threshold = float(_get(self.kwargs, "score_threshold", default=0.5))

            rollout_fn = create_rollout_fn(
                rollout_engine=_get(self.kwargs, "rollout_engine", default=None),
                tools=_get(self.kwargs, "tools", default=None),
                reward_fn=primary_reward,
                system_prompt=_get(self.kwargs, "system_prompt", default=None),
                max_steps=_get(self.kwargs, "max_steps_per_turn", "max_steps", default=20),
                engine_kwargs=_get(self.kwargs, "engine_kwargs", default=None),
            )
            logger.info("[TrlAgenticBCO] rollout_fn built via create_rollout_fn ✓")

            TrainerClass = _OnlineBCOTrainer.build(
                rollout_fn=rollout_fn,
                prompt_pool=self._prompt_pool,
                prompts_per_epoch=prompts_per_epoch,
                score_threshold=score_threshold,
                num_generations=num_generations,
            )
            logger.info("[TrlAgenticBCO] OnlineBCOTrainer subclass created ✓")
        else:
            TrainerClass = BCOTrainer
            logger.info("[TrlAgenticBCO] Using standard offline BCOTrainer ✓")

        # ── Strip agentic-only and meta keys before passing to BCOTrainer ──
        # NOTE: never strip "model", "ref_model", or "processing_class" here —
        # they were resolved above and must reach BCOTrainer.__init__.
        for agentic_key in (
            "tools",
            "rollout_engine",
            "rollout_backend",
            "reward_funcs",
            "max_steps_per_turn",
            "engine_kwargs",
            "num_generations",
            "prompts_per_epoch",
            "score_threshold",
            "prompt_pool",
            "use_rollouts",
            "system_prompt",
            # generation params consumed by rollout_fn, not by BCOTrainer
            "max_new_tokens",
            "temperature",
            # loading params consumed above
            "trust_remote_code",
            "device_map",
        ):
            trainer_kw.pop(agentic_key, None)

        # ── Instantiate trainer ───────────────────────────────────────────
        trainer_kw.setdefault("args", bco_config)
        trainer_kw["train_dataset"] = self.train_dataset
        trainer_kw["eval_dataset"] = self.eval_dataset

        # BCOTrainer optionally accepts embedding_func and embedding_tokenizer
        # for UDM; pass through only if explicitly supplied.
        for opt_key in ("embedding_func", "embedding_tokenizer"):
            val = _get(self.kwargs, opt_key, default=None)
            if val is not None:
                trainer_kw.setdefault(opt_key, val)
            else:
                trainer_kw.setdefault(opt_key, None)

        # This pop is enough — pulls it out of trainer_kw so it's not passed twice
        peft_config = _resolve_peft_config(
            trainer_kw.pop("peft_config", None) or self.kwargs.get("peft_config")
        )
        self.trainer = TrainerClass(**trainer_kw, peft_config=peft_config)
        logger.info("[TrlAgenticBCO] BCOTrainer instantiated ✓")

        if self._agentic_mode:
            tool_names = [
                getattr(t, "__name__", str(t)) for t in (_get(self.kwargs, "tools") or [])
            ]
            if tool_names:
                logger.info(f"[TrlAgenticBCO] Registered tools: {tool_names}")

    # ─────────────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────────────

    def train(self) -> dict[str, Any]:
        """Full pipeline: setup_data → setup_trainer → train → save."""
        self.setup_data()
        self.setup_trainer()

        logger.info(
            f"[TrlAgenticBCO] ▶ Starting "
            f"{'agentic ' if self._agentic_mode else 'standard '}BCO training …"
        )
        t0 = time.time()
        self._train_result = self.trainer.train()
        elapsed = time.time() - t0
        logger.info(f"[TrlAgenticBCO] ✓ Training finished in {elapsed:.1f}s")

        output_dir = _get(self.kwargs, "output_dir", default="./output/bco_agentic")
        self.save_model(output_dir)
        return self.get_training_stats()

    # ─────────────────────────────────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────────────────────────────────

    def save_model(
        self,
        path: str | None = None,
        push_to_hub: bool = False,
        **extra_meta,
    ) -> str:
        save_path = path or _get(self.kwargs, "output_dir", default="./output/bco_agentic")
        Path(save_path).mkdir(parents=True, exist_ok=True)

        if self.trainer is not None:
            self.trainer.save_model(save_path)
            logger.info(f"[TrlAgenticBCO] Model saved → {save_path}")

            proc = getattr(self.trainer, "processing_class", None)
            if proc is not None and hasattr(proc, "save_pretrained"):
                proc.save_pretrained(save_path)
                logger.info(f"[TrlAgenticBCO] Tokenizer saved → {save_path}")

        config_path = Path(save_path) / "bco_config.yaml"
        cfg_dict = (
            self.trainer.args.to_dict()
            if self.trainer and hasattr(self.trainer.args, "to_dict")
            else self._serialisable_kwargs()
        )
        with open(config_path, "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)

        stats = self.get_training_stats()
        stats.update(extra_meta)
        stats_path = Path(save_path) / "training_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info(f"[TrlAgenticBCO] Artefacts saved → {save_path}")

        from agenttune.utils.provenance import write_provenance

        write_provenance(
            save_path,
            method="agentic.bco",
            base_model=_get(self.kwargs, "model"),
            dataset=_get(self.kwargs, "dataset_name", "dataset"),
            dataset_config=_get(self.kwargs, "dataset_config", "config_name"),
        )

        if push_to_hub or bool(_get(self.kwargs, "push_to_hub", default=False)):
            logger.info("[TrlAgenticBCO] Pushing to Hub …")
            self.trainer.push_to_hub()

        return str(Path(save_path).resolve())

    def load_model(self, path: str, **kwargs) -> None:
        from transformers import AutoTokenizer

        from agenttune.utils.model_class_resolver import resolve_model_class

        device_map = kwargs.pop("device_map", _get(self.kwargs, "device_map", default="auto"))
        trust_remote_code = kwargs.pop(
            "trust_remote_code", _get(self.kwargs, "trust_remote_code", default=False)
        )

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
            self.trainer.processing_class = tokenizer
        else:
            self.kwargs["model"] = model
            self.kwargs["processing_class"] = tokenizer

        logger.info(f"[TrlAgenticBCO] Model loaded from {path} ✓")

    # ─────────────────────────────────────────────────────────────────────
    # Stats & introspection
    # ─────────────────────────────────────────────────────────────────────

    def get_training_stats(self) -> dict[str, Any]:
        tr = self._train_result
        metrics = getattr(tr, "metrics", {}) if tr else {}
        tools = _get(self.kwargs, "tools")

        return {
            "model": str(_get(self.kwargs, "model", default="unknown")),
            "dataset_name": _get(self.kwargs, "dataset_name", "dataset", default="unknown"),
            "output_dir": _get(self.kwargs, "output_dir", default="./output/bco_agentic"),
            "use_rollouts": self._use_rollouts,
            "agentic_mode": self._agentic_mode,
            "tools": [getattr(t, "__name__", str(t)) for t in tools] if tools else [],
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "eval_size": len(self.eval_dataset) if self.eval_dataset else 0,
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


# ─────────────────────────────────────────────────────────────────────────────
# Usage examples
# ─────────────────────────────────────────────────────────────────────────────
#
# MODE 1 — Standard offline BCO (complete dataset):
# ─────────────────────────────────────────────────
# trainer = TrlAgenticBCO(
#     model="Qwen/Qwen3-0.6B",
#     ref_model="Qwen/Qwen3-0.6B",
#     train_dataset=full_ds,           # prompt / completion / label columns
#     output_dir="./runs/bco_offline",
#     beta=0.1,
# )
# results = trainer.train()
#
#
# MODE 2 — Online rollout BCO (reward-scored, no tools):
# ───────────────────────────────────────────────────────
# def my_reward(prompts, responses):
#     return [1.0 if len(r) > 10 else 0.0 for r in responses]
#
# trainer = TrlAgenticBCO(
#     model="Qwen/Qwen3-0.6B",
#     ref_model="Qwen/Qwen3-0.6B",
#     reward_funcs=my_reward,
#     train_dataset=prompt_only_ds,    # only "prompt" column needed
#     use_rollouts=True,
#     score_threshold=0.5,
#     num_generations=2,
#     prompts_per_epoch=16,
#     output_dir="./runs/bco_online",
#     beta=0.1,
# )
# results = trainer.train()
#
#
# MODE 3 — Agentic tool-calling BCO:
# ────────────────────────────────────
# trainer = TrlAgenticBCO(
#     model="Qwen/Qwen3-0.6B",
#     ref_model="Qwen/Qwen3-0.6B",
#     reward_funcs=my_reward,
#     tools=[calculator, web_search],
#     train_dataset=prompt_only_ds,
#     score_threshold=0.5,
#     num_generations=2,
#     prompts_per_epoch=16,
#     output_dir="./runs/bco_agentic",
#     beta=0.1,
# )
# results = trainer.train()
