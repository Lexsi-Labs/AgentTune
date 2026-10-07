"""
TrlAgenticDPO
=============
A kwargs-driven wrapper around TRL's DPOTrainer with full support for
agentic (multi-turn tool-calling) rollouts in OFFLINE mode only.

Uses standard DPOTrainer — NOT OnlineDPOTrainer (which is experimental).
Fresh chosen/rejected pairs are generated every training step via rollout_func,
delegated entirely to create_dpo_rollout_fn from rollout_factory.

Three operating modes
─────────────────────
MODE 1 — Standard offline DPO (complete dataset, no live generation):
    Dataset must have ``prompt``, ``chosen``, ``rejected`` columns.
    reward_funcs is NOT required. If passed it is silently ignored.

    TrlAgenticDPO(model=..., train_dataset=full_ds, beta=0.1)

MODE 2 — Reward-ranked rollouts (use_rollouts=True, no tools):
    Dataset needs only a ``prompt`` column.
    Completions are generated each step; reward_funcs picks chosen/rejected.

    TrlAgenticDPO(
        model=..., reward_funcs=my_reward,
        train_dataset=prompt_only_ds, use_rollouts=True,
    )

    Also auto-triggered when ``tools`` or ``rollout_engine`` are passed.

MODE 3 — Agentic tool-calling rollouts:
    Same as Mode 2 but generation goes through multi-turn tool calls.

    TrlAgenticDPO(
        model=..., reward_funcs=my_reward,
        tools=[calculator, web_search], train_dataset=prompt_only_ds,
    )

NOTE — passing reward_funcs with a complete dataset:
    reward_funcs alone does NOT enable live generation. Your static
    chosen/rejected data is used as-is, exactly like standard DPO.

    TrlAgenticDPO(
        model=..., reward_funcs=eval_reward,   # safe — no rollouts triggered
        train_dataset=full_ds,
    )
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get(kwargs: dict[str, Any], *keys, default=None):
    """Return the first matching key found in kwargs, else default."""
    for k in keys:
        if k in kwargs:
            return kwargs[k]
    return default


def _split_kwargs_for_offline_dpo(kwargs: dict[str, Any]) -> tuple[dict, dict]:
    """Split kwargs into (dpo_config_kwargs, dpo_trainer_kwargs)."""
    from agenttune.utils.environment import patch_colab_outstream_close

    patch_colab_outstream_close()
    from trl import DPOConfig, DPOTrainer

    config_keys = set(inspect.signature(DPOConfig.__init__).parameters) - {"self"}
    trainer_keys = set(inspect.signature(DPOTrainer.__init__).parameters) - {"self"}

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
# Patch: DPOTrainer.training_step
# ─────────────────────────────────────────────────────────────────────────────


def _patch_dpo_training_step(trainer_class) -> None:
    """
    Patches DPOTrainer.training_step to:
      1. Decode prompts from the current batch.
      2. Run self.rollout_func to generate fresh chosen/rejected pairs.
      3. Build new input tensors and call compute_loss normally.

    The patch is a no-op if self.rollout_func is not set, so standard
    (non-rollout) DPOTrainer instances are completely unaffected.

    rollout_func must return a dict with keys:
        prompt_ids   : list[list[int]]
        chosen_ids   : list[list[int]]
        rejected_ids : list[list[int]]
    as produced by create_dpo_rollout_fn from rollout_factory.
    """
    import torch

    _original_training_step = trainer_class.training_step

    def _patched_training_step(self, model, inputs, num_items_in_batch=None):
        # ── Guard: fall back to normal step if no rollout ─────────────────
        if getattr(self, "rollout_func", None) is None:
            return _original_training_step(self, model, inputs, num_items_in_batch)

        model.train()
        device = self.accelerator.device
        tokenizer = self.processing_class
        # A multimodal processor (e.g. Qwen3.5's Qwen3VLProcessor) has no pad/eos
        # ids itself; they live on its inner tokenizer.
        inner_tok = getattr(self.processing_class, "tokenizer", self.processing_class)
        pad_id = (
            getattr(self, "pad_token_id", None)
            or getattr(self.processing_class, "pad_token_id", None)
            or getattr(inner_tok, "pad_token_id", None)
            or getattr(self.processing_class, "eos_token_id", None)
            or getattr(inner_tok, "eos_token_id", None)
        )

        # ── 1. Decode prompts ─────────────────────────────────────────────
        # input_ids layout from DataCollatorForPreference:
        #   [chosen_0..n | rejected_0..n]
        # completion_mask = 1 for completion tokens, 0 for prompt tokens.
        # We extract prompt-only token ids from the chosen half.
        batch_size = inputs["input_ids"].size(0) // 2
        full_ids = inputs["input_ids"][:batch_size]  # chosen half
        comp_mask = inputs["completion_mask"][:batch_size]  # 1 = completion
        prompt_mask_bool = (comp_mask == 0) & (inputs["attention_mask"][:batch_size] == 1)

        raw_prompts = []
        for b in range(batch_size):
            p_ids = full_ids[b][prompt_mask_bool[b]]
            raw_prompts.append(tokenizer.decode(p_ids, skip_special_tokens=True))

        # Any dataset column beyond "prompt" (not just "answer") -- e.g. a custom
        # gold_answer/gold_chunk_ids a reward_fn depends on. Without this, only a
        # column literally named "answer" ever survived the decode-then-reconstruct
        # round trip below, silently starving reward_fn of everything else the
        # caller's dataset carried.
        lookup = getattr(self, "_extra_cols_by_prompt", None) or {}
        rollout_prompts = []
        for p in raw_prompts:
            extra = lookup.get(p)
            if extra is None:
                for src, e in lookup.items():
                    if src and (src in p or p in src):
                        extra = e
                        break
            rollout_prompts.append({"prompt": p, **extra} if extra else p)

        # ── 2. Rollout → already-paired chosen/rejected ───────────────────
        pairs = self.rollout_func(rollout_prompts, trainer=self)
        if not pairs["chosen_ids"]:
            logger.warning("[TrlAgenticDPO] no ranked rollout pairs this step; skipping")
            loss = torch.zeros((), device=device, requires_grad=True)
            self.accelerator.backward(loss)
            return loss.detach() / self.args.gradient_accumulation_steps

        # ── 3. Pad helper ─────────────────────────────────────────────────
        def _pad_stack(seqs, pad_val: int, side: str = "right"):
            max_len = max(len(s) for s in seqs)
            result = []
            for s in seqs:
                t = torch.tensor(s, dtype=torch.long, device=device)
                pad_len = max_len - len(t)
                padding = (0, pad_len) if side == "right" else (pad_len, 0)
                result.append(torch.nn.functional.pad(t, padding, value=pad_val))
            return torch.stack(result)

        prompt_tensor = _pad_stack(pairs["prompt_ids"], pad_id, "left")
        chosen_tensor = _pad_stack(pairs["chosen_ids"], pad_id, "right")
        rejected_tensor = _pad_stack(pairs["rejected_ids"], pad_id, "right")

        prompt_mask_t = (prompt_tensor != pad_id).long()
        chosen_mask_t = (chosen_tensor != pad_id).long()
        rejected_mask_t = (rejected_tensor != pad_id).long()

        # ── 4. Build full [prompt | completion] sequences ─────────────────
        chosen_full = torch.cat([prompt_tensor, chosen_tensor], dim=1)
        rejected_full = torch.cat([prompt_tensor, rejected_tensor], dim=1)
        chosen_attn = torch.cat([prompt_mask_t, chosen_mask_t], dim=1)
        rejected_attn = torch.cat([prompt_mask_t, rejected_mask_t], dim=1)
        chosen_comp_mask = torch.cat([torch.zeros_like(prompt_mask_t), chosen_mask_t], dim=1)
        rejected_comp_mask = torch.cat([torch.zeros_like(prompt_mask_t), rejected_mask_t], dim=1)

        # ── 5. Pad chosen & rejected to same length ───────────────────────
        max_len = max(chosen_full.size(1), rejected_full.size(1))

        def _rpad(t, length, val):
            return torch.nn.functional.pad(t, (0, length - t.size(1)), value=val)

        chosen_full = _rpad(chosen_full, max_len, pad_id)
        rejected_full = _rpad(rejected_full, max_len, pad_id)
        chosen_attn = _rpad(chosen_attn, max_len, 0)
        rejected_attn = _rpad(rejected_attn, max_len, 0)
        chosen_comp_mask = _rpad(chosen_comp_mask, max_len, 0)
        rejected_comp_mask = _rpad(rejected_comp_mask, max_len, 0)

        # ── 6. Assemble new inputs — layout: [chosen... | rejected...] ────
        new_inputs = {
            "input_ids": torch.cat([chosen_full, rejected_full], dim=0),
            "attention_mask": torch.cat([chosen_attn, rejected_attn], dim=0),
            "completion_mask": torch.cat([chosen_comp_mask, rejected_comp_mask], dim=0),
        }

        # ── 7. Standard DPO loss (untouched) ──────────────────────────────
        loss = self.compute_loss(model, new_inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

    trainer_class.training_step = _patched_training_step
    logger.info("[TrlAgenticDPO] Patched DPOTrainer.training_step ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class TrlAgenticDPO:
    """
    Offline DPO trainer with optional agentic rollouts.

    See module docstring for the three operating modes.

    Key parameters
    --------------
    model               : str or PreTrainedModel
    reward_funcs        : callable(prompts, completions) -> list[float]
                          Required when use_rollouts=True.
                          Safe to pass with a complete dataset when
                          use_rollouts=False — will NOT trigger rollouts.
    train_dataset       : HF Dataset
                          Complete mode : needs prompt / chosen / rejected.
                          Rollout mode  : needs only prompt.
    eval_dataset        : optional HF Dataset
    use_rollouts        : bool — explicit flag to enable live generation.
                          Default: auto-inferred. True only when tools or
                          rollout_engine are present, NOT merely because
                          reward_funcs was supplied.
    num_generations     : int >= 2 (completions per prompt, rollout mode)
    tools               : list of tool callables (auto-enables rollouts)
    rollout_engine      : pre-built external RolloutEngine
    max_steps_per_turn  : int (max tool-call steps per turn)
    system_prompt       : str (prepended to every prompt)
    max_new_tokens      : int (generation length, default 256)
    temperature         : float (default 0.7)
    beta                : float (DPO beta, default 0.1)
    output_dir          : str
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = kwargs.get("eval_dataset", None)
        self._train_result = None
        self.training_history: list[dict] = []

        # ── Resolve use_rollouts ──────────────────────────────────────────
        # Priority order:
        #   1. Explicit use_rollouts=True/False → always honoured.
        #   2. tools / rollout_engine present   → always rollouts.
        #   3. reward_funcs present             → TENTATIVELY True here;
        #      setup_data() will flip it to False if the dataset already has
        #      real chosen/rejected columns (complete dataset case).
        #   4. None of the above               → False (standard DPO).
        explicit = kwargs.get("use_rollouts", None)
        if explicit is not None:
            self._use_rollouts = bool(explicit)
            self._use_rollouts_explicit = True  # never overridden by dataset check
        else:
            self._use_rollouts_explicit = False
            self._use_rollouts = bool(
                kwargs.get("tools")
                or kwargs.get("rollout_engine")
                or kwargs.get("reward_funcs")  # tentative — refined in setup_data
            )

        # _agentic_mode mirrors _use_rollouts; kept in sync after setup_data
        self._agentic_mode = self._use_rollouts

        logger.info(
            f"[TrlAgenticDPO] Initialized "
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

        Dummy chosen/rejected columns are injected ONLY when
        use_rollouts=True AND the columns are actually missing.
        Complete datasets with real chosen/rejected are never touched.
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
                f"[TrlAgenticDPO] Using pre-loaded dataset " f"({len(self.train_dataset)} examples)"
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
                "task_type": "dpo",
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

            logger.info(f"[TrlAgenticDPO] Train: {len(self.train_dataset)} examples")
            if self.eval_dataset:
                logger.info(f"[TrlAgenticDPO] Eval : {len(self.eval_dataset)} examples")

        # ── Refine _use_rollouts now that we have the actual dataset ─────
        # Rule: if use_rollouts was NOT set explicitly and the dataset
        # already has both 'chosen' and 'rejected' columns, treat it as a
        # complete dataset and disable rollouts — even if reward_funcs was
        # passed (it will be ignored at training time).
        if not self._use_rollouts_explicit:
            ds_cols = set(self.train_dataset.column_names)
            has_pairs = ("chosen" in ds_cols) and ("rejected" in ds_cols)
            # tools/rollout_engine always win over the dataset check
            has_agentic_signal = bool(self.kwargs.get("tools") or self.kwargs.get("rollout_engine"))
            if has_pairs and not has_agentic_signal:
                if self._use_rollouts:
                    logger.info(
                        "[TrlAgenticDPO] Dataset has 'chosen'/'rejected' columns — "
                        "disabling rollouts (use use_rollouts=True to override)."
                    )
                self._use_rollouts = False
                self._agentic_mode = False

        logger.info(f"[TrlAgenticDPO] use_rollouts finalised → {self._use_rollouts}")

        # TRL DPOTrainer tokenises chosen/rejected at init. Rollout steps
        # replace those tensors. Do not copy gold/answer and do not add rows.
        if self._use_rollouts:
            n = len(self.train_dataset)
            if "chosen" not in self.train_dataset.column_names:
                self.train_dataset = self.train_dataset.add_column("chosen", [""] * n)
            if "rejected" not in self.train_dataset.column_names:
                self.train_dataset = self.train_dataset.add_column("rejected", [""] * n)
            if self.eval_dataset is not None:
                m = len(self.eval_dataset)
                if "chosen" not in self.eval_dataset.column_names:
                    self.eval_dataset = self.eval_dataset.add_column("chosen", [""] * m)
                if "rejected" not in self.eval_dataset.column_names:
                    self.eval_dataset = self.eval_dataset.add_column("rejected", [""] * m)

    # ─────────────────────────────────────────────────────────────────────
    # Trainer setup
    # ─────────────────────────────────────────────────────────────────────

    def setup_trainer(self) -> None:
        from trl import DPOConfig, DPOTrainer

        config_kw, trainer_kw = _split_kwargs_for_offline_dpo(self.kwargs)

        # ── Config defaults ───────────────────────────────────────────────
        output_dir = _get(self.kwargs, "output_dir", default="./output/dpo_agentic")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        config_defaults: dict[str, Any] = {
            "output_dir": output_dir,
            "num_train_epochs": _get(self.kwargs, "num_train_epochs", "epochs", default=1),
            "per_device_train_batch_size": _get(
                self.kwargs, "per_device_train_batch_size", "batch_size", default=1
            ),
            "gradient_accumulation_steps": _get(
                self.kwargs, "gradient_accumulation_steps", default=4
            ),
            "learning_rate": _get(self.kwargs, "learning_rate", "lr", default=1e-6),
            "warmup_steps": _get(self.kwargs, "warmup_steps", default=10),
            "max_grad_norm": _get(self.kwargs, "max_grad_norm", default=1.0),
            "seed": _get(self.kwargs, "seed", default=42),
            "logging_steps": _get(self.kwargs, "logging_steps", default=10),
            "save_steps": _get(self.kwargs, "save_steps", default=100),
            "remove_unused_columns": False,
            "beta": _get(self.kwargs, "beta", default=0.1),
            "max_length": _get(self.kwargs, "max_length", default=1024),
            "loss_type": _get(self.kwargs, "loss_type", default="sigmoid"),
        }

        if _get(self.kwargs, "max_steps") is not None:
            config_defaults["max_steps"] = _get(self.kwargs, "max_steps")

        for k, v in config_defaults.items():
            config_kw.setdefault(k, v)

        dpo_config = DPOConfig(**config_kw)
        logger.info(
            f"[TrlAgenticDPO] DPOConfig built "
            f"(output_dir={dpo_config.output_dir}, use_rollouts={self._use_rollouts})"
        )

        # ── Validate required args ────────────────────────────────────────
        if "model" not in trainer_kw and "model" not in self.kwargs:
            raise ValueError("[TrlAgenticDPO] 'model' is required.")

        # ── Resolve rollout_func (live-rollout mode only) ─────────────────
        rollout_func: Callable | None = None

        if self._use_rollouts:
            from agenttune.agentic.rollout_engines.rollout_factory import create_dpo_rollout_fn

            raw_reward = trainer_kw.pop("reward_funcs", None) or self.kwargs.get("reward_funcs")
            if raw_reward is None:
                raise ValueError(
                    "[TrlAgenticDPO] 'reward_funcs' is required when use_rollouts=True "
                    "to rank completions into chosen/rejected pairs."
                )
            from agenttune.agentic.rewards.composite import combine_rewards

            primary_reward = combine_rewards(
                raw_reward,
                weights=_get(self.kwargs, "reward_weights", default=None),
            )

            num_generations = int(_get(self.kwargs, "num_generations", default=2))
            if num_generations < 2:
                raise ValueError("[TrlAgenticDPO] num_generations must be >= 2.")

            rollout_func = create_dpo_rollout_fn(
                reward_fn=primary_reward,
                rollout_engine=_get(self.kwargs, "rollout_engine", default=None),
                tools=_get(self.kwargs, "tools", default=None),
                max_steps=_get(self.kwargs, "max_steps_per_turn", "max_steps", default=20),
                system_prompt=_get(self.kwargs, "system_prompt", default=None),
                num_generations=num_generations,
                engine_kwargs=_get(self.kwargs, "engine_kwargs", default=None),
            )
            logger.info("[TrlAgenticDPO] rollout_func built via create_dpo_rollout_fn ✓")

            # Patch the class — no-op for instances without rollout_func set
            _patch_dpo_training_step(DPOTrainer)

        # ── Strip all agentic-only keys before passing to DPOTrainer ─────
        for agentic_key in (
            "tools",
            "rollout_engine",
            "rollout_backend",
            "reward_funcs",
            "reward_weights",
            "max_steps_per_turn",
            "engine_kwargs",
            "num_generations",
            "use_rollouts",
        ):
            trainer_kw.pop(agentic_key, None)

        # ── Instantiate DPOTrainer ────────────────────────────────────────
        trainer_kw.setdefault("args", dpo_config)
        trainer_kw["train_dataset"] = self.train_dataset
        trainer_kw["eval_dataset"] = self.eval_dataset

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
        self.trainer = DPOTrainer(
            **trainer_kw,
            peft_config=peft_config,
        )
        logger.info("[TrlAgenticDPO] DPOTrainer instantiated ✓")

        # ── Attach rollout_func AFTER __init__ (rollout mode only) ────────
        # DPOTrainer.__init__ tokenises the dataset; we must not interfere
        # before that is done.
        if rollout_func is not None:
            self.trainer.rollout_func = rollout_func
            self.trainer._max_new_tokens = int(_get(self.kwargs, "max_new_tokens", default=256))
            self.trainer._temperature = float(_get(self.kwargs, "temperature", default=0.7))
            self._num_generations = num_generations
            extra_cols = [c for c in self.train_dataset.column_names if c != "prompt"]
            if extra_cols:
                self.trainer._extra_cols_by_prompt = {
                    row["prompt"]: {c: row[c] for c in extra_cols} for row in self.train_dataset
                }
            logger.info("[TrlAgenticDPO] rollout_func attached to trainer ✓")

            if self._agentic_mode:
                tool_names = [
                    getattr(t, "__name__", str(t)) for t in (_get(self.kwargs, "tools") or [])
                ]
                logger.info(f"[TrlAgenticDPO] Registered tools: {tool_names}")

    # ─────────────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────────────

    def train(self) -> dict[str, Any]:
        """Full pipeline: setup_data → setup_trainer → train → save."""
        self.setup_data()
        self.setup_trainer()

        logger.info(
            f"[TrlAgenticDPO] ▶ Starting "
            f"{'agentic ' if self._agentic_mode else 'standard '}offline DPO training …"
        )
        t0 = time.time()
        self._train_result = self.trainer.train()
        elapsed = time.time() - t0
        logger.info(f"[TrlAgenticDPO] ✓ Training finished in {elapsed:.1f}s")

        output_dir = _get(self.kwargs, "output_dir", default="./output/dpo_agentic")
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
        save_path = path or _get(self.kwargs, "output_dir", default="./output/dpo_agentic")
        Path(save_path).mkdir(parents=True, exist_ok=True)

        if self.trainer is not None:
            self.trainer.save_model(save_path)
            logger.info(f"[TrlAgenticDPO] Model saved → {save_path}")

            proc = getattr(self.trainer, "processing_class", None)
            if proc is not None and hasattr(proc, "save_pretrained"):
                proc.save_pretrained(save_path)
                logger.info(f"[TrlAgenticDPO] Tokenizer saved → {save_path}")

        config_path = Path(save_path) / "dpo_config.yaml"
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

        logger.info(f"[TrlAgenticDPO] Artefacts saved → {save_path}")

        from agenttune.utils.provenance import write_provenance

        write_provenance(
            save_path,
            method="agentic.dpo",
            base_model=_get(self.kwargs, "model"),
            dataset=_get(self.kwargs, "dataset_name", "dataset"),
            dataset_config=_get(self.kwargs, "dataset_config", "config_name"),
        )

        if push_to_hub or bool(_get(self.kwargs, "push_to_hub", default=False)):
            logger.info("[TrlAgenticDPO] Pushing to Hub …")
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

        logger.info(f"[TrlAgenticDPO] Model loaded from {path} ✓")

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
            "output_dir": _get(self.kwargs, "output_dir", default="./output/dpo_agentic"),
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
