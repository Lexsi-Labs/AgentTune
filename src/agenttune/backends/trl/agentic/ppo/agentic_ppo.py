# agenttune/agentic/trainers/agentic_ppo.py
"""
TrlAgenticPpo
=============
A thin, kwargs-driven wrapper around TRL's PPOTrainer with full support
for agentic (multi-turn tool-calling) rollouts.

Key differences from DPO:
  - PPO needs a value_model AND reward_model (or reward_fn wrapped in one)
  - Generation happens inside train() loop — we patch batch_generation only
  - Rewards are per-token, not per-sequence — we handle the conversion
  - We can wrap a reward_fn callable into a fake nn.Module reward model

Patching strategy
-----------------
We do NOT rewrite train(). Instead we do two targeted patches on the
PPOTrainer INSTANCE after construction:

  1. _patch_batch_generation(trainer, rollout_fn)
     Replaces the module-level `batch_generation` function with a closure
     that routes through our agentic rollout_fn. The rest of train() is
     completely untouched — the PPO update maths, logging, checkpointing
     all run exactly as TRL wrote them.

  2. _patch_generate_completions(trainer, rollout_fn)  [optional]
     Replaces the `batch_generation` call inside generate_completions() so
     eval sampling also uses agentic rollouts. Only applied when
     num_sample_generations > 0.

Both patches are instance-scoped (via types.MethodType or module-swap with
a __del__ restore hook) so they never affect other PPOTrainer instances.

─────────────────────────────────────────────────────────────────────────
Usage examples
--------------

# Standard PPO (original behaviour)
trainer = TrlAgenticPpo(
    reward_mode  = "model",
    reward_model = my_reward_nn_module,
    value_mode   = "model",
    value_model  = my_value_nn_module,
    ...
)

# Fully callable (no nn.Module needed)
trainer = TrlAgenticPpo(
    reward_mode  = "fn",
    reward_funcs = my_reward_fn,
    value_mode   = "fn",
    value_fn     = my_value_fn,
    ...
)

# Mixed
trainer = TrlAgenticPpo(
    reward_mode  = "fn",
    reward_funcs = my_reward_fn,
    value_mode   = "wrapper",   # auto-wraps policy backbone
    ...
)

# Auto (default — just pass whatever you have)
trainer = TrlAgenticPpo(
    reward_funcs = my_reward_fn,   # no reward_mode needed
    ...
)
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

import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get(kwargs: dict[str, Any], *keys, default=None):
    for k in keys:
        if k in kwargs:
            return kwargs[k]
    return default


def _split_kwargs_for_ppo(kwargs: dict[str, Any]) -> tuple[dict, dict]:
    """Split kwargs into (ppo_config_kwargs, ppo_trainer_kwargs)."""
    # from trl import PPOConfig, PPOTrainer
    from trl.experimental.ppo import PPOConfig, PPOTrainer

    config_keys = set(inspect.signature(PPOConfig.__init__).parameters) - {"self"}
    trainer_keys = set(inspect.signature(PPOTrainer.__init__).parameters) - {"self"}

    config_kw: dict[str, Any] = {}
    trainer_kw: dict[str, Any] = {}

    for k, v in kwargs.items():
        if k in trainer_keys:
            trainer_kw[k] = v
        elif k in config_keys:
            config_kw[k] = v

    return config_kw, trainer_kw


def _alias_ppo_printer_loss(trainer) -> None:
    """TRL PPO logs loss/policy_avg. The HF notebook table only prints `loss`."""
    orig = trainer.log

    def log(logs, *args, **kwargs):
        if (
            isinstance(logs, dict)
            and "loss" not in logs
            and logs.get("loss/policy_avg") is not None
        ):
            logs = {**logs, "loss": logs["loss/policy_avg"]}
        return orig(logs, *args, **kwargs)

    trainer.log = log


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
# Reward function → nn.Module wrapper
#
# PPOTrainer.train() calls:
#   _, score, _ = get_reward(reward_model, postprocessed_query_response,
#                            processing_class.pad_token_id, context_length)
#
# get_reward() does:
#   reward_model(input_ids, attention_mask) -> output with .logits shape (B, T, 1)
#   score = output.logits[:, context_length - 1, 0]
#
# So we wrap a callable reward_fn into an nn.Module that:
#   1. Decodes input_ids back to text (split at ctx_len)
#   2. Calls reward_fn(prompts, completions)
#   3. Returns a fake logits tensor so score lands at the right position
# ─────────────────────────────────────────────────────────────────────────────


class RewardFnWrapper(nn.Module):
    """
    Wraps a callable reward function into an nn.Module compatible with
    PPOTrainer's get_reward() call.

    get_reward() expects:
        model(input_ids, attention_mask) -> output with .logits shape (B, T, 1)
        score = output.logits[:, context_length - 1, 0]

    We produce a fake logits tensor of shape (B, T, 1) where the value at
    [b, context_length-1, 0] is the reward for example b.
    """

    def __init__(self, reward_fn: Callable, tokenizer):
        super().__init__()
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        # Dummy parameter so accelerator can move this "model" to device
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._ctx_len = 0  # updated by patch before each get_reward call
        # Updated by _agentic_batch_generation right before each get_reward()
        # call, from the same rollout that produced this batch's completions.
        # None when unavailable (e.g. no tools) or when the batch size doesn't
        # line up (stale value from a different-sized batch).
        self._tool_call_counts = None

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        device = self._dummy.device
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        ctx_len = self._ctx_len if self._ctx_len > 0 else seq_len // 2

        prompt_ids = input_ids[:, :ctx_len]
        completion_ids = input_ids[:, ctx_len:]

        prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        # Only trust a stashed tool_call_counts if it actually matches this batch —
        # see the guard where _tool_call_counts is set in _agentic_batch_generation.
        tcc = self._tool_call_counts
        extra_kwargs = (
            {"tool_call_counts": tcc} if tcc is not None and len(tcc) == batch_size else {}
        )

        with torch.no_grad():
            try:
                raw_rewards = self.reward_fn(
                    prompts=prompts, completions=completions, **extra_kwargs
                )
            except TypeError:
                try:
                    raw_rewards = self.reward_fn(
                        prompts=prompts, responses=completions, **extra_kwargs
                    )
                except TypeError:
                    raw_rewards = self.reward_fn(completions)

        reward_tensor = torch.tensor(
            [float(r) if r is not None else 0.0 for r in raw_rewards],
            dtype=torch.float32,
            device=device,
        )

        # Build fake logits: shape (B, T, 1)
        # PPOTrainer: score = logits[:, context_length - 1, 0]
        logits = torch.zeros(batch_size, seq_len, 1, device=device)
        logits[:, ctx_len - 1, 0] = reward_tensor

        class _FakeOutput:
            def __init__(self, logits):
                self.logits = logits
                self.hidden_states = (logits,)

        return _FakeOutput(logits)

    def score(self, hidden_states):
        """Compatibility shim — not used in our wrapper path."""
        return hidden_states[..., :1]

    @property
    def base_model_prefix(self):
        return "base_model"

    @property
    def base_model(self):
        return self

    @property
    def config(self):
        return None


def _resolve_config_attr(config, name: str):
    """`config.hidden_size` / `config.vocab_size` aren't there for a
    composite/omni-modal config (e.g. Gemma4Config, which nests separate
    text_config/vision_config/audio_config instead of exposing either at the
    top level -- the same shape resolve_model_class already has to account for
    with AutoModelForImageTextToText). Falls back to config.text_config.<name>,
    the standard HF convention for that case, since PPO's value head and
    generation logits only ever concern the text tower regardless of how the
    checkpoint is organised.
    """
    value = getattr(config, name, None)
    if value is not None:
        return value
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, name, None) if text_config is not None else None
    if value is not None:
        return value
    raise AttributeError(
        f"[TrlAgenticPpo] Could not resolve {name!r} from {type(config).__name__} "
        f"(checked config.{name} and config.text_config.{name})."
    )


def _resolve_hidden_size(config) -> int:
    return _resolve_config_attr(config, "hidden_size")


def _resolve_vocab_size(config) -> int:
    return _resolve_config_attr(config, "vocab_size")


class ValueModelWrapper(nn.Module):
    """
    Wraps the policy model as a value model when no explicit value_model
    is provided. Uses a linear head on top of the last hidden state.

    PPOTrainer calls:
        full_value, _, _ = get_reward(value_model, query_response, pad_id, ctx_len)
        value = full_value[:, context_length - 1 : -1].squeeze(-1)

    So value_model must return logits of shape (B, T, 1).
    """

    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.score = nn.Linear(hidden_size, 1, bias=False)
        # nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.weight)
        self.score = self.score.to(next(backbone.parameters()).dtype)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **{k: v for k, v in kwargs.items() if k not in ("input_ids", "attention_mask")},
        )
        hidden = outputs.hidden_states[-1]  # (B, T, H)
        values = self.score(hidden)  # (B, T, 1)

        class _FakeOutput:
            def __init__(self, logits, hidden_states):
                self.logits = logits
                self.hidden_states = hidden_states

        return _FakeOutput(values, outputs.hidden_states)

    @property
    def base_model_prefix(self):
        return "backbone"

    @property
    def config(self):
        return self.backbone.config


class ValueFnWrapper(nn.Module):
    """
    Wraps a callable value_fn into an nn.Module compatible with
    PPOTrainer's get_reward() call for value estimation.

    value_fn signature:
        value_fn(prompts: list[str], completions: list[str]) -> list[float]

    Returns logits of shape (B, T, 1) with the value estimate placed at
    position [b, ctx_len-1, 0], matching what PPOTrainer expects.

    value_fn runs under torch.no_grad() (it's a plain Python callable, not a
    trainable model), so — same as RewardFnWrapper — the produced value has no
    gradient; score() reads it straight back off the fake logits instead of
    routing through a learnable head.
    """

    def __init__(self, value_fn: Callable, tokenizer, hidden_size: int):
        super().__init__()
        self.value_fn = value_fn
        self.tokenizer = tokenizer
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._ctx_len = 0

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        device = self._dummy.device
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        ctx_len = self._ctx_len if self._ctx_len > 0 else seq_len // 2

        prompt_ids = input_ids[:, :ctx_len]
        completion_ids = input_ids[:, ctx_len:]

        prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        with torch.no_grad():
            raw_values = self.value_fn(prompts=prompts, completions=completions)

        value_tensor = torch.tensor(
            [float(v) if v is not None else 0.0 for v in raw_values],
            dtype=torch.float32,
            device=device,
        )

        logits = torch.zeros(batch_size, seq_len, 1, device=device)
        logits[:, ctx_len - 1, 0] = value_tensor

        class _FakeOutput:
            def __init__(self, logits):
                self.logits = logits
                self.hidden_states = (logits,)

        return _FakeOutput(logits)

    def score(self, hidden_states):
        return hidden_states[..., :1]

    @property
    def base_model_prefix(self):
        return "base_model"

    @property
    def base_model(self):
        return self

    @property
    def config(self):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Rollout builder for PPO
# ─────────────────────────────────────────────────────────────────────────────


def _build_ppo_rollout_fn(
    rollout_engine=None,
    tools: list | None = None,
    max_steps: int = 20,
    system_prompt: str | None = None,
    engine_kwargs: dict | None = None,
    rollout_backend: str | None = None,
) -> Callable:
    """
    Builds a PPO-compatible rollout function by wrapping create_rollout_fn.

    Returns a callable:
        fn(prompts: list[str], trainer) -> {
            "completion_ids": list[list[int]],
            "logprobs":       list[list[tuple]],
            "prompt_ids":     list[list[int]],
            "responses":      list[str],
            "trajectories":   list[Trajectory],
        }

    The trainer reference is forwarded into _execute_trajectory so that
    _gen() can use trainer.model / trainer.accelerator for generation
    (Path A in the rollout factory) without needing a separate engine.
    """
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

    _base_rollout = create_rollout_fn(
        rollout_engine=rollout_engine,
        rollout_backend=rollout_backend,
        tools=tools,
        max_steps=max_steps,
        system_prompt=system_prompt,
        engine_kwargs=engine_kwargs or {},
    )

    def ppo_rollout_fn(prompts: list[str], trainer=None) -> dict[str, Any]:
        # Pass trainer through so _gen() can use it for model.generate()
        batch = _base_rollout(prompts, trainer=trainer)
        return {
            "completion_ids": batch["completion_ids"],
            "logprobs": batch["logprobs"],
            "prompt_ids": batch["prompt_ids"],
            "responses": batch["responses"],
            "trajectories": batch.get("trajectories", []),
            # Carried through so _agentic_batch_generation can stash it on the
            # RewardFnWrapper — get_reward() only sees raw token ids, with no
            # other path back to per-completion rollout metadata like this.
            "tool_call_counts": batch.get("tool_call_counts"),
        }

    return ppo_rollout_fn


# ─────────────────────────────────────────────────────────────────────────────
# Shared rollout → tensor conversion
# ─────────────────────────────────────────────────────────────────────────────


def _rollout_to_batch_generation_outputs(
    rollout_output: dict[str, Any],
    queries: torch.Tensor,
    pad_id: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert rollout_fn output into the (query_responses, logitss) tensors
    that PPOTrainer.train() expects from batch_generation().

    query_responses : (B, ctx_len + max_comp_len)
    logitss         : (B, max_comp_len, vocab_size)
        fake one-hot so that selective_log_softmax(logitss, response) ≈ rollout logprobs

    Why fake one-hot?
    -----------------
    train() computes OLD logprobs from logitss for the PPO ratio:
        logprob = selective_log_softmax(logits, response)
    We place each token's rollout log-prob at its position via:
        logit[token_id] = lp + log(vocab_size)   (all others = 0)
    so that log_softmax(logits)[token_id] ≈ lp.

    The POLICY UPDATE uses fresh logprobs from a real forward() pass, so
    this approximation only affects the stored old_logprob reference value,
    which is acceptable — any PPO implementation that uses off-policy rollouts
    faces the same approximation.
    """
    import math

    comp_ids_list = rollout_output["completion_ids"]  # list[list[int]]
    lp_list = rollout_output["logprobs"]  # list[list[tuple|float]]

    max_comp_len = max(len(c) for c in comp_ids_list) if comp_ids_list else 1

    # Pad completions and stack into a single tensor
    padded_comp = []
    for c in comp_ids_list:
        t = torch.tensor(c, dtype=torch.long, device=device)
        pad_len = max_comp_len - len(c)
        t = torch.nn.functional.pad(t, (0, pad_len), value=pad_id)
        padded_comp.append(t)
    completion_tensor = torch.stack(padded_comp)  # (B, max_comp_len)

    # query_responses = concat(prompt, completion)
    query_responses = torch.cat([queries, completion_tensor], dim=1)  # (B, ctx+comp)

    # Build fake logitss: (B, comp_len, vocab_size)
    logitss = torch.zeros(
        queries.shape[0],
        max_comp_len,
        vocab_size,
        device=device,
        dtype=torch.float32,
    )
    for b_idx, (lp_seq, comp_seq) in enumerate(zip(lp_list, comp_ids_list, strict=False)):
        for t_idx, (lp_val, token_id) in enumerate(
            zip(lp_seq[:max_comp_len], comp_seq[:max_comp_len], strict=False)
        ):
            lp = lp_val[0] if isinstance(lp_val, tuple | list) else float(lp_val)
            logitss[b_idx, t_idx, token_id] = lp + math.log(vocab_size)

    return query_responses, logitss


# ─────────────────────────────────────────────────────────────────────────────
# Minimal patch 1: replace batch_generation in ppo_trainer module
#
# train() calls the module-level `batch_generation` by name, so we swap it
# in the module dict. A __del__ hook on a dynamically-subclassed trainer
# restores the original when the instance is garbage-collected.
# ─────────────────────────────────────────────────────────────────────────────


def _patch_batch_generation(trainer_instance, rollout_func: Callable) -> None:
    """
    Replaces trl.trainer.ppo_trainer.batch_generation with a closure that
    calls rollout_func and converts the output into the (query_responses,
    logitss) tensors train() expects.

    Restores the original automatically when trainer_instance is GC'd.
    """
    import trl.experimental.ppo.ppo_trainer as _ppo_mod

    _original = _ppo_mod.batch_generation

    def _agentic_batch_generation(
        model,  # unwrapped_model.policy — not used; rollout_func handles gen
        queries,  # (B, ctx_len) prompt token ids
        local_rollout_forward_batch_size,  # ignored for agentic
        pad_token_id,
        generation_config,
    ):
        device = queries.device
        processing_class = trainer_instance.processing_class
        ctx_len = queries.shape[1]

        # Keep reward wrapper in sync with current context length
        if isinstance(trainer_instance.reward_model, RewardFnWrapper):
            trainer_instance.reward_model._ctx_len = ctx_len

        # Decode to text — rollout_func (and _execute_trajectory inside it)
        # accept plain strings and forward the trainer reference to _gen()
        raw_prompts = processing_class.batch_decode(queries, skip_special_tokens=True)

        rollout_output = rollout_func(raw_prompts, trainer=trainer_instance)

        # Stash per-completion rollout metadata (e.g. tool_call_counts) on the
        # reward wrapper, keyed by this batch's order, so forward() — which only
        # ever sees raw input_ids/attention_mask from get_reward() — can still
        # forward it to reward_fn. Only trusted if the length lines up with this
        # batch; a mismatch (e.g. a stale value from a differently-sized batch)
        # is worse than none, so it's cleared instead.
        if isinstance(trainer_instance.reward_model, RewardFnWrapper):
            tcc = rollout_output.get("tool_call_counts")
            trainer_instance.reward_model._tool_call_counts = (
                tcc if tcc is not None and len(tcc) == len(raw_prompts) else None
            )

        vocab_size = _resolve_vocab_size(trainer_instance.model.policy.config)
        return _rollout_to_batch_generation_outputs(
            rollout_output, queries, pad_token_id, vocab_size, device
        )

    # Swap into module namespace
    _ppo_mod.batch_generation = _agentic_batch_generation

    # Restore original on GC via a one-off subclass
    _orig_del = getattr(trainer_instance.__class__, "__del__", None)

    def _restore_and_del(self):
        _ppo_mod.batch_generation = _original
        if _orig_del is not None:
            _orig_del(self)

    trainer_instance.__class__ = type(
        trainer_instance.__class__.__name__,
        (trainer_instance.__class__,),
        {"__del__": _restore_and_del},
    )

    logger.info("[TrlAgenticPpo] Patched trl.trainer.ppo_trainer.batch_generation ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Minimal patch 2: replace generate_completions (eval sampling only)
# ─────────────────────────────────────────────────────────────────────────────


def _patch_generate_completions(trainer_instance, rollout_func: Callable) -> None:
    """
    Binds a new generate_completions() to the trainer instance that uses
    rollout_func instead of batch_generation for eval sample generation.

    Only applied when num_sample_generations > 0.
    """
    import pandas as pd
    from accelerate.utils import gather_object
    from trl.trainer.ppo_trainer import truncate_response
    from trl.trainer.utils import get_reward

    def _agentic_generate_completions(self, sampling: bool = False):
        if self.eval_dataset is None:
            return
        args = self.args
        processing_class = self.processing_class

        table: dict[str, list] = {"query": [], "model response": [], "score": []}

        for batch in self.eval_dataloader:
            query = batch["input_ids"]
            with torch.no_grad():
                context_length = query.shape[1]

                raw_prompts = processing_class.batch_decode(query, skip_special_tokens=True)
                rollout_output = rollout_func(raw_prompts, trainer=self)

                vocab_size = _resolve_vocab_size(self.model.policy.config)
                query_responses, _ = _rollout_to_batch_generation_outputs(
                    rollout_output,
                    query,
                    processing_class.pad_token_id,
                    vocab_size,
                    self.accelerator.device,
                )

                response = query_responses[:, context_length:]
                postprocessed_response = response
                if self.stop_token_id is not None:
                    postprocessed_response = truncate_response(
                        self.stop_token_id, processing_class.pad_token_id, response
                    )

                table["query"].extend(
                    gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                )
                table["model response"].extend(
                    gather_object(processing_class.batch_decode(postprocessed_response))
                )

                postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                if isinstance(self.reward_model, RewardFnWrapper):
                    self.reward_model._ctx_len = context_length
                _, score, _ = get_reward(
                    self.reward_model,
                    postprocessed_query_response,
                    processing_class.pad_token_id,
                    context_length,
                )
                table["score"].extend(
                    self.accelerator.gather_for_metrics(score).float().cpu().numpy()
                )

            if sampling:
                break

        df = pd.DataFrame(table)
        if self.accelerator.is_main_process:
            try:
                from rich.console import Console
                from rich.table import Table as RichTable

                console = Console()
                rt = RichTable(show_lines=True)
                for col in df.columns:
                    rt.add_column(col)
                for _, row in df.head(5).iterrows():
                    rt.add_row(*row.astype(str).tolist())
                console.print(rt)
            except ImportError:
                pass
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})
            if "comet_ml" in args.report_to:
                from trl.trainer.utils import log_table_to_comet_experiment

                log_table_to_comet_experiment(name="completions.csv", table=df)

    trainer_instance.generate_completions = types.MethodType(
        _agentic_generate_completions, trainer_instance
    )
    logger.info("[TrlAgenticPpo] Patched generate_completions ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class TrlAgenticPPO:
    """
    kwargs-driven PPO trainer supporting agentic multi-turn rollouts
    and reward_fn / value_fn callables (no separate nn.Module required).

    ┌──────────────────────────────────────────────────────────────────────┐
    │  Required                                                            │
    │  model          str or nn.Module — policy model                     │
    │  train_dataset  Dataset with "input_ids" column (tokenized prompts) │
    │  eval_dataset   Dataset (required by PPOTrainer)                    │
    │  One of:                                                             │
    │    reward_funcs   callable(prompts, completions) -> list[float]     │
    │    reward_model   nn.Module (standard PPO)                          │
    ├──────────────────────────────────────────────────────────────────────┤
    │  reward_mode                                                         │
    │  "auto"  → use reward_model if given, else wrap reward_funcs (def.) │
    │  "model" → force nn.Module (ValueError if absent)                   │
    │  "fn"    → force wrap callable (ValueError if absent)              │
    │                                                                      │
    │  value_mode                                                          │
    │  "auto"    → use value_model if given, else ValueModelWrapper (def.)│
    │  "model"   → force nn.Module (ValueError if absent)                 │
    │  "fn"      → wrap value_fn callable into ValueFnWrapper             │
    │  "wrapper" → always ValueModelWrapper(policy_backbone)              │
    ├──────────────────────────────────────────────────────────────────────┤
    │  Agentic params                                                      │
    │  tools              list of BaseTool / callables                    │
    │  rollout_engine     optional pre-built RolloutEngine                │
    │  rollout_backend    "auto"|"transformers"|"vllm"|"api"             │
    │  max_steps_per_turn int (default 20)                                │
    │  system_prompt      str                                             │
    ├──────────────────────────────────────────────────────────────────────┤
    │  PPOConfig params  (auto-routed via introspection)                  │
    │  output_dir, total_episodes, response_length, learning_rate,        │
    │  kl_coef, cliprange, vf_coef, gamma, lam, temperature,             │
    │  num_ppo_epochs, num_mini_batches, local_rollout_forward_batch_size │
    └──────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.trainer: Any = None
        self.train_dataset = None
        self.eval_dataset = kwargs.get("eval_dataset", None)
        self._train_result = None
        self.training_history: list[dict] = []

        self._agentic_mode = bool(kwargs.get("tools") or kwargs.get("rollout_engine"))
        logger.info(f"[TrlAgenticPpo] Initialized (agentic={self._agentic_mode})")

    # ─────────────────────────────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────────────────────────────

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

        def _maybe_tokenize(ds):
            """
            If the dataset has a 'prompt' column but no 'input_ids', tokenize
            it in-place using the processing_class passed by the caller.
            Allows users to pass raw prompt datasets without pre-tokenizing.
            """
            if ds is None:
                return ds
            if "input_ids" in ds.column_names:
                return ds  # already tokenized — nothing to do

            if "prompt" not in ds.column_names:
                raise ValueError(
                    "[TrlAgenticPpo] Dataset must have either 'input_ids' "
                    "(pre-tokenized) or 'prompt' (raw text) column."
                )

            tokenizer = self.kwargs.get("processing_class") or self.kwargs.get("tokenizer")
            if tokenizer is None:
                model_arg = self.kwargs.get("model")
                if isinstance(model_arg, str):
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(model_arg)
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    # Cache so setup_trainer() reuses the same instance
                    self.kwargs["processing_class"] = tokenizer
                    logger.info(f"[TrlAgenticPpo] Auto-loaded tokenizer from '{model_arg}' ✓")
                else:
                    raise ValueError(
                        "[TrlAgenticPpo] Cannot auto-tokenize: pass 'processing_class' "
                        "or a model name string so the tokenizer can be loaded."
                    )

            max_length = _get(self.kwargs, "max_prompt_length", "max_length", default=128)

            def _tokenize(example):
                enc = tokenizer(
                    example["prompt"],
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                )
                return {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                }

            logger.info(f"[TrlAgenticPpo] Auto-tokenizing dataset (max_length={max_length}) …")
            # PPOTrainer's default collator tensorizes every remaining column.
            # Drop prompt/answer/… so only input_ids and attention_mask remain.
            ds = ds.map(_tokenize, remove_columns=list(ds.column_names))
            logger.info("[TrlAgenticPpo] Auto-tokenization done ✓")
            return ds

        # ── PATH 1: pre-loaded dataset ────────────────────────────────────────
        if "train_dataset" in self.kwargs:
            self.train_dataset = _maybe_tokenize(_apply_fmt(self.kwargs["train_dataset"]))
            if "eval_dataset" in self.kwargs:
                self.eval_dataset = _maybe_tokenize(_apply_fmt(self.kwargs["eval_dataset"]))
            logger.info(
                f"[TrlAgenticPpo] Using pre-loaded dataset " f"({len(self.train_dataset)} examples)"
            )
            return

        # ── PATH 2: DataManager ───────────────────────────────────────────────
        from agenttune.data.manager import DataManager

        dataset_name = _get(self.kwargs, "dataset_name", "dataset", default="trl-lib/tldr")
        config_name = _get(self.kwargs, "dataset_config", "config_name", default=None)
        split = _get(self.kwargs, "split", default=None)
        max_samples = _get(self.kwargs, "max_samples", default=None)
        dm_cfg = _get(self.kwargs, "data_manager_config", default=None)

        dm_kwargs = {
            "task_type": "ppo",
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

        train_ds = _maybe_tokenize(_apply_fmt(dataset_dict.get("train", None)))
        eval_ds = _maybe_tokenize(_apply_fmt(dataset_dict.get("validation", None)))

        if train_ds and max_samples and len(train_ds) > max_samples:
            train_ds = train_ds.select(range(max_samples))

        self.train_dataset = train_ds
        if self.eval_dataset is None:
            self.eval_dataset = eval_ds

        logger.info(f"[TrlAgenticPpo] Train: {len(self.train_dataset)} examples")

    # ─────────────────────────────────────────────────────────────────────
    # Trainer setup
    # ─────────────────────────────────────────────────────────────────────

    def setup_trainer(self) -> None:
        # from trl import PPOConfig, PPOTrainer
        from transformers import AutoTokenizer
        from trl.experimental.ppo import PPOConfig, PPOTrainer

        from agenttune.utils.model_class_resolver import resolve_model_class

        config_kw, trainer_kw = _split_kwargs_for_ppo(self.kwargs)

        # ── Resolve model and tokenizer ───────────────────────────────────
        model_arg = trainer_kw.get("model") or self.kwargs.get("model")
        if model_arg is None:
            raise ValueError("[TrlAgenticPpo] 'model' is required.")

        tokenizer = trainer_kw.get("processing_class") or self.kwargs.get("processing_class")
        if tokenizer is None:
            trust_remote_code = _get(self.kwargs, "trust_remote_code", default=False)
            if isinstance(model_arg, str):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_arg, trust_remote_code=trust_remote_code
                )
            else:
                raise ValueError("[TrlAgenticPpo] 'processing_class' (tokenizer) is required.")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if isinstance(model_arg, str):
            # DPO/BCO/GRPO/RLOO all read trust_remote_code from kwargs (default
            # False) -- this was hardcoded False here with no override at all,
            # so PPO alone couldn't load a checkpoint that ships custom modeling
            # code (e.g. microsoft/Phi-mini-MoE-instruct's PhiMoEForCausalLM).
            trust_remote_code = _get(self.kwargs, "trust_remote_code", default=False)
            policy_model = resolve_model_class(
                model_arg, trust_remote_code=trust_remote_code
            ).from_pretrained(
                model_arg,
                device_map=_get(self.kwargs, "device_map", default="auto"),
                torch_dtype=_get(self.kwargs, "torch_dtype", default=torch.bfloat16),
                trust_remote_code=trust_remote_code,
            )
        else:
            policy_model = model_arg

        # ── Resolve reward model ──────────────────────────────────────────
        reward_model = trainer_kw.get("reward_model") or self.kwargs.get("reward_model")
        raw_reward = self.kwargs.get("reward_funcs") or self.kwargs.get("reward_fn")
        reward_mode = _get(self.kwargs, "reward_mode", default="auto")
        # reward_mode options:
        #   "auto"  → use reward_model if given, else wrap reward_fn (default)
        #   "model" → force use of reward_model nn.Module (error if not given)
        #   "fn"    → force wrap reward_fn into RewardFnWrapper (error if not given)

        if reward_mode == "model":
            if reward_model is None:
                raise ValueError(
                    "[TrlAgenticPpo] reward_mode='model' requires 'reward_model' (nn.Module)."
                )
            logger.info("[TrlAgenticPpo] Using explicit reward_model (nn.Module) ✓")

        elif reward_mode == "fn":
            if raw_reward is None:
                raise ValueError(
                    "[TrlAgenticPpo] reward_mode='fn' requires 'reward_funcs' callable."
                )
            reward_fn_callable = raw_reward[0] if isinstance(raw_reward, list) else raw_reward
            reward_model = RewardFnWrapper(reward_fn=reward_fn_callable, tokenizer=tokenizer)
            logger.info(
                "[TrlAgenticPpo] reward_mode='fn': wrapped reward_fn into RewardFnWrapper ✓"
            )

        else:  # "auto"
            if reward_model is not None:
                logger.info("[TrlAgenticPpo] Auto-detected reward_model (nn.Module) ✓")
            elif raw_reward is not None:
                reward_fn_callable = raw_reward[0] if isinstance(raw_reward, list) else raw_reward
                reward_model = RewardFnWrapper(reward_fn=reward_fn_callable, tokenizer=tokenizer)
                logger.info("[TrlAgenticPpo] Auto-wrapped reward_fn into RewardFnWrapper ✓")
            else:
                raise ValueError("[TrlAgenticPpo] Provide 'reward_funcs' or 'reward_model'.")

        # ── Resolve value model ───────────────────────────────────────────
        value_model = trainer_kw.get("value_model") or self.kwargs.get("value_model")
        raw_value_fn = self.kwargs.get("value_fn")
        value_mode = _get(self.kwargs, "value_mode", default="auto")
        # value_mode options:
        #   "auto"    → use value_model if given, else create ValueModelWrapper (default)
        #   "model"   → force use of value_model nn.Module (error if not given)
        #   "wrapper" → force create ValueModelWrapper from policy backbone
        #   "fn"      → wrap a value_fn callable into a ValueFnWrapper

        if value_mode == "model":
            if value_model is None:
                raise ValueError(
                    "[TrlAgenticPpo] value_mode='model' requires 'value_model' (nn.Module)."
                )
            logger.info("[TrlAgenticPpo] Using explicit value_model (nn.Module) ✓")

        elif value_mode == "wrapper":
            hidden_size = _resolve_hidden_size(policy_model.config)
            value_model = ValueModelWrapper(policy_model, hidden_size)
            logger.info(
                f"[TrlAgenticPpo] value_mode='wrapper': created ValueModelWrapper "
                f"(hidden_size={hidden_size}) ✓"
            )

        elif value_mode == "fn":
            if raw_value_fn is None:
                raise ValueError("[TrlAgenticPpo] value_mode='fn' requires 'value_fn' callable.")
            hidden_size = _resolve_hidden_size(policy_model.config)
            value_model = ValueFnWrapper(
                value_fn=raw_value_fn, tokenizer=tokenizer, hidden_size=hidden_size
            )
            logger.info("[TrlAgenticPpo] value_mode='fn': wrapped value_fn into ValueFnWrapper ✓")

        else:  # "auto"
            if value_model is not None:
                logger.info("[TrlAgenticPpo] Auto-detected value_model (nn.Module) ✓")
            elif raw_value_fn is not None:
                hidden_size = _resolve_hidden_size(policy_model.config)
                value_model = ValueFnWrapper(
                    value_fn=raw_value_fn, tokenizer=tokenizer, hidden_size=hidden_size
                )
                logger.info("[TrlAgenticPpo] Auto-wrapped value_fn into ValueFnWrapper ✓")
            else:
                hidden_size = _resolve_hidden_size(policy_model.config)
                value_model = ValueModelWrapper(policy_model, hidden_size)
                logger.info(
                    f"[TrlAgenticPpo] Auto-created ValueModelWrapper "
                    f"(hidden_size={hidden_size}) ✓"
                )

        # ── PPOConfig defaults ────────────────────────────────────────────
        output_dir = _get(self.kwargs, "output_dir", default="./output/ppo_agentic")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        config_defaults: dict[str, Any] = {
            "output_dir": output_dir,
            "total_episodes": _get(self.kwargs, "total_episodes", default=1000),
            "per_device_train_batch_size": _get(
                self.kwargs, "per_device_train_batch_size", "batch_size", default=1
            ),
            "gradient_accumulation_steps": _get(
                self.kwargs, "gradient_accumulation_steps", default=1
            ),
            "learning_rate": _get(self.kwargs, "learning_rate", "lr", default=1e-6),
            "response_length": _get(self.kwargs, "response_length", "max_new_tokens", default=256),
            "temperature": _get(self.kwargs, "temperature", default=0.7),
            "kl_coef": _get(self.kwargs, "kl_coef", default=0.05),
            "cliprange": _get(self.kwargs, "cliprange", default=0.2),
            "cliprange_value": _get(self.kwargs, "cliprange_value", default=0.2),
            "vf_coef": _get(self.kwargs, "vf_coef", default=0.1),
            "gamma": _get(self.kwargs, "gamma", default=1.0),
            "lam": _get(self.kwargs, "lam", default=0.95),
            "num_ppo_epochs": _get(self.kwargs, "num_ppo_epochs", default=4),
            "num_mini_batches": _get(self.kwargs, "num_mini_batches", default=1),
            "local_rollout_forward_batch_size": _get(
                self.kwargs, "local_rollout_forward_batch_size", default=4
            ),
            "seed": _get(self.kwargs, "seed", default=42),
            "logging_steps": _get(self.kwargs, "logging_steps", default=10),
            "save_steps": _get(self.kwargs, "save_steps", default=100),
            "num_sample_generations": _get(self.kwargs, "num_sample_generations", default=0),
            "missing_eos_penalty": _get(self.kwargs, "missing_eos_penalty", default=None),
            "whiten_rewards": _get(self.kwargs, "whiten_rewards", default=False),
            "kl_estimator": _get(self.kwargs, "kl_estimator", default="k1"),
        }
        for k, v in config_defaults.items():
            config_kw.setdefault(k, v)

        ppo_config = PPOConfig(**config_kw)
        logger.info(
            f"[TrlAgenticPpo] PPOConfig built "
            f"(output_dir={ppo_config.output_dir}, agentic={self._agentic_mode})"
        )

        # ── Step 1: Build rollout_func ────────────────────────────────────
        rollout_func = None
        if self._agentic_mode:
            rollout_func = _build_ppo_rollout_fn(
                rollout_engine=_get(self.kwargs, "rollout_engine", default=None),
                rollout_backend=_get(self.kwargs, "rollout_backend", default=None),
                tools=_get(self.kwargs, "tools", default=None),
                max_steps=_get(self.kwargs, "max_steps_per_turn", "max_steps", default=20),
                system_prompt=_get(self.kwargs, "system_prompt", default=None),
                engine_kwargs=_get(self.kwargs, "engine_kwargs", default=None),
            )
            logger.info("[TrlAgenticPpo] rollout_func built ✓")

        # This pop is enough — pulls it out of trainer_kw so it's not passed twice
        peft_config = _resolve_peft_config(
            trainer_kw.pop("peft_config", None) or self.kwargs.get("peft_config")
        )

        # ── Step 2: Instantiate PPOTrainer ────────────────────────────────
        self.trainer = PPOTrainer(
            args=ppo_config,
            processing_class=tokenizer,
            model=policy_model,
            ref_model=self.kwargs.get("ref_model", None),
            reward_model=reward_model,
            train_dataset=self.train_dataset,
            value_model=value_model,
            data_collator=self.kwargs.get("data_collator", None),
            eval_dataset=self.eval_dataset,
            optimizers=self.kwargs.get("optimizers", (None, None)),
            callbacks=self.kwargs.get("callbacks", None),
            peft_config=peft_config,
        )
        _alias_ppo_printer_loss(self.trainer)
        logger.info("[TrlAgenticPpo] PPOTrainer instantiated ✓")

        # ── Step 3: Apply minimal patches ────────────────────────────────
        # We patch ONLY:
        #   - trl.trainer.ppo_trainer.batch_generation  (core training loop)
        #   - trainer.generate_completions              (eval sampling only)
        #
        # The entire train() loop, all PPO update maths, logging, and
        # checkpointing run exactly as TRL wrote them. No rewrite needed.
        if self._agentic_mode and rollout_func is not None:
            self.trainer.rollout_func = rollout_func

            _patch_batch_generation(self.trainer, rollout_func)

            if ppo_config.num_sample_generations > 0:
                _patch_generate_completions(self.trainer, rollout_func)

            tool_names = [
                getattr(t, "__name__", str(t)) for t in (_get(self.kwargs, "tools") or [])
            ]
            logger.info(f"[TrlAgenticPpo] Agentic patches applied. Tools: {tool_names}")

    # ─────────────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────────────

    def train(self) -> dict[str, Any]:
        self.setup_data()
        self.setup_trainer()

        logger.info(
            f"[TrlAgenticPpo] ▶ Starting "
            f"{'agentic ' if self._agentic_mode else ''}PPO training …"
        )
        t0 = time.time()
        self._train_result = self.trainer.train()
        elapsed = time.time() - t0
        logger.info(f"[TrlAgenticPpo] ✓ Done in {elapsed:.1f}s")

        self.save_model(_get(self.kwargs, "output_dir", default="./output/ppo_agentic"))
        return self.get_training_stats()

    # ─────────────────────────────────────────────────────────────────────
    # Save / Load / Stats
    # ─────────────────────────────────────────────────────────────────────

    def save_model(self, path: str | None = None, push_to_hub: bool = False, **extra) -> str:
        save_path = path or _get(self.kwargs, "output_dir", default="./output/ppo_agentic")
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
        with open(Path(save_path) / "ppo_config.yaml", "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)

        stats = {**self.get_training_stats(), **extra}
        with open(Path(save_path) / "training_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

        from agenttune.utils.provenance import write_provenance

        write_provenance(
            save_path,
            method="agentic.ppo",
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
        ).from_pretrained(
            path,
            device_map=kwargs.pop("device_map", "auto"),
            **kwargs,
        )
        if self.trainer:
            self.trainer.policy_model = model
            self.trainer.processing_class = tokenizer
        else:
            self.kwargs["model"] = model
            self.kwargs["processing_class"] = tokenizer

    def get_training_stats(self) -> dict[str, Any]:
        tr = self._train_result
        metrics = getattr(tr, "metrics", {}) if tr else {}
        history = []
        global_step = getattr(tr, "global_step", None) if tr is not None else None
        if self.trainer is not None and getattr(self.trainer, "state", None) is not None:
            history = list(self.trainer.state.log_history or [])
            if global_step is None:
                global_step = getattr(self.trainer.state, "global_step", None)
        if not metrics and history:
            metrics = dict(history[-1])
        final_loss = getattr(tr, "training_loss", None) if tr is not None else None
        if final_loss is None and isinstance(metrics, dict):
            final_loss = metrics.get(
                "train_loss", metrics.get("loss", metrics.get("loss/policy_avg"))
            )
        tools = _get(self.kwargs, "tools")
        return {
            "model": str(_get(self.kwargs, "model", default="unknown")),
            "output_dir": _get(self.kwargs, "output_dir", default="./output/ppo_agentic"),
            "agentic_mode": self._agentic_mode,
            "tools": [getattr(t, "__name__", str(t)) for t in tools] if tools else [],
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "final_loss": final_loss,
            "total_steps": global_step,
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
