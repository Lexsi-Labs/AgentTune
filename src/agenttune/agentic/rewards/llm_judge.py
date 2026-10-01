"""
agenttune.agentic.rewards.llm_judge
=====================================
LLM-as-judge trajectory evaluator.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Score dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class JudgeScore:
    score: float  # normalised [0, 1]
    explanation: str = ""
    trajectory_id: str | None = None
    raw_response: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Default rubrics
# ─────────────────────────────────────────────────────────────────────────────

ABSOLUTE_RUBRIC = """
You are evaluating a single AI agent trajectory against a task.
Score the trajectory from 0.0 to 1.0 where:
  1.0 = task fully accomplished, efficient, no errors
  0.5 = partial progress or significant inefficiency
  0.0 = task not accomplished, harmful, or completely wrong

Criteria:
- Goal completion carries the most weight.
- Penalise unnecessary tool calls, loops, or detours.
- Give partial credit for meaningful progress.
""".strip()

RELATIVE_RUBRIC = """
You are comparing multiple AI agent trajectories that were all given the same task.
Score each trajectory from 0.0 to 1.0 relative to the others:
  - A trajectory that accomplishes the goal MUST score higher than one that does not.
  - Prefer trajectories that are more efficient (fewer unnecessary steps).
  - If one is only slightly better, the score gap should be small.
  - You MAY give partial credit for progress towards the goal.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────


def _format_trajectory(trajectory: Any, traj_id: str | None = None) -> str:
    lines = []
    if traj_id:
        lines.append(f'<trajectory id="{traj_id}">')

    if hasattr(trajectory, "steps"):
        for s in trajectory.steps:
            action = getattr(s, "action", {})
            observation = getattr(s, "observation", "")
            thought = getattr(s, "thought", "")
            lines.append(
                f"  Step {getattr(s, 'step_number', '?')}: "
                f"action={json.dumps(action)}  obs={str(observation)[:300]}"
            )
            if thought and thought != observation:
                lines.append(f"    thought: {str(thought)[:200]}")
    elif isinstance(trajectory, list) and trajectory and isinstance(trajectory[0], dict):
        lines.append(json.dumps(trajectory, indent=2))
    else:
        lines.append(str(trajectory))

    if traj_id:
        lines.append("</trajectory>")
    return "\n".join(lines)


def _build_absolute_prompt(
    task: str,
    trajectory: Any,
    criteria: dict[str, float],
    rubric: str,
    system_prompt: str | None,
) -> tuple:
    system = system_prompt if system_prompt is not None else rubric
    crit_text = "\n".join(f"  - {k} (weight {v:.2f})" for k, v in criteria.items())
    traj_text = _format_trajectory(trajectory)
    user = (
        f"Task: {task}\n\n"
        + (f"Criteria:\n{crit_text}\n\n" if criteria else "")
        + f"Trajectory:\n{traj_text}\n\n"
        + 'Respond ONLY with a JSON object: {"score": <float 0-1>, "explanation": "<one sentence>"}'
    )
    return system, user


def _build_relative_prompt(
    task: str,
    trajectories: list[Any],
    rubric: str,
    system_prompt: str | None,
) -> tuple:
    system = system_prompt if system_prompt is not None else rubric
    traj_blocks = "\n\n".join(
        _format_trajectory(t, traj_id=str(i + 1)) for i, t in enumerate(trajectories)
    )
    ids = [str(i + 1) for i in range(len(trajectories))]
    user = (
        f"Task: {task}\n\n"
        f"Trajectories:\n{traj_blocks}\n\n"
        f"Respond ONLY with a JSON object:\n"
        f'{{"scores": [{{"id": "{ids[0]}", "score": 0.0, "explanation": "one sentence"}}, '
        f'{{"id": "{ids[-1]}", "score": 0.0, "explanation": "one sentence"}}]}}\n'
        f"Include exactly one entry per trajectory id: {ids}"
    )
    return system, user


def _messages_from(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ─────────────────────────────────────────────────────────────────────────────
# Response parsers
# ─────────────────────────────────────────────────────────────────────────────


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n")[1:] if not l.strip().startswith("```")).strip()
    return text


def _parse_absolute(text: str) -> JudgeScore:
    text = _strip_fences(text)
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict) or "score" not in obj:
            raise ValueError("missing score")
        return JudgeScore(
            score=max(0.0, min(1.0, float(obj["score"]))),
            explanation=str(obj.get("explanation", "")),
            raw_response=text,
        )
    except Exception:
        import re

        m = re.search(r"\b([01](?:\.\d+)?)\b", text)
        if m:
            return JudgeScore(score=float(m.group(1)), raw_response=text)
        return JudgeScore(
            score=0.0,
            explanation="unparsed judge output",
            raw_response=text,
        )


def _parse_relative(text: str, n: int) -> list[JudgeScore]:
    text = _strip_fences(text)
    expected = {str(i + 1) for i in range(n)}
    try:
        obj = json.loads(text)
        entries = obj.get("scores", obj) if isinstance(obj, dict) else obj
        if not isinstance(entries, list) or len(entries) != n:
            raise ValueError
        result = [
            JudgeScore(
                score=max(0.0, min(1.0, float(e["score"]))),
                explanation=str(e.get("explanation", "")),
                trajectory_id=str(e.get("id", "")),
                raw_response=text,
            )
            for e in entries
        ]
        if {s.trajectory_id for s in result} != expected:
            raise ValueError
        return result
    except Exception:
        return [
            JudgeScore(
                score=0.0,
                explanation="unparsed judge output",
                raw_response=text,
            )
            for _ in range(n)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Self-contained local backends
# ─────────────────────────────────────────────────────────────────────────────


class _TransformersLocalJudge:
    """Loads and owns a HuggingFace transformers model for judging."""

    def __init__(self, model_path: str, gen_kwargs: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.gen_kwargs = gen_kwargs
        logger.info(f"[LLMJudge/transformers] Loading {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._device = next(self.model.parameters()).device
        logger.info("[LLMJudge/transformers] Ready.")

    def call(self, messages: list[dict]) -> str:
        import torch

        # AFTER
        ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )

        # Handle both raw tensor and BatchEncoding dict
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids

        ids = ids.to(self._device)

        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=self.gen_kwargs.get("max_new_tokens", 256),
                temperature=self.gen_kwargs.get("temperature", 0.0),
                do_sample=self.gen_kwargs.get("do_sample", False),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new = out[:, ids.shape[1] :]
        return self.tokenizer.decode(new[0], skip_special_tokens=True)


class _VLLMLocalJudge:
    """
    Judge backend built on TRL's VLLMGeneration (colocate mode).

    Why not bare ``vllm.LLM``?
    --------------------------
    * ``vllm.LLM`` spawns a separate engine-core subprocess via ZMQ.  In many
      notebook / single-GPU / restricted environments that subprocess fails to
      handshake (``RuntimeError: Engine core initialization failed``).
    * TRL's ``VLLMGeneration`` runs in *colocate* mode — the vLLM engine lives
      in the same process, avoiding the multiprocessing spawn entirely.

    We own the model + processing_class here (no external RolloutEngine
    needed), and use the same generate() path as VLLMRolloutEngine so the
    behaviour is identical.
    """

    def __init__(self, model_path: str, gen_kwargs: dict, vllm_kwargs: dict):
        try:
            from trl.generation.vllm_generation import VLLMGeneration
        except ImportError as e:
            raise ImportError(  # noqa: B904
                "TRL is required for backend='vllm'. Install with: pip install trl\n"
                f"Original error: {e}"
            )
        try:
            import torch
            from accelerate import Accelerator
            from transformers import AutoModelForCausalLM, AutoProcessor, ProcessorMixin
        except ImportError as e:
            raise ImportError(f"transformers + accelerate required: {e}")  # noqa: B904

        self.gen_kwargs = gen_kwargs
        trust = vllm_kwargs.get("trust_remote_code", False)

        logger.info(f"[LLMJudge/vllm] Loading model: {model_path}")
        dtype_str = vllm_kwargs.get("dtype", None)
        model_kwargs = {"trust_remote_code": trust}
        if dtype_str:
            model_kwargs["torch_dtype"] = getattr(torch, dtype_str)

        model = AutoModelForCausalLM.from_pretrained(model_path)

        processing_class = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust,
            truncation_side="left",
            padding_side="left",
        )
        # Grab the underlying tokenizer for chat-template rendering
        self.tokenizer = (
            processing_class.tokenizer
            if isinstance(processing_class, ProcessorMixin)
            else processing_class
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        accelerator = Accelerator()

        gpu_mem = vllm_kwargs.get("gpu_memory_utilization", 0.4)
        tp_size = vllm_kwargs.get("tensor_parallel_size", 1)
        max_len = vllm_kwargs.get("max_model_len", None)
        max_seqs = vllm_kwargs.get("max_num_seqs", 8)
        max_comp = gen_kwargs.get("max_new_tokens", 256)
        temp = gen_kwargs.get("temperature", 0.0)

        logger.info("[LLMJudge/vllm] Initialising TRL VLLMGeneration (colocate) ...")
        self.vllm_gen = VLLMGeneration(
            model=model,
            accelerator=accelerator,
            is_fsdp_enabled=False,
            processing_class=processing_class,
            mode="colocate",
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_mem,
            max_model_length=max_len,
            max_num_seqs=max_seqs,
            enable_sleep_mode=False,
            temperature=temp,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            max_completion_length=max_comp,
            repetition_penalty=1.0,
            structured_outputs_regex=None,
            # chat_template=None,
            # chat_template_kwargs={},
            # tools=[],
            # rollout_func=None,
        )
        logger.info("[LLMJudge/vllm] Ready.")
        del model

    # AFTER
    def call(self, messages: list[dict]) -> str:
        import contextlib
        import inspect

        # tokenize to IDs instead of string
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,  # ← IDs not string
        )
        if hasattr(token_ids, "input_ids"):
            token_ids = token_ids.input_ids
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        # ensure flat list of ints, not nested
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        _gen_sig = inspect.signature(self.vllm_gen.generate).parameters
        _gen_kwargs = {
            "prompts": [token_ids],  # ← pass IDs not string
            "num_generations": 1,
            "profiler": contextlib.nullcontext(),
            "images": None,
        }
        _supported_gen = {k: v for k, v in _gen_kwargs.items() if k in _gen_sig}

        result = self.vllm_gen.generate(**_supported_gen)

        completion_ids = result[1] if isinstance(result, tuple) else result
        if hasattr(completion_ids, "tolist"):
            completion_ids = completion_ids.tolist()

        ids = completion_ids[0] if completion_ids else []
        if ids and isinstance(ids[0], list):
            ids = ids[0]

        return self.tokenizer.decode(ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# API backend helpers  (sync + async pairs)
# ─────────────────────────────────────────────────────────────────────────────


def _call_openai_compat(messages, model, api_key, base_url, max_tokens) -> str:
    from openai import OpenAI

    resp = OpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


async def _acall_openai_compat(messages, model, api_key, base_url, max_tokens) -> str:
    from openai import AsyncOpenAI

    resp = await AsyncOpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(messages, model, api_key, max_tokens) -> str:
    import anthropic

    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    usr_msgs = [m for m in messages if m["role"] != "system"]
    resp = anthropic.Anthropic(api_key=api_key).messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=usr_msgs + [{"role": "assistant", "content": "{"}],
    )
    return "{" + resp.content[0].text


async def _acall_anthropic(messages, model, api_key, max_tokens) -> str:
    import anthropic

    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    usr_msgs = [m for m in messages if m["role"] != "system"]
    resp = await anthropic.AsyncAnthropic(api_key=api_key).messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=usr_msgs + [{"role": "assistant", "content": "{"}],
    )
    return "{" + resp.content[0].text


def _call_rollout_engine(messages, engine, gen_kwargs) -> str:
    return engine.generate(prompts=messages, tools=[], gen_cfg=gen_kwargs)["completions"][0]


async def _acall_rollout_engine(messages, engine, gen_kwargs) -> str:
    import asyncio

    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: _call_rollout_engine(messages, engine, gen_kwargs)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cache key
# ─────────────────────────────────────────────────────────────────────────────


def _cache_key(task: str, trajectory: Any, criteria: dict) -> str:
    if hasattr(trajectory, "steps"):
        raw = task + str(
            [(getattr(s, "action", ""), getattr(s, "observation", "")) for s in trajectory.steps]
        )
    elif isinstance(trajectory, list):
        raw = task + json.dumps(trajectory, sort_keys=True)
    else:
        raw = task + str(trajectory)
    # Cache key only — not a security digest (hence usedforsecurity=False).
    return hashlib.md5(
        (raw + json.dumps(criteria, sort_keys=True)).encode(), usedforsecurity=False
    ).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# LLMJudge
# ─────────────────────────────────────────────────────────────────────────────


class LLMJudge:
    """
    LLM-as-judge trajectory evaluator.

    See module docstring for full usage examples.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        backend: Literal["transformers", "vllm"] | None = None,
        model_path: str | None = None,
        rollout_engine: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        # ── Prompting ─────────────────────────────────────────────────────────
        system_prompt: str | None = None,
        absolute_rubric: str = ABSOLUTE_RUBRIC,
        relative_rubric: str = RELATIVE_RUBRIC,
        # ── Misc ──────────────────────────────────────────────────────────────
        cache_size: int = 10_000,
        judge_max_tokens: int = 256,
        local_gen_kwargs: dict | None = None,
        vllm_kwargs: dict | None = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.absolute_rubric = absolute_rubric
        self.relative_rubric = relative_rubric
        self.cache_size = cache_size
        self.judge_max_tokens = judge_max_tokens
        self._api_key = api_key
        self._base_url = base_url
        self._cache: dict[str, float] = {}

        self.local_gen_kwargs = local_gen_kwargs or {
            "max_new_tokens": 256,
            "temperature": 0.0,
            "do_sample": False,
        }

        # ── Build local backend or record external engine ─────────────────────
        self._local_judge = None
        self._rollout_engine = None

        if rollout_engine is not None:
            self._rollout_engine = rollout_engine
            self._provider = "external_engine"

        elif backend == "transformers":
            if not model_path:
                raise ValueError("model_path required for backend='transformers'")
            self._local_judge = _TransformersLocalJudge(model_path, self.local_gen_kwargs)
            self._provider = "local_transformers"

        elif backend == "vllm":
            if not model_path:
                raise ValueError("model_path required for backend='vllm'")
            self._local_judge = _VLLMLocalJudge(
                model_path, self.local_gen_kwargs, vllm_kwargs or {}
            )
            self._provider = "local_vllm"

        else:
            # API — detect from model string
            if model.startswith("claude"):
                self._provider = "anthropic"
            elif "/" in model:
                self._provider = "openrouter"
            else:
                self._provider = "openai"

    # ── Key resolution ────────────────────────────────────────────────────────

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        env = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
        }.get(self._provider, "OPENAI_API_KEY")
        key = os.getenv(env)
        if not key:
            raise ValueError(f"Set {env} or pass api_key=")
        return key

    def _resolve_base_url(self) -> str:
        if self._base_url:
            return self._base_url
        return (
            "https://openrouter.ai/api/v1"
            if self._provider == "openrouter"
            else "https://api.openai.com/v1"
        )

    # ── Sync dispatch ─────────────────────────────────────────────────────────

    def _call(self, messages: list[dict]) -> str:
        if self._provider in ("local_transformers", "local_vllm"):
            return self._local_judge.call(messages)
        if self._provider == "external_engine":
            return _call_rollout_engine(messages, self._rollout_engine, self.local_gen_kwargs)
        if self._provider == "anthropic":
            return _call_anthropic(
                messages, self.model, self._resolve_api_key(), self.judge_max_tokens
            )
        return _call_openai_compat(
            messages,
            self.model,
            self._resolve_api_key(),
            self._resolve_base_url(),
            self.judge_max_tokens,
        )

    # ── Async dispatch ────────────────────────────────────────────────────────

    async def _acall(self, messages: list[dict]) -> str:
        import asyncio

        if self._provider in ("local_transformers", "local_vllm"):
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._local_judge.call(messages)
            )
        if self._provider == "external_engine":
            return await _acall_rollout_engine(
                messages, self._rollout_engine, self.local_gen_kwargs
            )
        if self._provider == "anthropic":
            return await _acall_anthropic(
                messages, self.model, self._resolve_api_key(), self.judge_max_tokens
            )
        return await _acall_openai_compat(
            messages,
            self.model,
            self._resolve_api_key(),
            self._resolve_base_url(),
            self.judge_max_tokens,
        )

    # ── Public sync ───────────────────────────────────────────────────────────

    def _score_absolute(
        self,
        task: str,
        trajectory: Any,
        criteria: dict[str, float] | None = None,
    ) -> JudgeScore:
        criteria = criteria or {}
        key = _cache_key(task, trajectory, criteria)
        if key in self._cache:
            return JudgeScore(score=self._cache[key], explanation="cached")

        system, user = _build_absolute_prompt(
            task, trajectory, criteria, self.absolute_rubric, self.system_prompt
        )
        try:
            parsed = _parse_absolute(self._call(_messages_from(system, user)))
        except Exception:
            import traceback

            traceback.print_exc()
            parsed = JudgeScore(score=0.0, explanation="judge call failed")

        if 0 < self.cache_size and len(self._cache) < self.cache_size:
            self._cache[key] = parsed.score
        return parsed

    def evaluate_trajectory(
        self,
        task: str,
        trajectory: Any,
        criteria: dict[str, float] | None = None,
    ) -> float:
        """Score a single trajectory. Returns float in [0, 1]."""
        return self._score_absolute(task, trajectory, criteria).score

    def evaluate_batch(
        self,
        task: str,
        trajectories: list[Any],
        mode: Literal["absolute", "relative"] = "absolute",
        criteria: dict[str, float] | None = None,
    ) -> list[JudgeScore]:
        """
        Score a list of trajectories.

        mode="relative"  all shown to judge at once — better signal for GRPO
        mode="absolute"  each scored independently
        """
        if not trajectories:
            return []

        if mode == "relative":
            system, user = _build_relative_prompt(
                task, trajectories, self.relative_rubric, self.system_prompt
            )
            try:
                return _parse_relative(self._call(_messages_from(system, user)), len(trajectories))
            except Exception:
                return [
                    JudgeScore(score=0.0, explanation="unparsed judge output") for _ in trajectories
                ]

        return [self._score_absolute(task, t, criteria) for t in trajectories]

    # ── Public async ──────────────────────────────────────────────────────────

    async def async_evaluate_trajectory(
        self,
        task: str,
        trajectory: Any,
        criteria: dict[str, float] | None = None,
    ) -> float:
        """Async version of evaluate_trajectory."""
        criteria = criteria or {}
        key = _cache_key(task, trajectory, criteria)
        if key in self._cache:
            return self._cache[key]

        system, user = _build_absolute_prompt(
            task, trajectory, criteria, self.absolute_rubric, self.system_prompt
        )
        try:
            score = _parse_absolute(await self._acall(_messages_from(system, user))).score
        except Exception:
            score = 0.0

        if 0 < self.cache_size and len(self._cache) < self.cache_size:
            self._cache[key] = score
        return score

    async def async_evaluate_batch(
        self,
        task: str,
        trajectories: list[Any],
        mode: Literal["absolute", "relative"] = "relative",
        criteria: dict[str, float] | None = None,
    ) -> list[JudgeScore]:
        """Async batch. absolute=concurrent gather, relative=single call."""
        import asyncio

        if not trajectories:
            return []

        if mode == "relative":
            system, user = _build_relative_prompt(
                task, trajectories, self.relative_rubric, self.system_prompt
            )
            try:
                return _parse_relative(
                    await self._acall(_messages_from(system, user)), len(trajectories)
                )
            except Exception:
                return [
                    JudgeScore(score=0.0, explanation="unparsed judge output") for _ in trajectories
                ]

        results = await asyncio.gather(
            *[self.async_evaluate_trajectory(task, t, criteria) for t in trajectories],
            return_exceptions=True,
        )
        return [
            (
                JudgeScore(score=r)
                if isinstance(r, float)
                else JudgeScore(score=0.0, explanation="judge call failed")
            )
            for r in results
        ]

    # ── reward_fn adapter ─────────────────────────────────────────────────────

    def as_reward_fn(self, criteria: dict[str, float] | None = None):
        """
        Returns a callable for create_rollout_fn(reward_fn=...).

            rollout_fn = create_rollout_fn(..., reward_fn=judge.as_reward_fn())
        """

        def _fn(trajectory) -> float:
            return self.evaluate_trajectory(getattr(trajectory, "task", ""), trajectory, criteria)

        return _fn

    def clear_cache(self):
        self._cache.clear()

    def __repr__(self) -> str:
        label = {
            "local_transformers": "Transformers(local)",
            "local_vllm": "vLLM(local)",
            "external_engine": type(self._rollout_engine).__name__,
            "anthropic": f"Anthropic({self.model})",
            "openrouter": f"OpenRouter({self.model})",
            "openai": f"OpenAI({self.model})",
        }.get(self._provider, self._provider)
        return f"LLMJudge(backend={label}, cached={len(self._cache)})"
