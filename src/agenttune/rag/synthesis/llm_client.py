"""
LLM client with per-call metadata tracking.

Wraps a Groq (OpenAI-compatible) chat client and records token usage, latency,
and cost for every call — the metadata backbone for reproducibility/cost
accounting. The client itself is injectable (`LLMClient` Protocol), so the
pipeline logic is testable with a `FakeLLMClient` on CPU with no network/key
(same pattern as `datagen.py`'s `QAGenerator`/`Solver` callables).

Groq pricing is model-dependent; `PRICING` holds per-1M-token rates and is
updated as models change. Cost = prompt_price * prompt_tokens/1e6 +
completion_price * completion_tokens/1e6.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Protocol

from .schema import LLMCallRecord


def _safe_max_workers(requested: int) -> int:
    """Clamp the thread-pool size to what this process can actually spawn.

    Containers (vast.ai, docker defaults) cap threads/PIDs via cgroup
    `pids.max` — the box this pipeline runs on has pids.max=768 with ~165
    already in use by venv/torch/GPU, so ThreadPoolExecutor(max_workers=800)
    dies mid-submit with RuntimeError("can't start new thread") (observed).
    Reads the cgroup limit, subtracts current usage + a safety margin, and
    clamps; falls back to 512 on hosts without cgroup files. Returns at
    least 8 so a pathological limit never deadlocks the pipeline.
    """
    try:
        with open("/sys/fs/cgroup/pids.max") as f:
            raw = f.read().strip()
        if raw != "max":
            limit = int(raw)
            try:
                with open("/sys/fs/cgroup/pids.current") as f:
                    current = int(f.read().strip())
            except Exception:
                current = 0
            return max(8, min(requested, limit - current - 64))
    except Exception:
        pass
    return max(8, min(requested, 512))


def parallel_chat(
    llm, messages_list: list[list[dict]], *, max_workers: int = 4, **chat_kw
) -> list[tuple[str, LLMCallRecord]]:
    """Run many `llm.chat()` calls concurrently with a bounded thread pool.

    LLM calls are I/O-bound, so threads give real parallelism. `max_workers`
    defaults to 4 (a conservative floor) but callers pass their endpoint's
    budget: DeepSeek's official API sustains hundreds of concurrent requests
    (measured: 400 in flight → ~93 req/s, zero errors), so Stage 0/2/3
    batches fan out to hundreds of workers for near-linear wall-clock
    speedups. The pool size is clamped by `_safe_max_workers` so it can
    never crash the container. Preserves input order.

    Each `messages` in `messages_list` is passed to `llm.chat(messages, **chat_kw)`.
    """
    workers = _safe_max_workers(max(1, max_workers))

    def _one(msgs):
        return llm.chat(msgs, **chat_kw)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, messages_list))


# Per-1M-token USD pricing (prompt, completion). Groq models used here.
# qwen3.6-27b = the generation/verifier model the user picked.
PRICING = {
    "qwen/qwen3.6-27b": (0.29, 0.39),
    "qwen3.6-27b": (0.29, 0.39),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    # DeepSeek official (2026-08-13): flash = $0.14/M input (cache miss,
    # $0.0028 cache hit), $0.28/M output. Pro = $0.435/$0.87.
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-v4-pro": (0.435, 0.87),
    "default": (0.50, 0.80),  # fallback estimate
}


class LLMClient(Protocol):
    """Interface every LLM client implements. Injectable for testing."""

    model: str

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stage: str,
        purpose: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[str, LLMCallRecord]:
        """Return (response_text, call_record)."""
        ...


def _cost(model: str, pt: int, ct: int) -> float:
    p = PRICING.get(model, PRICING["default"])
    return p[0] * pt / 1e6 + p[1] * ct / 1e6


class GroqLLMClient:
    """Real Groq client. Uses the `groq` SDK, which scopes to Groq's API itself
    (no base_url — passing one doubles the path to /openai/v1/openai/v1/...).

    Note on the default model: `qwen/qwen3.6-27b` is a reasoning model — it emits
    a thinking trace before the final answer, so callers must pass a generous
    `max_tokens` (>= 1024) for the JSON answer to survive past the trace.
    """

    def __init__(
        self,
        model: str = "qwen/qwen3.6-27b",
        api_key: str | None = None,
        base_url: str | None = None,
        json_mode: bool = True,
    ):
        import os

        from groq import Groq

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not set (env var or api_key=...).")
        self.model = model
        self.json_mode = json_mode
        kwargs = {"api_key": key}
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = Groq(**kwargs)

    def chat(self, messages, *, stage, purpose, temperature=0.0, max_tokens=12288):
        t0 = time.perf_counter()
        err = ""
        success = True
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            pt = usage.prompt_tokens if usage else 0
            ct = usage.completion_tokens if usage else 0
        except Exception as e:
            text, pt, ct = "", 0, 0
            err, success = repr(e), False
        latency = (time.perf_counter() - t0) * 1000.0
        rec = LLMCallRecord(
            stage=stage,
            purpose=purpose,
            model=self.model,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=latency,
            cost_usd=_cost(self.model, pt, ct),
            request_preview=_truncate(_messages_preview(messages)),
            response_preview=_truncate(text),
            success=success,
            error=err,
            timestamp=datetime.now(UTC).isoformat(),
        )
        return text, rec


class OpenAICompatLLMClient:
    """OpenAI-compatible chat client — works with any /v1/chat/completions endpoint
    (vLLM, TGI, Together, hosted OpenAI-compatible providers, local LLM servers,
    etc.).

    Used when Groq's quota is exhausted or for self-hosted endpoints. Same
    per-call metadata tracking as GroqLLMClient. Has a per-call timeout + retry
    so a single hanging endpoint call fails fast instead of stalling the batch
    (some providers occasionally stall on a request).

    Qwen3 reasoning models (e.g. Qwen3.6-35B-A3B) emit a thinking trace before
    the answer by default — this burns the token budget (a 128-token solver call
    is consumed entirely by the trace, so the JSON answer never appears →
    spurious UNANSWERABLE → 0.0 answerability) and makes generation echo the
    prompt template. Pass `enable_thinking=False` (default for Qwen3 models) to
    disable the trace via `extra_body={"chat_template_kwargs": ...}`. This is
    NOT OpenAI-standard; non-Qwen endpoints ignore it. Verified working on a
    hosted Qwen3.6 endpoint (7 tokens, clean JSON, vs 128 tokens of trace).
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 90.0,
        max_retries: int = 2,
        enable_thinking: bool | None = None,
        extra_body: dict | None = None,
        json_mode: bool = True,
        reasoning_effort: str | None = "low",
    ):
        from openai import OpenAI

        self.model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self.json_mode = json_mode
        self.reasoning_effort = reasoning_effort
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        # Auto-disable thinking for Qwen3 reasoning models unless explicitly set.
        # The reasoning trace is the root cause of the 0.0 answerability +
        # placeholder-generation failures seen on the hosted Qwen3.6 endpoint.
        is_qwen3 = "qwen3" in model.lower()
        if enable_thinking is None:
            enable_thinking = False if is_qwen3 else None
        # DeepSeek v4 (official API): thinking is ON by DEFAULT with effort
        # HIGH, and its trace is MASSIVE (measured 17K-35K chars ≈ 4-9K tokens
        # — far more than the JSON needs). max_tokens is a SHARED budget for
        # thinking + content: when the trace eats it all (common under parallel
        # load on long prompts), the API returns content="" (empty) — the root
        # cause of the parse-failure storms AND the cost blowup (each empty
        # response still bills the full max_tokens). Fix (official mechanism):
        # pass reasoning_effort as a TOP-LEVEL OpenAI param (mapping low→low,
        # medium→high, high→high, max→max) so thinking is bounded while still
        # enabled for quality. Default "low" — measured 4x fewer completion
        # tokens with content intact.
        is_deepseek = "deepseek" in model.lower() or "deepseek.com" in (base_url or "")
        self._is_deepseek = is_deepseek
        # Build the base extra_body: Qwen3 sampling params + thinking control.
        # These match the official Qwen3 serving recipe (top_k=20, top_p=0.8,
        # presence_penalty=1.5). Callers can override/extend via `extra_body`.
        base_extra: dict[str, Any] = {}
        if is_qwen3:
            base_extra["top_k"] = 20
        if enable_thinking is not None:
            base_extra["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        # DeepSeek official pattern (docs 2026-08-13): reasoning_effort is a
        # TOP-LEVEL param (low/medium/high/max), and the thinking toggle goes
        # in extra_body: {"thinking": {"type": "enabled"}}. Both together =
        # bounded thinking at the requested effort, so content always
        # completes within max_tokens (no empty-content parse failures).
        if is_deepseek:
            base_extra["thinking"] = {"type": "enabled"}
        if extra_body:
            base_extra.update(extra_body)
        self._extra_body = base_extra or None

    def chat(
        self,
        messages,
        *,
        stage,
        purpose,
        temperature=0.0,
        max_tokens=12288,
        json_mode: bool | None = None,
    ):
        t0 = time.perf_counter()
        err = ""
        success = True
        text, pt, ct = "", 0, 0
        last_err = None
        for _attempt in range(self._max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                # DeepSeek official: reasoning_effort is a TOP-LEVEL OpenAI
                # param (thinking toggle lives in extra_body). Bounds the
                # thinking trace so content always completes within max_tokens
                # (the empty-content root cause) — see __init__.
                if self._is_deepseek and self.reasoning_effort is not None:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                # NOTE: no `response_format=json_object` — DeepSeek's json-object
                # mode returned literal EMPTY content under parallel load (~70%),
                # but plain prompting + extract_json is reliable.
                if self._extra_body is not None:
                    kwargs["extra_body"] = self._extra_body
                resp = self._client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                usage = resp.usage
                pt = usage.prompt_tokens if usage else 0
                ct = usage.completion_tokens if usage else 0
                # documented DeepSeek quirk: may return EMPTY content — retry
                # like a timeout instead of shipping a parse failure
                if not text.strip():
                    last_err = RuntimeError("empty content (DeepSeek JSON-mode quirk)")
                    continue
                last_err = None
                break
            except Exception as e:
                last_err = e
                # retry on timeout/connection errors; bail on auth/400-style
                msg = repr(e).lower()
                if any(k in msg for k in ("timeout", "connection", "read timed out", "reset")):
                    continue
                else:
                    break
        if last_err is not None:
            text, pt, ct = "", 0, 0
            err, success = repr(last_err), False
        latency = (time.perf_counter() - t0) * 1000.0
        rec = LLMCallRecord(
            stage=stage,
            purpose=purpose,
            model=self.model,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=latency,
            cost_usd=_cost(self.model, pt, ct),
            request_preview=_truncate(_messages_preview(messages)),
            response_preview=_truncate(text),
            success=success,
            error=err,
            timestamp=datetime.now(UTC).isoformat(),
        )
        return text, rec


class FakeLLMClient:
    """Deterministic fake for CPU testing — no network, no key.

    `responder(purpose, messages) -> str` decides the response. Default returns
    a fixed string; tests pass a callable to script multi-call flows (e.g.
    generate → verify → revise).
    """

    def __init__(self, model: str = "fake/test", responder=None):
        self.model = model
        self._responder = responder or (lambda purpose, messages: "FAKE_RESPONSE")
        self.calls: list[LLMCallRecord] = []

    def chat(self, messages, *, stage, purpose, temperature=0.0, max_tokens=12288):
        text = self._responder(purpose, messages)
        rec = LLMCallRecord(
            stage=stage,
            purpose=purpose,
            model=self.model,
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=0.1,
            cost_usd=0.0,
            request_preview=_truncate(_messages_preview(messages)),
            response_preview=_truncate(text),
            timestamp=datetime.now(UTC).isoformat(),
        )
        self.calls.append(rec)
        return text, rec


def _messages_preview(messages: list[dict[str, str]]) -> str:
    parts = []
    for m in messages:
        parts.append(f"[{m.get('role','?')}] {m.get('content','')}")
    return "\n".join(parts)


def _truncate(s: str, n: int = 500) -> str:
    return s if len(s) <= n else s[:n] + "…"
