# agenttune/agentic/engines/api_engine.py
"""
Universal API rollout engine.

Supports any provider that LiteLLM speaks to:
    openai       → model="gpt-4o"
    anthropic    → model="claude-sonnet-4-5"
    openrouter   → model="openrouter/meta-llama/llama-3.1-70b-instruct"
    groq         → model="groq/llama-3.3-70b-versatile"
    ollama       → model="ollama/llama3"
    together     → model="together_ai/mistralai/Mixtral-8x7B-v0.1"
    azure        → model="azure/my-deployment"
    litellm proxy→ model="gpt-4o-mini", base_url="http://localhost:4000"

Install:  pip install litellm
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Any

from .base import RolloutEngine

# ─────────────────────────────────────────────────────────────────────────────
# Provider helpers
# ─────────────────────────────────────────────────────────────────────────────

# Providers that support tools but must NOT receive tool_choice="auto"
# (they either ignore it, error, or behave unexpectedly)
_NO_TOOL_CHOICE_PROVIDERS = {"groq", "ollama", "together_ai", "together"}

# Providers that do not support tool/function calling at all
_NO_TOOLS_PROVIDERS: set[str] = set()

# Default env-var names per provider prefix
_ENV_VAR_MAP: dict[str, str] = {
    "gpt": "OPENAI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "o1": "OPENAI_API_KEY",
    "o3": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "together": "TOGETHERAI_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "vertex_ai": "VERTEXAI_PROJECT",
    "azure": "AZURE_API_KEY",
    "perplexity": "PERPLEXITYAI_API_KEY",
    "replicate": "REPLICATE_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
    "fireworks_ai": "FIREWORKS_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "ollama": None,  # no key needed
}


def _provider_prefix(model: str) -> str:
    """
    Extract the provider prefix from a model string.

    Examples
    --------
    "groq/llama-3.3-70b-versatile"         → "groq"
    "openrouter/meta-llama/llama-3.1-8b"   → "openrouter"
    "gpt-4o-mini"                           → "gpt"   (OpenAI bare name)
    "claude-haiku-4-5-20251001"             → "claude" (Anthropic bare name)
    """
    if "/" in model:
        return model.split("/")[0].lower()
    # Bare model names — match by prefix
    for prefix in _ENV_VAR_MAP:
        if model.lower().startswith(prefix):
            return prefix
    return model.lower()


def _resolve_api_key(model: str, explicit_key: str | None) -> str | None:
    """Return explicit key if given, else look up the env-var for this provider."""
    if explicit_key:
        return explicit_key
    prefix = _provider_prefix(model)
    env_var = _ENV_VAR_MAP.get(prefix)
    if env_var is None:
        return None  # e.g. ollama — no key needed
    key = os.environ.get(env_var)
    if not key:
        warnings.warn(
            f"No API key found for provider '{prefix}'. "
            f"Set {env_var} or pass api_key= to APIRolloutEngine.",
            UserWarning,
            stacklevel=3,
        )
    return key


# ─────────────────────────────────────────────────────────────────────────────
# Message normalisation
# ─────────────────────────────────────────────────────────────────────────────


def _to_messages(prompts: Any) -> list[dict]:
    """
    Normalise whatever _execute_trajectory passes as `prompts` into
    a list[dict] conversation (OpenAI message format).

    Handles:
      - str                           → [{"role": "user", "content": str}]
      - dict (single message)         → [dict]
      - list[dict] (conversation)     → as-is
      - list[str]                     → joined into one user message
    """
    if isinstance(prompts, str):
        return [{"role": "user", "content": prompts}]
    if isinstance(prompts, dict):
        return [prompts]
    if isinstance(prompts, list):
        if not prompts:
            return []
        if isinstance(prompts[0], dict) and "role" in prompts[0]:
            return prompts  # already a conversation
        # list[str] → single user message
        return [{"role": "user", "content": "\n".join(str(p) for p in prompts)}]
    return [{"role": "user", "content": str(prompts)}]


# ─────────────────────────────────────────────────────────────────────────────
# Tool call extraction
# ─────────────────────────────────────────────────────────────────────────────


def _extract_tool_call_structs(response_message) -> list[dict] | None:
    tcs = getattr(response_message, "tool_calls", None)
    if not tcs:
        return None

    out = []
    for tc in tcs:
        fn = tc.function
        args = fn.arguments

        # ALWAYS serialise to a JSON string.
        # Some providers (Nvidia, Mistral) require arguments as a string,
        # and conversation history is replayed verbatim — a dict causes 400s.
        if isinstance(args, dict):
            args = json.dumps(args)
        elif not isinstance(args, str):
            args = json.dumps(args)

        out.append(
            {
                "type": "function",
                "id": getattr(tc, "id", f"call_{fn.name}"),
                "function": {
                    "name": fn.name,
                    "arguments": args,  # ← always a string now
                },
            }
        )

    return out or None


# ─────────────────────────────────────────────────────────────────────────────
# Retry helper
# ─────────────────────────────────────────────────────────────────────────────


def _with_retry(fn, max_retries: int = 3, base_delay: float = 1.0):
    """
    Simple exponential-backoff retry for rate-limit / transient errors.
    Retries on: RateLimitError, APIConnectionError, Timeout, 500/502/503/529.
    """
    import litellm

    retryable = (
        litellm.exceptions.RateLimitError,
        litellm.exceptions.APIConnectionError,
        litellm.exceptions.Timeout,
        litellm.exceptions.ServiceUnavailableError,
    )

    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except retryable as e:
            last_exc = e
            delay = base_delay * (2**attempt)
            warnings.warn(
                f"[APIRolloutEngine] {type(e).__name__}: {e}. "
                f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})",
                UserWarning,
            )
            time.sleep(delay)
        except Exception:
            raise  # non-retryable — surface immediately

    raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────


class APIRolloutEngine(RolloutEngine):
    """
    Universal API rollout engine backed by LiteLLM.

    Parameters
    ----------
    model : str
        Any LiteLLM model string. Examples:
            "gpt-4o-mini"
            "claude-haiku-4-5-20251001"
            "groq/llama-3.3-70b-versatile"
            "groq/llama-3.1-8b-instant"
            "openrouter/meta-llama/llama-3.1-70b-instruct"
            "openrouter/deepseek/deepseek-r1"
            "together_ai/mistralai/Mixtral-8x7B-v0.1"
            "ollama/llama3"
            "azure/my-deployment"
    api_key : str, optional
        API key. If omitted, looked up from the standard env-var for the
        provider (OPENAI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, …).
    base_url : str, optional
        Override the API base URL. Useful for:
          - Local Ollama:          "http://localhost:11434"
          - LiteLLM proxy server:  "http://localhost:4000"
          - Custom OpenAI-compat:  "https://my-proxy.example.com/v1"
    max_retries : int
        Number of retry attempts on rate-limit / connection errors. Default 3.
    retry_base_delay : float
        Base delay (seconds) for exponential backoff. Default 1.0.
    provider : str, optional
        Kept for backwards compatibility — ignored. Model string is sufficient.
    extra_litellm_kwargs : dict, optional
        Any extra kwargs forwarded verbatim to litellm.completion().
        Examples: {"top_p": 0.95, "stop": ["<|eot_id|>"]}
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        # backwards-compat — ignored
        provider: str | None = None,
        extra_litellm_kwargs: dict | None = None,
        **kwargs,
    ):
        try:
            import litellm  # noqa — validate installed
        except ImportError:
            raise ImportError(  # noqa: B904
                "litellm is required for APIRolloutEngine.\n" "Install with:  pip install litellm"
            )

        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.extra_litellm_kwargs = extra_litellm_kwargs or {}

        # Resolve and cache the API key once at init time
        self._api_key = _resolve_api_key(model, api_key)

        # Provider prefix — used to apply provider-specific patches
        self._prefix = _provider_prefix(model)

        # No local tokenizer — _execute_trajectory falls back to prompt-text mode
        self._tokenizer = None

    # ── RolloutEngine protocol ────────────────────────────────────────────────

    def _get_tokenizer(self):
        return self._tokenizer

    # ── Core generate ─────────────────────────────────────────────────────────

    def generate(
        self,
        prompts: Any,
        tools: list[dict] | None = None,
        gen_cfg: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a single model turn. Called by _execute_trajectory.

        Parameters
        ----------
        prompts : str | dict | list[dict] | list[str]
            The conversation so far (or a plain string prompt).
        tools : list[dict], optional
            OpenAI-format tool schemas. LiteLLM translates for each provider.
        gen_cfg : dict, optional
            Generation config. Recognised keys:
                max_new_tokens / max_tokens  (int,  default 1024)
                temperature                  (float, default 0.7)
                top_p                        (float, optional)
                stop                         (list[str], optional)

        Returns
        -------
        {
            "completions":            list[str],
            "completions_structured": list[dict],
            "logprobs":               list[None],
            "metadata":               dict,
        }
        """
        import litellm

        gen_cfg = gen_cfg or {}
        messages = _to_messages(prompts)

        # ── Build litellm.completion kwargs ───────────────────────────────────
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": gen_cfg.get("max_new_tokens", gen_cfg.get("max_tokens", 1024)),
            "temperature": gen_cfg.get("temperature", 0.7),
        }

        # Optional generation params
        if "top_p" in gen_cfg:
            call_kwargs["top_p"] = gen_cfg["top_p"]
        if "stop" in gen_cfg:
            call_kwargs["stop"] = gen_cfg["stop"]

        # API key + base URL
        if self._api_key:
            call_kwargs["api_key"] = self._api_key
        if self.base_url:
            call_kwargs["base_url"] = self.base_url

        # ── Tool schemas ──────────────────────────────────────────────────────
        clean_tools = [t for t in (tools or []) if isinstance(t, dict)]
        has_tools = bool(clean_tools) and self._prefix not in _NO_TOOLS_PROVIDERS

        if has_tools:
            call_kwargs["tools"] = clean_tools
            # Some providers support tools but reject the tool_choice param
            if self._prefix not in _NO_TOOL_CHOICE_PROVIDERS:
                call_kwargs["tool_choice"] = "auto"

        # ── User overrides (highest priority) ─────────────────────────────────
        call_kwargs.update(self.extra_litellm_kwargs)

        # ── Call with retry ───────────────────────────────────────────────────
        response = _with_retry(
            lambda: litellm.completion(**call_kwargs),
            max_retries=self.max_retries,
            base_delay=self.retry_base_delay,
        )

        msg = response.choices[0].message
        raw_text = msg.content or ""

        # ── Build structured assistant message ────────────────────────────────
        structured: dict[str, Any] = {"role": "assistant", "content": raw_text}
        tool_calls = _extract_tool_call_structs(msg)
        if tool_calls:
            structured["tool_calls"] = tool_calls

        # ── Usage stats (for logging / debugging) ─────────────────────────────
        usage = getattr(response, "usage", None)
        meta: dict[str, Any] = {
            "backend": "api",
            "provider": self._prefix,
            "model": self.model,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
        }

        return {
            "completions": [raw_text],
            "completions_structured": [structured],
            "logprobs": [None],
            "metadata": meta,
        }

    # ── Convenience ───────────────────────────────────────────────────────────

    def list_models(self) -> list[str]:
        """
        List available models for this provider.
        Only works for providers that expose a model list endpoint
        (OpenAI, OpenRouter). Returns empty list for others.
        """
        import litellm

        try:
            models = litellm.utils.get_valid_models()
            prefix = self._prefix
            return [m for m in models if m.startswith(prefix)]
        except Exception:
            return []

    def __repr__(self) -> str:
        return f"APIRolloutEngine(model={self.model!r}, " f"base_url={self.base_url!r})"
