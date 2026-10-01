"""
agenttune.agentic.scenarios.generator
======================================
Backend-agnostic scenario generator.  Synchronous -- no async needed.

Accepted tool forms
-------------------
  - Plain Python callables  (functions, lambdas, callable objects)
  - agenttune BaseTool instances
  - Raw dicts  {"name": ..., "description": ..., "parameters": ...}
  - MCP MCPTool / MCPResource objects  (optional dep, no hard import)
  - Any duck-typed object with .name / .description

Backends (first match wins)
---------------------------
  1. rollout_engine kwarg         -> vLLM / Transformers via existing engine
  2. rollout_backend / model_path -> builds engine via create_rollout_engine
  3. generator_model="claude-*"   -> Anthropic SDK
  4. default                      -> OpenAI-compatible (OpenAI / OpenRouter / custom)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .collection import ScenarioCollection
from .normalizer import normalize_resources, normalize_tools

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config dataclass  (optional -- callers can also use kwargs directly)
# -----------------------------------------------------------------------------


@dataclass
class ScenarioGeneratorConfig:
    """
    All generation knobs in one place.

    Pass to generate_scenarios(config=cfg), or just use keyword arguments.
    Individual kwargs always override config fields when both are supplied.
    """

    # What to generate
    num_scenarios: int = 24
    show_preview: bool = True
    custom_instructions: str | None = None
    domain: str | None = None  # e.g. "customer support", "DevOps"
    require_summary: bool = True  # append "summarise results" to each task

    # Remote API
    generator_model: str = "openai/gpt-4.1-mini"
    generator_api_key: str | None = None
    generator_base_url: str = "https://openrouter.ai/api/v1"

    # Local model (vLLM / Transformers)
    rollout_engine: Any | None = None  # pre-built RolloutEngine instance
    rollout_backend: str | None = None  # "vllm" | "transformers" | "auto"
    model: Any | None = None  # HuggingFace model object
    tokenizer: Any | None = None  # HuggingFace tokenizer
    model_path: str | None = None  # model name or local path
    custom_prompt: str | None = None

    # Generation kwargs forwarded to local engines
    local_gen_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "max_new_tokens": 2048,
            "temperature": 0.3,
            "do_sample": True,
        }
    )


# -----------------------------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------------------------


def _build_prompt(
    tools_info: list[dict],
    resources_info: list[dict],
    num_scenarios: int,
    custom_instructions: str | None,
    domain: str | None,
    require_summary: bool,
) -> str:
    tools_description = json.dumps(tools_info, indent=2)
    resources_description = (
        json.dumps(resources_info, indent=2) if resources_info else "No resources available"
    )

    domain_line = f"\nDomain context: {domain}" if domain else ""
    summary_req = (
        "\n6. Each task should conclude by producing a summary and analysis of results."
        if require_summary
        else ""
    )

    prompt = (
        f"You are an expert at creating realistic test scenarios for AI agents "
        f"that use tools to accomplish tasks.{domain_line}\n\n"
        f"Given the following available tools"
        f"{' and resources' if resources_info else ''}, "
        f"generate {num_scenarios} diverse, realistic scenarios that a user might "
        f"want to accomplish using these tools.\n\n"
        f"AVAILABLE TOOLS:\n{tools_description}\n\n"
        f"AVAILABLE RESOURCES:\n{resources_description}\n\n"
        f"Requirements:\n"
        f"1. Each scenario should be a concrete task accomplishable with the available tools.\n"
        f"2. Vary complexity -- some simple (1-2 tool calls), some complex (multi-step).\n"
        f"3. Cover different use-cases and tool combinations. "
        f"Do NOT name specific tools in the task description.\n"
        f"4. Keep tasks realistic -- something a real user would actually request.\n"
        f"5. Assign difficulty: 1 (trivial, single call) to 5 (hard, multi-step reasoning)."
        f"{summary_req}\n\n"
        f'You MUST respond with a JSON object containing a "scenarios" array of exactly '
        f"{num_scenarios} objects. Each object must have:\n"
        f'  - "task": string -- the scenario description\n'
        f'  - "difficulty": integer 1-5\n\n'
        f"Return ONLY the raw JSON object. No markdown fences, no explanation."
    )

    if custom_instructions:
        prompt += f"\n\nAdditional instructions:\n{custom_instructions}"

    return prompt


def _response_schema(num_scenarios: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "scenarios": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["task", "difficulty"],
                    "additionalProperties": False,
                },
                "minItems": num_scenarios,
                "maxItems": num_scenarios,
            }
        },
        "required": ["scenarios"],
        "additionalProperties": False,
    }


# -----------------------------------------------------------------------------
# Response parser
# -----------------------------------------------------------------------------


def _parse_content(content: str, num_scenarios: int) -> list[dict]:
    """Robustly parse JSON from any backend, including small/local models
    that truncate output or wrap it in markdown."""
    import re

    cleaned = content.strip()

    # ── Strip markdown fences ─────────────────────────────────────────────────
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(ln for ln in lines[1:] if ln.strip() not in ("```", "```json")).strip()

    def _extract_list(result) -> list[dict]:
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            if "scenarios" in result:
                return result["scenarios"]
            for v in result.values():
                if isinstance(v, list):
                    return v
        return []

    # ── Attempt 1: clean parse ────────────────────────────────────────────────
    try:
        scenarios = _extract_list(json.loads(cleaned))
        if scenarios:
            if len(scenarios) != num_scenarios:
                _dim(
                    f"[warn] Expected {num_scenarios}, got {len(scenarios)} — using what was parsed"
                )
            return scenarios
    except json.JSONDecodeError:
        pass

    # ── Attempt 2: find first {...} block via regex ───────────────────────────
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            scenarios = _extract_list(json.loads(match.group()))
            if scenarios:
                _dim(f"[warn] Recovered {len(scenarios)} scenarios via regex block extraction")
                return scenarios
        except json.JSONDecodeError:
            pass

    # ── Attempt 3: truncated JSON — extract all complete task objects ─────────
    partial = re.findall(
        r'\{\s*"task"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"difficulty"\s*:\s*([1-5])\s*\}',
        cleaned,
    )
    if partial:
        scenarios = [{"task": t, "difficulty": int(d)} for t, d in partial]
        _dim(
            f"[warn] Truncated output — recovered {len(scenarios)}/{num_scenarios} via partial parse"
        )
        return scenarios

    # ── Attempt 4: reverse order (difficulty before task) ────────────────────
    partial_rev = re.findall(
        r'\{\s*"difficulty"\s*:\s*([1-5])\s*,\s*"task"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        cleaned,
    )
    if partial_rev:
        scenarios = [{"task": t, "difficulty": int(d)} for d, t in partial_rev]
        _dim(f"[warn] Recovered {len(scenarios)}/{num_scenarios} via reverse-order partial parse")
        return scenarios

    # ── Attempt 5: last resort — extract any quoted task strings ─────────────
    tasks_only = re.findall(r'"task"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned)
    if tasks_only:
        _dim(
            f"[warn] Could only extract task strings ({len(tasks_only)}) — difficulty defaulting to 3"
        )
        return [{"task": t, "difficulty": 3} for t in tasks_only]

    raise ValueError(
        f"Could not parse any scenarios from model output.\n"
        f"First 500 chars of response: {cleaned[:500]}"
    )


# -----------------------------------------------------------------------------
# Backend implementations
# -----------------------------------------------------------------------------


def _via_openai_compat(
    prompt: str,
    num_scenarios: int,
    model: str,
    api_key: str,
    base_url: str,
) -> str:
    """OpenAI / OpenRouter / any OpenAI-compatible endpoint."""
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=8000,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "scenario_list",
                "schema": _response_schema(num_scenarios),
                "strict": True,
            },
        },
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI-compat API returned empty content")
    return content


def _via_anthropic(
    prompt: str,
    num_scenarios: int,
    model: str,
    api_key: str,
) -> str:
    """Anthropic Claude via the native SDK."""
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("pip install anthropic  (required for claude-* models)") from exc

    client = anthropic.Anthropic(api_key=api_key)
    # Prefill "{" forces the model to return pure JSON immediately
    response = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "{"},
        ],
    )
    return "{" + response.content[0].text


def _via_rollout_engine(
    prompt: str,
    engine: Any,
    gen_kwargs: dict,
) -> str:
    """
    Single-turn generation through any RolloutEngine (vLLM / Transformers / API).
    Scenario generation is always single-turn -- no tools, no loop needed.
    """
    conversation = [{"role": "user", "content": prompt}]
    result = engine.generate(
        prompts=conversation,
        tools=[],
        gen_cfg=gen_kwargs,
    )
    content = result["completions"][0]
    if not content:
        raise ValueError("RolloutEngine returned empty completion")
    return content


# -----------------------------------------------------------------------------
# Logging helpers (graceful fallback when agenttune logger is unavailable)
# -----------------------------------------------------------------------------


def _make_loggers():
    try:
        from agenttune.utils.logging import dim, err, info, ok, step

        return info, ok, err, step, dim
    except ImportError:

        def _p(msg):
            logger.info(f"  {msg}")

        return _p, _p, _p, _p, _p


_info, _ok, _err, _step, _dim = _make_loggers()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def generate_scenarios(
    tools: list[Any],
    resources: list[Any] | None = None,
    *,
    config: ScenarioGeneratorConfig | None = None,
    # Shortcuts -- each overrides the matching config field when provided
    num_scenarios: int | None = None,
    show_preview: bool | None = None,
    custom_instructions: str | None = None,
    domain: str | None = None,
    require_summary: bool | None = None,
    generator_model: str | None = None,
    generator_api_key: str | None = None,
    generator_base_url: str | None = None,
    rollout_engine: Any | None = None,
    rollout_backend: str | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_path: str | None = None,
    local_gen_kwargs: dict | None = None,
    custom_prompt: str | None = None,
) -> ScenarioCollection:
    """
    Generate evaluation scenarios for any tool-using agent.  Synchronous.

    Parameters
    ----------
    tools : list
        Tools in *any* format -- callables, dicts, BaseTool, MCPTool, ToolInfo.
        Mixed lists are fine.
    resources : list, optional
        Resources in any format.
    config : ScenarioGeneratorConfig, optional
        Config object.  Individual kwargs always override it.
    num_scenarios : int
        Number of scenarios to generate (default 24).
    show_preview : bool
        Print first 5 scenarios after generation (default True).
    custom_instructions : str, optional
        Extra instructions appended to the generation prompt.
    domain : str, optional
        Domain hint, e.g. "DevOps", "e-commerce", "medical records".
    require_summary : bool
        Append a "summarise results" step to every task (default True).
    generator_model : str
        Model identifier for API backends (default "openai/gpt-4.1-mini").
    generator_api_key : str, optional
        API key.  Falls back to env vars:
        ANTHROPIC_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY.
    generator_base_url : str
        Base URL for OpenAI-compat backends (default OpenRouter).
    rollout_engine : RolloutEngine, optional
        Pre-built engine -- reuse directly (zero rebuild cost).
    rollout_backend : str, optional
        "vllm" | "transformers" | "auto" -- builds engine on the fly.
    model : optional
        HuggingFace model object (transformers path).
    tokenizer : optional
        HuggingFace tokenizer (transformers path).
    model_path : str, optional
        Model name or local path (vLLM or transformers).
    local_gen_kwargs : dict, optional
        Generation kwargs forwarded to local RolloutEngine.

    Returns
    -------
    ScenarioCollection
        Iterable of Scenario objects.  Key helpers:
          .tasks()                      -> List[str]  (prompt-ready)
          .filter_by_difficulty(min, max)
          .to_json(path)
          .from_json(path)  (classmethod)

    Examples
    --------
    # Quickstart -- plain functions, OpenRouter default
    >>> scenarios = generate_scenarios([search, run_code], num_scenarios=10)

    # Claude backend
    >>> scenarios = generate_scenarios(
    ...     tools, generator_model="claude-haiku-4-5-20251001"
    ... )

    # Native OpenAI
    >>> scenarios = generate_scenarios(
    ...     tools,
    ...     generator_model="gpt-4o-mini",
    ...     generator_base_url="https://api.openai.com/v1",
    ... )

    # Reuse a vLLM engine from training (no rebuild)
    >>> scenarios = generate_scenarios(tools, rollout_engine=my_vllm_engine)

    # Build a new local Transformers engine on the fly
    >>> scenarios = generate_scenarios(
    ...     tools, rollout_backend="transformers", model_path="Qwen/Qwen2.5-7B-Instruct"
    ... )

    # Config object style
    >>> cfg = ScenarioGeneratorConfig(num_scenarios=16, domain="DevOps")
    >>> scenarios = generate_scenarios(tools, config=cfg)

    # Downstream use
    >>> prompts = scenarios.filter_by_difficulty(min_difficulty=3).tasks()
    """
    t0 = time.perf_counter()

    if not tools and not resources:
        raise ValueError("Provide at least one tool or resource")

    # Merge config + kwargs (kwargs win)
    cfg = config or ScenarioGeneratorConfig()

    def _pick(kwarg_val, cfg_val):
        return kwarg_val if kwarg_val is not None else cfg_val

    n_scenarios = _pick(num_scenarios, cfg.num_scenarios)
    preview = _pick(show_preview, cfg.show_preview)
    instructions = _pick(custom_instructions, cfg.custom_instructions)
    dom = _pick(domain, cfg.domain)
    req_sum = _pick(require_summary, cfg.require_summary)
    gen_model = _pick(generator_model, cfg.generator_model)
    gen_key = _pick(generator_api_key, cfg.generator_api_key)
    gen_url = _pick(generator_base_url, cfg.generator_base_url)
    engine = _pick(rollout_engine, cfg.rollout_engine)
    rb = _pick(rollout_backend, cfg.rollout_backend)
    hf_model = _pick(model, cfg.model)
    hf_tok = _pick(tokenizer, cfg.tokenizer)
    m_path = _pick(model_path, cfg.model_path)
    gen_kwargs = _pick(local_gen_kwargs, cfg.local_gen_kwargs)

    # Normalise tools / resources to plain dicts
    tools_info = [t.to_dict() for t in normalize_tools(tools)]
    resources_info = [r.to_dict() for r in normalize_resources(resources)]

    _info(f"Tools: {len(tools_info)}  |  Resources: {len(resources_info)}")
    custom_prompt_val = _pick(custom_prompt, cfg.custom_prompt)  # add with other _pick() calls

    # Build prompt
    _step("Building generation prompt ...")
    # prompt = _build_prompt(
    #     tools_info, resources_info, n_scenarios,
    #     instructions, dom, req_sum,
    # )
    prompt = custom_prompt_val or _build_prompt(
        tools_info,
        resources_info,
        n_scenarios,
        instructions,
        dom,
        req_sum,
    )

    # Select backend and generate
    content: str
    use_local = engine is not None or rb or hf_model or m_path

    if use_local:
        # Build engine on the fly if not pre-supplied
        if engine is None:
            from agenttune.agentic.rollout_engines.rollout_factory import (
                create_rollout_engine,
            )

            engine = create_rollout_engine(
                backend=rb or "auto",
                model=hf_model,
                tokenizer=hf_tok,
                model_path=m_path,
            )
        label = type(engine).__name__
        _step(f"Backend: {label} (local)")
        t1 = time.perf_counter()
        content = _via_rollout_engine(prompt, engine, gen_kwargs)

    elif gen_model and gen_model.startswith("claude"):
        api_key = gen_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Claude models require an API key. "
                "Pass generator_api_key or set ANTHROPIC_API_KEY."
            )
        _step(f"Backend: Anthropic  ({gen_model})")
        t1 = time.perf_counter()
        content = _via_anthropic(prompt, n_scenarios, gen_model, api_key)

    else:
        # OpenAI-compatible (OpenAI / OpenRouter / custom)
        api_key = gen_key
        if api_key is None:
            for env in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
                api_key = os.getenv(env)
                if api_key:
                    break
        if not api_key:
            raise ValueError(
                "No API key found. "
                "Pass generator_api_key or set OPENROUTER_API_KEY / OPENAI_API_KEY."
            )

        # Auto-switch base_url for native OpenAI model names
        base_url = gen_url
        if gen_model:
            _native_prefixes = ("gpt-", "o1", "o3", "o4", "text-", "ft:")
            if any(gen_model.startswith(p) for p in _native_prefixes) and "openrouter" in base_url:
                base_url = "https://api.openai.com/v1"
                _info("Native OpenAI model detected -- switching base_url to api.openai.com")

        _step(f"Backend: OpenAI-compat  ({gen_model}  @  {base_url})")
        t1 = time.perf_counter()
        content = _via_openai_compat(prompt, n_scenarios, gen_model, api_key, base_url)

    _ok(f"Response received in {time.perf_counter() - t1:.2f}s  ({len(content)} chars)")

    # Parse
    try:
        raw_scenarios = _parse_content(content, n_scenarios)
    except Exception as exc:
        _err(f"JSON parse failed: {exc}")
        _dim(f"First 500 chars: {content[:500]}")
        raise

    _ok(f"Parsed {len(raw_scenarios)} scenarios")

    # Build and return collection
    collection = ScenarioCollection.from_dicts(raw_scenarios)
    collection.print_difficulty_distribution()
    if preview:
        collection.preview(n=min(5, n_scenarios))

    _ok(f"Done -- {len(collection)} scenarios in {time.perf_counter() - t0:.2f}s total")
    return collection
