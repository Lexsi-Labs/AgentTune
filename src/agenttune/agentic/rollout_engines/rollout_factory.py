import asyncio
import inspect
import json
import re
import uuid
import warnings
from collections.abc import Callable
from typing import Any

import torch

from ..trajectory.dataset import Step, Trajectory
from .base import RolloutEngine
from .tool_call_parse import (
    _parse_cohere_action,
    _parse_cohere_json,
    _parse_cohere_legacy_action,
    _parse_fenced_after_name,
    _parse_harmony_tool_calls,
    _parse_react,
    extract_tool_calls_from_text,
)

# Once a Harmony-format turn ISN'T a tool call (e.g. the model's actual final
# answer, on the "final" channel), the same "<|start|>...<|channel|>...
# <|message|>...<|end|>/<|call|>/<|return|>" wrapper is still present in the
# raw completion text (kept raw so _parse_harmony_tool_calls above can match
# -- see the "completions_raw" preference in _gen). Left unstripped, it leaks
# into Trajectory.final_response / Step.observation as literal template
# control tokens instead of the plain answer text. Stripped the same way
# <think> blocks already are elsewhere in this module.
#
# The leading "<|start|>ROLE" is OPTIONAL in the pattern below: it's part of
# the rendered PROMPT (the "assistant" turn opener added by
# add_generation_prompt=True), so whether it shows up in the *completion*
# text depends on the engine -- verified the transformers backend's raw
# completion includes it ("<|start|>assistant<|channel|>...") while the vLLM
# backend's doesn't (generation starts right at "<|channel|>..."). Requiring
# it unconditionally left vLLM completions completely unstripped.
_HARMONY_WRAPPER_RE = re.compile(
    r"(?:<\|start\|>\S*)?<\|channel\|>\S+(?:\s+to=\S+)?(?:<\|constrain\|>\S+)?<\|message\|>"
    r"|<\|end\|>|<\|call\|>|<\|return\|>"
)


def _strip_harmony_markup(text):
    if not isinstance(text, str) or "<|channel|>" not in text:
        return text
    return _HARMONY_WRAPPER_RE.sub("", text).strip()


# Lazy import GRPOTrainer
# from trl import GRPOTrainer
# generate_rollout_completions is imported lazily inside _gen's vLLM branch
# (trl.experimental.openenv is an unstable API and only needed when a trainer
# actually runs in use_vllm mode — importing it at module top level would fail
# import-time on trl builds that lack trl.experimental). See _vllm_generate().
# ─────────────────────────────────────────────────────────────────────────────
# Backend detection
# ─────────────────────────────────────────────────────────────────────────────


def _detect_best_backend() -> str:
    try:
        import vllm  # noqa

        return "vllm"
    except ImportError:
        return "transformers"


def _is_vllm_engine(engine) -> bool:
    try:
        from .vllm_engine import VLLMRolloutEngine

        return isinstance(engine, VLLMRolloutEngine)
    except ImportError:
        return False


def _build_conversation(
    prompt: Any,
    system_prompt: str | None = None,
) -> list[dict]:
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        if system_prompt and prompt[0].get("role") != "system":
            return [{"role": "system", "content": system_prompt}] + prompt
        return prompt
    if isinstance(prompt, dict) and prompt.get("role") is None:
        prompt = prompt.get("prompt") or prompt.get("content") or str(prompt)

    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": prompt})
    return conversation


# How a tool result reads once folded into a user turn. Says what the turn is
# and what to do next: without that, Tiny Aya answers "Thank you!".
DEFAULT_TOOL_RESULT_FORMAT = (
    "Tool result for {name}: {content}\nUse this result to answer the original question."
)


def fold_tool_messages_into_user(
    conversation: list[dict], tool_result_format: str = DEFAULT_TOOL_RESULT_FORMAT
) -> list[dict]:
    """Rewrite any "tool"-role message as a plain "user" turn instead, as
    ``tool_result_format`` (``{name}``, ``{content}``).

    Some chat templates (e.g. Gemma's) only know "user"/"model" and hard-
    fail ("Conversation roles must alternate...") on any other role,
    including "tool" -- they were never designed with a tool-result
    turn at all. Folding the tool result into a "user" turn keeps strict
    two-role alternation intact (consecutive tool messages, from parallel
    tool calls, are merged into ONE user turn rather than two -- two
    "user" turns in a row would break alternation just as badly).

    Engines call this as a fallback, retrying the render after the
    template rejects the unmodified conversation, so models that DO
    support a "tool" role are unaffected.
    """
    folded: list[dict] = []
    for msg in conversation:
        if msg.get("role") == "tool":
            content = tool_result_format.format(
                name=msg.get("name") or "the tool", content=msg.get("content", "")
            )
            if folded and folded[-1]["role"] == "user":
                folded[-1] = {**folded[-1], "content": f"{folded[-1]['content']}\n\n{content}"}
            else:
                folded.append({"role": "user", "content": content})
        elif msg.get("role") == "user" and folded and folded[-1]["role"] == "user":
            folded[-1] = {
                **folded[-1],
                "content": f"{folded[-1]['content']}\n\n{msg.get('content', '')}",
            }
        else:
            folded.append(msg)
    return folded


def _is_folded_tool_result(content: str, tool_result_format: str) -> bool:
    """True if a user turn's ``content`` was written by ``fold_tool_messages_into_user``
    with ``tool_result_format``."""
    pattern = re.escape(tool_result_format)
    for field in ("name", "content"):
        pattern = pattern.replace(re.escape("{" + field + "}"), ".*?")
    return re.match(pattern, str(content), flags=re.DOTALL) is not None


def coerce_tool_call_arguments_to_dict(conversation: list[dict]) -> list[dict]:
    """Parse any assistant tool_calls[].function.arguments that are JSON
    strings back into plain dicts.

    Every wire format _extract_tool_calls parses is normalised to a JSON
    *string* for "arguments" (matching most vendors' own chat templates,
    e.g. DeepSeek's). But re-encoding that assistant turn back into the
    conversation for the next generation call, some templates (e.g.
    Qwen3.5's -- confirmed by reading its real template, which iterates
    `arguments.items()` to render each "<parameter=K>V</parameter>" tag)
    require a dict there instead, and crash with a plain TypeError
    ("Can only get item pairs from a mapping") on a string. Engines call
    this as a fallback retry, the same way fold_tool_messages_into_user
    is, so templates that already want a string are unaffected.
    """
    coerced: list[dict] = []
    for msg in conversation:
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            coerced.append(msg)
            continue
        new_calls = []
        for tc in tool_calls:
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
            args = fn.get("arguments") if fn else None
            if isinstance(args, str):
                try:
                    fn = {**fn, "arguments": json.loads(args)}
                    tc = {**tc, "function": fn}
                except json.JSONDecodeError:
                    pass
            new_calls.append(tc)
        coerced.append({**msg, "tool_calls": new_calls})
    return coerced


# Templates probed by _template_drops_tool_results -> whether they drop tool turns.
_TOOL_TURN_DROPPED: dict[str, bool] = {}


def _template_drops_tool_results(tokenizer, chat_template: str | None = None) -> bool:
    """True if the chat template renders a "tool" turn without its content
    (Tiny Aya and North render it as an empty turn, with no error). Probed
    once per template; warns the first time."""
    template = chat_template or getattr(tokenizer, "chat_template", None)
    key = str(template)
    if key not in _TOOL_TURN_DROPPED:
        sentinel = "agenttune-tool-result-probe"
        call = {"type": "function", "function": {"name": "f", "arguments": {}}}
        probe = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "name": "f", "content": sentinel},
        ]
        try:
            rendered = tokenizer.apply_chat_template(probe, chat_template=template, tokenize=False)
            dropped = sentinel not in str(rendered)
        except Exception:
            dropped = False  # it raises on a tool turn; the render retries fold it
        _TOOL_TURN_DROPPED[key] = dropped
        if dropped:
            name = getattr(tokenizer, "name_or_path", "this model")
            warnings.warn(
                f"The chat template of {name} renders tool results as an empty turn; "
                "folding them into user turns instead.",
                UserWarning,
                stacklevel=3,
            )
    return _TOOL_TURN_DROPPED[key]


def _keep_tool_results(
    tokenizer,
    conversation: list[dict],
    chat_template=None,
    tool_result_format: str = DEFAULT_TOOL_RESULT_FORMAT,
) -> list[dict]:
    """``conversation`` with its tool turns folded into user turns if the
    template would otherwise drop them; unchanged otherwise."""
    if any(m.get("role") == "tool" for m in conversation) and _template_drops_tool_results(
        tokenizer, chat_template
    ):
        return fold_tool_messages_into_user(conversation, tool_result_format)
    return conversation


def _render_chat_template(
    tokenizer,
    conversation: list[dict],
    schemas: list[dict] | None = None,
    enable_thinking: bool = False,
    tool_result_format: str = DEFAULT_TOOL_RESULT_FORMAT,
    **kwargs,
) -> Any:
    """``tokenizer.apply_chat_template(..., add_generation_prompt=True)`` with
    the same repair chain the standalone engine uses, for every trainer-path
    render.

    Arguments are tried as dicts first: templates such as Command R7B's
    apply ``tojson`` to them unconditionally, so a JSON string is
    silently double-encoded rather than rejected. Templates that need a string
    raise on a dict and get the string on the next attempt. Tool results are
    folded into a user turn when the template rejects a "tool" role (Aya
    Expanse, Gemma 3) or renders it empty (Tiny Aya, North). A missing
    ``tool_calls[].function.arguments`` is defaulted to ``"{}"`` first
    (Llama-3.2's template ``tojson``s it unguarded).
    """
    kwargs.setdefault("add_generation_prompt", True)
    conversation = _ensure_tool_call_arguments_present(
        _keep_tool_results(tokenizer, conversation, tool_result_format=tool_result_format)
    )
    folded = fold_tool_messages_into_user(conversation, tool_result_format)
    variants = [
        coerce_tool_call_arguments_to_dict(conversation),
        conversation,
        coerce_tool_call_arguments_to_dict(folded),
        folded,
    ]
    option_sets = [
        {"tools": schemas or None, "enable_thinking": enable_thinking},
        {"tools": schemas or None},
        {"enable_thinking": enable_thinking},
        {},
    ]
    last_err: Exception | None = None
    for opts in option_sets:  # keep tools as long as any repair renders them
        call_kw = {**kwargs, **opts}
        for conv in variants:
            try:
                return tokenizer.apply_chat_template(conv, **call_kw)
            except Exception as e:
                last_err = e
    raise last_err


def _strip_special_tokens(text: str, tokenizer) -> str:
    """Remove the tokenizer's special-token strings (chat markers, EOS) from
    *text*. The engine's raw decode keeps them, and it also puts back the
    generation prompt (e.g. Cohere's ``<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
    <|START_RESPONSE|>``). The raw text is still what the tool-call parser
    reads; this is for the answer and the assistant turn sent back."""
    if not text or tokenizer is None:
        return text
    specials = set(getattr(tokenizer, "all_special_tokens", None) or [])
    specials |= {
        t.content
        for t in (getattr(tokenizer, "added_tokens_decoder", None) or {}).values()
        if getattr(t, "special", False)
    }
    for tok in sorted(filter(None, specials), key=len, reverse=True):
        text = text.replace(tok, "")
    return text.strip()


def _template_renders_tools(tokenizer, schemas: list[dict]) -> bool:
    """True if the chat template puts ``tools`` into the prompt: the render
    changes and names at least one tool. Some templates (Tiny Aya, Aya Expanse,
    Aya Vision, North, Gemma 3) silently ignore them."""
    probe = [{"role": "user", "content": "hi"}]
    try:
        with_tools = tokenizer.apply_chat_template(
            probe, tools=schemas, add_generation_prompt=True, tokenize=False
        )
        without = tokenizer.apply_chat_template(probe, add_generation_prompt=True, tokenize=False)
    except Exception:
        return False  # a template that errors on `tools` doesn't render them: use the fallback
    names = [(s.get("function") or s).get("name") for s in schemas if isinstance(s, dict)]
    return with_tools != without and (
        not any(names) or any(n and str(n) in str(with_tools) for n in names)
    )


# Default system-prompt text for templates that drop ``tools``. Asks for the
# Cohere/Command-R call format, which _extract_tool_calls parses.
DEFAULT_TOOLS_FALLBACK_PROMPT = (
    "You can call these tools:\n{tools}\n\n"
    "To call tools, reply with only a JSON list, one object per call, and no other text: "
    '[{{"tool_name": "<name>", "parameters": {{"<argument>": <value>}}}}]. '
    "Use a list even for one call. "
    'A user turn that starts with "Tool result" is the output of your call, not a new '
    "message from the user: use it to answer the user's original question as plain text."
)


def _ensure_tool_call_arguments_present(conversation: list[dict]) -> list[dict]:
    """Default a missing tool_calls[].function.arguments to "{}" rather than
    leaving the key absent entirely.

    Every parser in tool_call_parse.py does set "arguments" when it
    successfully extracts a call, but a call the parser only partially
    recovers (e.g. malformed/double-escaped JSON from a model that garbles
    its own tool-call syntax on a retry turn -- observed with Llama-3.2)
    can still end up missing the key. Some templates (Llama-3.2's: `{{-
    tool_call.arguments | tojson }}`) access it unconditionally with no
    `is defined` guard, and Jinja's attribute access on a missing key
    returns its `Undefined` sentinel rather than raising -- which `| tojson`
    then can't serialize, crashing with a TypeError that reads nothing like
    the actual missing-key cause. coerce_tool_call_arguments_to_dict (above)
    only fixes the *type* of an existing value; this fixes *presence*.
    """
    fixed: list[dict] = []
    for msg in conversation:
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            fixed.append(msg)
            continue
        new_calls = []
        for tc in tool_calls:
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
            if fn is not None and "arguments" not in fn:
                tc = {**tc, "function": {**fn, "arguments": "{}"}}
            new_calls.append(tc)
        fixed.append({**msg, "tool_calls": new_calls})
    return fixed


# AgentTune was originally built assuming every checkpoint is a plain
# causal LM, but newer tool-calling-capable models increasingly ship as
# vision+text wrapper classes even when used text-only -- e.g.
# Ministral-3's Mistral3ForConditionalGeneration, Gemma-3 12B+'s
# Gemma3ForConditionalGeneration. Loading one of those via
# AutoModelForCausalLM raises ValueError("Unrecognized configuration
# class ... for this kind of AutoModel"). Try the auto classes real
# tool-calling checkpoints actually use, in order, instead of hardcoding
# one and forcing callers to pre-load the model themselves to work around
# it.
_MODEL_AUTO_CLASS_NAMES = (
    "AutoModelForCausalLM",
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
    "AutoModelForSeq2SeqLM",
)


def _avoid_parallel_weight_loading_race() -> None:
    """Force transformers' checkpoint-shard loader to a single worker thread.

    transformers.core_model_loading loads/dequantizes weight shards with a
    thread pool (`GLOBAL_WORKERS = min(4, os.cpu_count())`). For on-the-fly
    dequantized MoE checkpoints (gpt-oss's MXFP4 -> bf16 fallback on GPUs
    without the native kernels), this races: concurrent threads writing
    tensors into the same MoE module occasionally leave one tensor un-moved
    to the target CUDA device, so the very next forward pass crashes with
    "Expected all tensors to be on the same device ... mat2 is on cpu" --
    intermittently (verified: ~50% failure rate across repeated loads of
    openai/gpt-oss-20b on the same GPU/checkpoint/device_map, 0/4 failures
    after this patch). Single-threaded loading is slightly slower, so this
    is only applied for gpt-oss checkpoints (see the model_type check at the
    call site) rather than globally for every model. Best-effort: silently
    no-ops on transformers versions that don't have this internal (older
    versions load single-threaded already; a future refactor may rename/
    remove it).
    """
    try:
        import transformers.core_model_loading as _cml

        _cml.GLOBAL_WORKERS = 1
    except (ImportError, AttributeError):
        pass


def load_causal_or_multimodal_model(model_path: str, **kwargs):
    """Try each auto class in _MODEL_AUTO_CLASS_NAMES in turn, returning
    the first that successfully loads `model_path`. Raises the last
    "architecture not recognised" ValueError if none of them match the
    checkpoint's architecture.

    Only that specific error is treated as "wrong class, try the next one" --
    transformers raises it with a fixed, distinctive message ("Unrecognized
    configuration class ... for this kind of AutoModel: ...", see
    auto_factory.py) whenever a config isn't in that class's model mapping.
    Any OTHER ValueError (e.g. an incompatible quantization_config) comes
    from a class whose architecture DID match, so it's a real failure and
    must be re-raised immediately -- swallowing it here would silently retry
    unrelated auto classes and surface THEIR "unrecognized configuration"
    error instead, masking the actual problem.
    """
    import transformers

    # Scoped to gpt-oss specifically (by actual config.model_type, not a
    # guess from the path string, so local paths/forks are still caught) --
    # see _avoid_parallel_weight_loading_race's docstring for why. Every
    # other architecture keeps the default multi-threaded loader.
    try:
        model_type = transformers.AutoConfig.from_pretrained(
            model_path, trust_remote_code=kwargs.get("trust_remote_code", False)
        ).model_type
    except Exception:
        model_type = None
    if model_type == "gpt_oss":
        _avoid_parallel_weight_loading_race()

    last_err: ValueError | None = None
    for cls_name in _MODEL_AUTO_CLASS_NAMES:
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(model_path, **kwargs)
        except ValueError as e:
            if "Unrecognized configuration class" not in str(e):
                raise
            last_err = e
            continue
    raise last_err


# ─────────────────────────────────────────────────────────────────────────────
# Tool schema builders
# ─────────────────────────────────────────────────────────────────────────────


def _fallback_tool_schema(fn) -> dict:
    name = getattr(fn, "__name__", "tool")
    doc = inspect.getdoc(fn) or name
    brief = doc.strip().split("\n", 1)[0]
    props: dict[str, Any] = {}
    required: list[str] = []
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        for pname, p in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            props[pname] = {"type": "string", "description": pname}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": brief,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


def _build_tool_schema(fn, use_vllm: bool = False) -> dict:
    try:
        from transformers.utils import get_json_schema

        schema = get_json_schema(fn)
        if schema:
            return schema
    except (ImportError, Exception):
        pass
    return _fallback_tool_schema(fn)


def _get_callable(tool) -> Callable:
    """Unwrap BaseTool or plain function to a callable."""
    if hasattr(tool, "execute"):
        return tool.execute
    if callable(tool):
        return tool
    raise TypeError(f"Tool '{tool}' is neither a BaseTool nor callable.")


# ─────────────────────────────────────────────────────────────────────────────
# Async execution helpers
# ─────────────────────────────────────────────────────────────────────────────


def _run_sync_or_async(fn: Callable, kwargs: dict) -> Any:
    if not asyncio.iscoroutinefunction(fn):
        return fn(**kwargs)

    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, fn(**kwargs))
            return future.result()
    except RuntimeError:
        return asyncio.run(fn(**kwargs))


def _stringify_tool_result(result: Any) -> str:
    """Extract the human-readable string from a tool result.

    BaseTool.execute() returns a ToolResult dataclass; str() on it gives the
    dataclass repr (``ToolResult(success=True, output='...', ...)``) which the
    model then sees as the tool output — confusing and token-wasteful. If the
    result is a ToolResult, return its ``output`` (stringified); otherwise
    stringify the result directly (plain-function tools that return str).
    """
    # ToolResult is a dataclass with an ``output`` field.
    if hasattr(result, "output") and hasattr(result, "success"):
        out = result.output
        if out is None:
            # ToolResult with no output (error case) — surface the error.
            err = getattr(result, "error", None)
            return f"Error: {err}" if err else "No output."
        return str(out)
    return str(result)


def _run_tools_parallel(calls: list[tuple]) -> list[tuple]:
    has_async = any(asyncio.iscoroutinefunction(fn) for _, fn, _ in calls)

    if not has_async:
        results = []
        for name, fn, kwargs in calls:
            try:
                try:
                    results.append((name, _stringify_tool_result(fn(**kwargs))))
                except TypeError:
                    if isinstance(kwargs, str):
                        kwargs = json.loads(kwargs)
                    results.append((name, _stringify_tool_result(fn(**kwargs))))
            except Exception as e:
                results.append((name, {"error": str(e)}))
        return results

    async def _call_one(name, fn, kwargs):
        try:
            try:
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(**kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: fn(**kwargs))
            except TypeError:
                if isinstance(kwargs, str):
                    kwargs = json.loads(kwargs)
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(**kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: fn(**kwargs))
            return name, _stringify_tool_result(result)
        except Exception as e:
            return name, {"error": str(e)}

    async def _gather():
        return await asyncio.gather(*[_call_one(n, f, k) for n, f, k in calls])

    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _gather())
            return future.result()
    except RuntimeError:
        return asyncio.run(_gather())


# ─────────────────────────────────────────────────────────────────────────────
# Logprob normalisation
#
# FIX #1: GRPOTrainer._generate_and_score_completions does:
#     sampling_per_token_logps = [torch.tensor(logps) for logps in logprobs]
# This requires logprobs to be a list[list[float]] — plain floats, NOT tuples.
# We keep _normalise_logprobs returning (float, int) tuples internally for
# token-id tracking. _ensure_logprob_tuples guarantees every element is a
# subscriptable (log_prob, token_id) tuple as GRPOTrainer expects.
# ─────────────────────────────────────────────────────────────────────────────


def _normalise_logprobs(
    logprobs: Any,
    token_ids: list[int] | None = None,
) -> list[tuple]:
    if not logprobs:
        return []

    first = logprobs[0]

    # vLLM returns list[list[float]] — take first logprob from each position
    if isinstance(first, list | tuple) and first and isinstance(first[0], int | float):
        ids = token_ids or []
        return [(float(lp[0]), int(ids[i]) if i < len(ids) else 0) for i, lp in enumerate(logprobs)]

    if (
        isinstance(first, tuple | list)
        and len(first) >= 2
        and not isinstance(first[0], list | tuple)
    ):
        return [tuple(lp) for lp in logprobs]

    if isinstance(first, dict):
        return [
            (lp.get("logprob", lp.get("log_prob", 0.0)), lp.get("token_id", 0)) for lp in logprobs
        ]

    ids = token_ids or []
    return [(float(lp), int(ids[i]) if i < len(ids) else 0) for i, lp in enumerate(logprobs)]


def _ensure_logprob_tuples(logprobs: list[Any]) -> list[tuple]:
    """
    GRPOTrainer._generate_single_turn does:
        logprobs = [[lp[0] for lp in seq] for seq in logprobs]
    so every element must be subscriptable — a tuple/list, NOT a plain float.

    This ensures each element is a (log_prob, token_id) tuple.
    Plain floats are wrapped as (float, 0) so lp[0] still works.
    Tool result tokens use 0.0 logprob so they contribute nothing to the loss.
    """
    result = []
    for lp in logprobs:
        if isinstance(lp, tuple | list) and len(lp) >= 1:
            result.append(tuple(lp))
        else:
            result.append((float(lp), 0))
    return result


def _ids_to_list(ids: Any) -> list[int]:
    if ids is None:
        return []
    # Handle tokenizer BatchEncoding / plain dict / any Mapping with 'input_ids'
    if hasattr(ids, "keys"):
        ids = ids.get("input_ids", [])
    # Handle tensors and numpy arrays
    if hasattr(ids, "tolist"):
        return ids.tolist()
    if isinstance(ids, list):
        if not ids:
            return []
        flat = []
        for x in ids:
            if hasattr(x, "tolist"):
                flat.extend(x.tolist())
            elif isinstance(x, list | tuple):
                flat.extend(int(i) for i in x)
            else:
                flat.append(int(x))
        return flat
    return [int(x) for x in ids]


# ─────────────────────────────────────────────────────────────────────────────
# Tool call parser
# ─────────────────────────────────────────────────────────────────────────────


def _load_first_json_object(text: str) -> Any:
    """json.loads(text), with a tolerant fallback for near-miss model output.

    Small fine-tuned agents often wrap a tool call in a mismatched/dangling
    XML tag and append stray characters after the closing brace (e.g. a
    trailing ``)``). Try a strict parse first, then recover the first balanced
    ``{...}`` object from the text so a correct call still executes.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def _extract_balanced_json_object(text: str, start: int) -> str | None:
    """From a "{" at or after ``start``, return the substring up to its
    matching "}" by counting brace depth (handles nested objects, unlike a
    non-greedy regex, which has no way to know where a *nested* "}" ends
    vs. the outer one without a following literal to anchor on)."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# Llama 3.1/3.2 built-in tool format: <function=NAME>{...json args...}</function>
_LLAMA_FUNC_RE = re.compile(r"<function=(\S+)>(\{.*?\})</function>", re.DOTALL)

# Qwen3.5's native format (verified against Qwen/Qwen3.5-4B's real
# generated output, using its own tokenizer chat template -- this is a
# completely different surface form from Qwen3's plain
# "<tool_call>{json}</tool_call>": each argument gets its own
# "<parameter=NAME>value</parameter>" tag inside a Llama-style
# "<function=NAME>...</function>" block, itself wrapped in the same
# "<tool_call>" tag Qwen3 uses. Superficially close to Llama's format
# (also "<function=NAME>...</function>") but the body here is never a
# bare JSON object, so it never matches _LLAMA_FUNC_RE.
_QWEN35_FUNCTION_RE = re.compile(r"<function=(\w+)>\s*(.*?)\s*</function>", re.DOTALL)
_QWEN35_PARAMETER_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL)

# Gemma 4's native format (verified against google/gemma-4-E2B-it's real
# tokenizer chat template output) -- a third, unrelated tool-call syntax
# from the same "Gemma" name: Gemma 3 has no tool-calling template at all;
# Gemma 4 has one, but it's a custom minified-object syntax, not JSON and
# not XML tags. Note the ASYMMETRIC delimiters -- opens "<|tool_call>",
# closes "<tool_call|>" (pipe on the other side) -- and string values are
# wrapped in a custom quote marker, "<|"|>...<|"|>", instead of `"`.
_GEMMA4_TOOL_CALL_RE = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.DOTALL)
_GEMMA4_ARG_RE = re.compile(r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]+))', re.DOTALL)

# Mistral v0.3+ format: a "[TOOL_CALLS]" marker followed by a JSON array of
# {"name": ..., "arguments": {...}} objects.
_MISTRAL_TOOL_CALLS_RE = re.compile(r"\[TOOL_CALLS\]\s*(\[.*\])", re.DOTALL)

# Newer Mistral "mistral_common" tokenizer backend format (e.g. the
# Ministral-3 line's AutoTokenizer auto-selects this backend over its own
# Jinja chat_template.jinja, which still describes the older
# "[TOOL_CALLS] [...]" format above). Verified against
# mistralai/Ministral-3-3B-Instruct-2512-BF16's real generated output --
# and it genuinely generates TWO different surface forms depending on
# decoding strategy, both reproducible, neither a fluke:
#   - sampling (do_sample=True): "[TOOL_CALLS]NAME[ARGS]{json}", using
#     "[TOOL_CALLS]" and "[ARGS]" as real special tokens (confirmed
#     against mistral_common's own tokenizer source,
#     SpecialTokens.tool_calls / .args and
#     _encode_tool_calls_in_assistant_message -- this is the format
#     mistral_common itself encodes when BUILDING a request, so it's the
#     authoritative one).
#   - greedy (do_sample=False): "[NAME]{json}", plain bracket characters,
#     no special tokens at all -- a different, non-canonical surface form
#     the model also produces.
# Both need support, and the TOOL_CALLS-anchored one MUST be tried first:
# a naive plain-bracket regex run against the sampling-mode text matches
# "[ARGS]" itself as if it were the function name (an earlier version of
# this parser did exactly that, guessed from a single sample that
# happened to be the greedy-only form).
_MISTRAL_BRACKET_NAME_RE = re.compile(r"\[TOOL_CALLS\](\w+)\[ARGS\]")
_MISTRAL_PLAIN_BRACKET_NAME_RE = re.compile(r"\[(?!ARGS\]|TOOL_CALLS\])(\w+)\]\s*(?=\{)")

# DeepSeek-V3/R1 chat template format (note: fullwidth "｜"/"▁" characters,
# not ASCII "|"/"_" — verified against deepseek-ai/DeepSeek-V3's own
# tokenizer_config.json chat template, not guessed):
#   <｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
#   ```json
#   {...}
#   ```<｜tool▁call▁end｜>
_DEEPSEEK_CALL_RE = re.compile(
    r"<｜tool▁call▁begin｜>(?:function)?<｜tool▁sep｜>(\S+)\s*```json\s*(\{.*?\})\s*```\s*<｜tool▁call▁end｜>",
    re.DOTALL,
)

# GLM-4.5/4.7 format — reuses the same <tool_call> tag as Qwen but with a
# non-JSON body: the function name, then repeated <arg_key>/<arg_value>
# pairs instead of a JSON object. Verified against zai-org/GLM-4.5-Air's
# real tokenizer chat template output, which puts a newline after the
# name; zai-org/GLM-4.7-Flash's template renders the same structure with
# no whitespace at all (name immediately followed by "<arg_key>"), so the
# separator between the name and the arg pairs must be optional, not a
# required "\n". The name itself is matched as non-whitespace, non-"<" so
# it can't swallow the next tag when there is no separator.
_GLM_CALL_RE = re.compile(
    r"<tool_call>([^\s<]+)\s*((?:<arg_key>.*?</arg_key>\s*<arg_value>.*?</arg_value>\s*)*)</tool_call>",
    re.DOTALL,
)
_GLM_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)

# Functionary (MeetKai) format: ">>>NAME\n{json args}". ">>>all" prefixes
# plain (non-tool-call) content, not a call. Verified against
# meetkai/functionary-small-v3.2's real tokenizer chat template output.
# Only the ">>>NAME\n" marker is matched by regex; the JSON body itself is
# located with _extract_balanced_json_object (not a "terminated by the
# next >>> or end of string" lookahead) because real generated text has a
# trailing special token (e.g. "<|eot_id|>") rendered as literal text
# right after the closing "}", which such a lookahead can't see past.
_FUNCTIONARY_CALL_RE = re.compile(r">>>(?!all\b)(\S+)\n(?=\{)", re.DOTALL)


def _normalise_tool_calls(parsed: list[dict]) -> list[dict] | None:
    """Turn a list of {name, arguments|parameters} / OpenAI-style dicts into
    the normalised ``{"type": "function", "function": {"name", "arguments"}}``
    shape the rollout loop expects. Anything else (e.g. a plain ``[2, 3]``
    the model wrote as its answer) is not a call: returns None."""
    if (
        not parsed
        or not all(isinstance(c, dict) for c in parsed)
        or not {"name", "function"} & set(parsed[0])
    ):
        return None
    normalised = []
    for call in parsed:
        if "type" in call and call["type"] == "function" and isinstance(call.get("function"), dict):
            fn = call["function"]
            if isinstance(fn.get("arguments"), dict):
                fn["arguments"] = json.dumps(fn["arguments"])
            normalised.append(call)
        else:
            # Llama's built-in JSON tool format is flatter than the OpenAI
            # shape above: {"type": "function", "function": "NAME",
            # "parameters": {...}} -- "function" is the name itself (a
            # string), not a nested {name, arguments} dict.
            name = call["function"] if isinstance(call.get("function"), str) else call.get("name")
            args = call.get("arguments", call.get("parameters", {}))
            if isinstance(args, dict):
                args = json.dumps(args)
            normalised.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args,
                    },
                }
            )
    return normalised


def _extract_tool_calls(completion_msg: Any, raw_text: str) -> list[dict] | None:
    # Strip <think>...</think> blocks before any parsing
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    if isinstance(completion_msg, dict):
        if isinstance(completion_msg.get("content"), str):
            completion_msg = dict(completion_msg)
            completion_msg["content"] = re.sub(
                r"<think>.*?</think>", "", completion_msg["content"], flags=re.DOTALL
            ).strip()
        tc = completion_msg.get("tool_calls")
        if tc:
            return tc if isinstance(tc, list) else [tc]

    # Cohere (Command R7B/A, North, Command R / Aya Expanse, and the
    # tools_fallback_prompt list), with or without its special-token markers.
    # Checked first: the generic JSON scan below would take a "[...]" in the
    # thinking block or miss the tool_name/parameters keys.
    cohere_calls = (
        _parse_cohere_action(raw_text)
        or _parse_cohere_legacy_action(raw_text)
        or _parse_cohere_json(raw_text)
    )
    if cohere_calls:
        return cohere_calls

    # Tolerate both the strict <tool_call> tag (Qwen chat template) and the
    # no-underscore <toolcall> variant some fine-tunes imitate, including
    # dangling/mismatched closing tags, before attempting JSON.
    cleaned = re.sub(r"</?tool_?call\s*>", "", raw_text, flags=re.IGNORECASE).strip()
    parsed = _load_first_json_object(cleaned)
    if parsed is not None:
        normalised = _normalise_tool_calls(parsed if isinstance(parsed, list) else [parsed])
        if normalised is not None:
            return normalised

    # Llama 3.1/3.2 built-in tool format. Validate each argument blob is real
    # JSON so a malformed call is skipped rather than passed downstream as an
    # unparseable "arguments" string.
    llama_calls = []
    for name, args in _LLAMA_FUNC_RE.findall(raw_text):
        try:
            json.loads(args)
        except json.JSONDecodeError:
            continue
        llama_calls.append({"name": name.strip(), "arguments": args})
    if llama_calls:
        normalised = _normalise_tool_calls(llama_calls)
        if normalised is not None:
            return normalised

    # Qwen3.5's native format: <function=NAME><parameter=K>v</parameter>...</function>
    qwen35_calls = []
    for name, body in _QWEN35_FUNCTION_RE.findall(raw_text):
        args = {}
        for k, v in _QWEN35_PARAMETER_RE.findall(body):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            args[k.strip()] = v
        if args:
            qwen35_calls.append({"name": name.strip(), "arguments": json.dumps(args)})
    if qwen35_calls:
        normalised = _normalise_tool_calls(qwen35_calls)
        if normalised is not None:
            return normalised

    # Gemma 4's native format: <|tool_call>call:NAME{k:<|"|>v<|"|>,...}<tool_call|>
    gemma4_calls = []
    for name, body in _GEMMA4_TOOL_CALL_RE.findall(raw_text):
        args = {}
        for k, quoted, bare in _GEMMA4_ARG_RE.findall(body):
            if quoted:
                v = quoted
            else:
                try:
                    v = json.loads(bare)
                except json.JSONDecodeError:
                    v = bare
            args[k.strip()] = v
        if args:
            gemma4_calls.append({"name": name.strip(), "arguments": json.dumps(args)})
    if gemma4_calls:
        normalised = _normalise_tool_calls(gemma4_calls)
        if normalised is not None:
            return normalised

    # Mistral v0.3+ "[TOOL_CALLS] [...]" format.
    mistral_match = _MISTRAL_TOOL_CALLS_RE.search(raw_text)
    if mistral_match:
        try:
            parsed = json.loads(mistral_match.group(1))
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            normalised = _normalise_tool_calls(parsed if isinstance(parsed, list) else [parsed])
            if normalised is not None:
                return normalised

    # Newer Mistral "mistral_common" backend, sampling-mode
    # "[TOOL_CALLS]NAME[ARGS]{json}" format.
    bracket_calls = []
    for m in _MISTRAL_BRACKET_NAME_RE.finditer(raw_text):
        args = _extract_balanced_json_object(raw_text, m.end())
        if args is None:
            continue
        try:
            json.loads(args)
        except json.JSONDecodeError:
            continue
        bracket_calls.append({"name": m.group(1).strip(), "arguments": args})
    if bracket_calls:
        normalised = _normalise_tool_calls(bracket_calls)
        if normalised is not None:
            return normalised

    # Same backend, greedy-mode "[NAME]{json}" format (no special tokens).
    # Only reached when the sampling-mode format above found nothing, so
    # this never runs against text that actually contains "[TOOL_CALLS]"/
    # "[ARGS]" (the regex also excludes those two names defensively).
    plain_bracket_calls = []
    for m in _MISTRAL_PLAIN_BRACKET_NAME_RE.finditer(raw_text):
        args = _extract_balanced_json_object(raw_text, m.end())
        if args is None:
            continue
        try:
            json.loads(args)
        except json.JSONDecodeError:
            continue
        plain_bracket_calls.append({"name": m.group(1).strip(), "arguments": args})
    if plain_bracket_calls:
        normalised = _normalise_tool_calls(plain_bracket_calls)
        if normalised is not None:
            return normalised

    # DeepSeek-V3/R1 format.
    deepseek_calls = []
    for name, args in _DEEPSEEK_CALL_RE.findall(raw_text):
        try:
            json.loads(args)
        except json.JSONDecodeError:
            continue
        deepseek_calls.append({"name": name.strip(), "arguments": args})
    if deepseek_calls:
        normalised = _normalise_tool_calls(deepseek_calls)
        if normalised is not None:
            return normalised

    # GLM-4.5 format: same <tool_call> tag as Qwen, but arg_key/arg_value
    # pairs instead of JSON, so it wasn't caught by the JSON path above.
    glm_calls = []
    for name, body in _GLM_CALL_RE.findall(raw_text):
        args = {}
        for k, v in _GLM_ARG_RE.findall(body):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            args[k.strip()] = v
        if args:
            glm_calls.append({"name": name.strip(), "arguments": json.dumps(args)})
    if glm_calls:
        normalised = _normalise_tool_calls(glm_calls)
        if normalised is not None:
            return normalised

    # Functionary (MeetKai) format.
    functionary_calls = []
    for m in _FUNCTIONARY_CALL_RE.finditer(raw_text):
        args = _extract_balanced_json_object(raw_text, m.end())
        if args is None:
            continue
        try:
            json.loads(args)
        except json.JSONDecodeError:
            continue
        functionary_calls.append({"name": m.group(1).strip(), "arguments": args})
    if functionary_calls:
        normalised = _normalise_tool_calls(functionary_calls)
        if normalised is not None:
            return normalised

    # OpenAI gpt-oss "Harmony" format: <|channel|>commentary
    # to=functions.NAME<|constrain|>json<|message|>{...}<|call|>
    harmony_calls = _parse_harmony_tool_calls(raw_text)
    if harmony_calls:
        return harmony_calls

    # ReAct "Action: name\nAction Input: {...}", then a fenced JSON block after
    # a "tool_call"/"function" marker. tool_call_parse returns them normalised.
    calls = _parse_react(raw_text) or _parse_fenced_after_name(raw_text)
    if calls:
        return calls

    # Last resort: every format tool_call_parse knows (e.g. <function=...>
    # blocks whose closing tags the template never emits).
    return extract_tool_calls_from_text(raw_text)


def _assign_tool_call_ids(tool_calls: list[dict]) -> list[dict]:
    """Give each parsed tool call a stable id, in place.

    None of the wire formats _extract_tool_calls parses carry a call id, so
    it never sets one. But the assistant turn appended to the conversation
    needs its tool_calls to carry the SAME ids as the "tool" result
    messages appended right after -- some client-side chat template
    validators (e.g. Mistral's mistral_common tokenizer) reject a "tool"
    message whose tool_call_id doesn't match one from the immediately
    preceding assistant turn's tool_calls list.

    The id is a bare 9-character alphanumeric string (no separators) since
    mistral_common's own validator rejects anything else outright ("must
    be a-z, A-Z, 0-9, with a length of 9") -- a readable "call_<name>"
    style id, fine for every other family, fails that check specifically.
    """
    for tc in tool_calls:
        if not tc.get("id"):
            tc["id"] = uuid.uuid4().hex[:9]
    return tool_calls


# ─────────────────────────────────────────────────────────────────────────────
# GRPO-compatible output formatter
#
# FIX #4: GRPOTrainer._generate_single_turn looks for tool/env mask as:
#     tool_mask = extra_fields.pop("env_mask", None)
# The key MUST be "env_mask", not "tool_masks" or anything else.
#
# FIX #1 applied here: logprobs returned as list[list[float]], not tuples.
# FIX #2 applied here: prompt_ids returned as list[list[int]].
# FIX #3 applied here: completion_ids returned as list[list[int]].
# ─────────────────────────────────────────────────────────────────────────────
def _wrap_reward_fn(reward_fn: Callable) -> Callable:
    """
    Normalises any reward function signature into:
        fn(responses: list[str], prompts: list[str]) -> list[float]

    Handles all three calling conventions:
        1. fn(responses, prompts)          — batch, your current style
        2. fn(responses)                   — batch, no prompts
        3. fn(trajectory: Trajectory)      — per-trajectory object
    """
    sig = inspect.signature(reward_fn)
    params = list(sig.parameters.values())

    def _wrapped(responses: list[str], prompts: list[str], trajectories: list) -> list[float]:
        # ── Style 3: single Trajectory object ────────────────────────────
        # Detected if the first param has a annotation or name suggesting Trajectory,
        # OR if calling with a list raises TypeError (runtime fallback).
        first_param = params[0] if params else None
        is_trajectory_style = first_param is not None and (
            getattr(first_param.annotation, "__name__", "") == "Trajectory"
            or first_param.name in ("trajectory", "traj", "t")
        )

        if is_trajectory_style:
            scores = [float(reward_fn(t)) for t in trajectories]
            return scores

        completions = []
        answers = []
        prompt_texts = []
        extra_cols: dict[str, list] = {}
        for t, resp, p in zip(trajectories, responses, prompts, strict=False):
            conv = (t.metadata or {}).get("conversation")
            msgs = conv or (t.to_trl_format().get("messages") or [])
            # Score what the model generated: the turns from its first reply on,
            # not the system/user prompt. The single-turn path records only the
            # prompt, so it falls back to the response text.
            first = next(
                (
                    i
                    for i, m in enumerate(msgs)
                    if isinstance(m, dict) and m.get("role") == "assistant"
                ),
                None,
            )
            completions.append(msgs[first:] if first is not None else resp)
            ans = (t.metadata or {}).get("answer")
            if ans is None and isinstance(p, dict):
                ans = p.get("answer")
            answers.append(ans)
            if isinstance(p, dict):
                prompt_texts.append(p.get("prompt") or p.get("content") or str(p))
                # Any other dataset column riding along on the prompt dict (e.g. a
                # custom gold_answer/gold_chunk_ids) -- mirrors TRL's own GRPO/RLOO
                # reward_funcs convention of forwarding every dataset column as a
                # same-named kwarg, which this rollout-mode path (DPO/BCO/PPO) didn't
                # do before, silently starving reward_fn of anything but "answer".
                for k, v in p.items():
                    if k in ("prompt", "content", "answer"):
                        continue
                    extra_cols.setdefault(k, []).append(v)
            else:
                prompt_texts.append(p if isinstance(p, str) else str(p))
        extra = {k: v for k, v in extra_cols.items() if len(v) == len(trajectories)}
        if any(a is not None and a != "" for a in answers):
            extra["answer"] = answers
        # Per-completion tool-call counts, same source TRL's own GRPO/RLOO path uses
        # (see tool_call_counts in _format_for_grpo below) -- previously only reached
        # reward_fn there, leaving DPO/BCO/PPO's reward function always seeing
        # tool_call_counts=None even when tools actually fired during the rollout.
        extra["tool_call_counts"] = [
            (t.metadata or {}).get("tool_call_count", 0) for t in trajectories
        ]

        # ── Style 1 or 2: batch call ──────────────────────────────────────
        # Richest call first; then the pre-#11 fn(responses, prompts=...) /
        # fn(responses) styles, which take no extra columns; then per-trajectory.
        calls = (
            lambda: reward_fn(completions, prompts=prompt_texts, **extra),
            lambda: reward_fn(completions, **extra),
            lambda: reward_fn(responses, prompts=prompt_texts),
            lambda: reward_fn(responses),
        )
        for call in calls:
            try:
                result = call()
                break
            except TypeError:
                continue
        else:
            result = [float(reward_fn(t)) for t in trajectories]

        return [float(s) for s in result]

    return _wrapped


def _format_for_grpo(
    trajectories: list[Trajectory],
    prompts: list[str],
    tools_enabled: bool,
) -> dict[str, Any]:
    prompt_ids_out = [
        _ids_to_list(t.metadata.get("prompt_ids") if t.metadata else None) for t in trajectories
    ]

    completion_ids_out = [
        _ids_to_list(t.metadata.get("completion_ids") if t.metadata else None) for t in trajectories
    ]

    # GRPOTrainer does torch.tensor(logps) directly on each sequence,
    # so must be list[list[float]] — plain floats, NOT (logprob, token_id)
    # tuples. TRL's _generate_and_score_completions does
    # `torch.tensor(logps)` on this to build sampling_per_token_logps (shape
    # [B, N]); tuples would make it [B, N, 2] and crash the importance-sampling
    # subtraction `old_per_token_logps - sampling_per_token_logps` under vLLM
    # (where old_per_token_logps is always computed). The transformers path
    # tolerated tuples only because it sets old_per_token_logps=None and skips
    # that subtraction — but emitting floats is correct for both paths, so we
    # normalise to floats here regardless of backend.
    logprobs_out = [
        (
            [float(lp[0]) if isinstance(lp, tuple | list) else float(lp) for lp in t.logprobs]
            if t.logprobs
            else []
        )
        for t in trajectories
    ]

    env_mask_out = [t.metadata.get("tool_mask", None) if t.metadata else None for t in trajectories]

    return {
        "prompt_ids": prompt_ids_out,
        "completion_ids": completion_ids_out,
        "logprobs": logprobs_out,
        "env_mask": env_mask_out,
        "queries": prompts,
        "responses": [t.final_response for t in trajectories],
        "rewards": [t.reward for t in trajectories],
        "trajectories": trajectories,
        "tools_used": tools_enabled,
        "conversations": [
            t.metadata.get("conversation", []) if t.metadata else [] for t in trajectories
        ],
        "tool_call_counts": [
            t.metadata.get("tool_call_count", 0) if t.metadata else 0 for t in trajectories
        ],
        # FinDER golden-chunk-recall plumbing: forwarded to reward fns as the
        # `retrieved_chunk_ids` kwarg (TRL passes rollout-output extra keys
        # through to reward_funcs exactly like tool_call_counts).
        "retrieved_chunk_ids": [
            t.metadata.get("retrieved_chunk_ids", []) if t.metadata else [] for t in trajectories
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Engine factory
# ─────────────────────────────────────────────────────────────────────────────


def create_rollout_engine(
    backend: str = "auto",
    model=None,
    tokenizer=None,
    model_path: str | None = None,
    api_provider: str | None = None,
    api_model: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> RolloutEngine:
    if backend == "auto":
        backend = _detect_best_backend()

    if backend == "vllm":
        try:
            from .vllm_engine import VLLMRolloutEngine

            path = model_path or (model if isinstance(model, str) else None)
            if not path:
                raise ValueError("vLLM backend requires model_path or a model string name")
            return VLLMRolloutEngine(path, **kwargs)
        except ImportError as e:
            warnings.warn(f"vLLM not available ({e}), falling back to transformers.", UserWarning)
            backend = "transformers"

    if backend == "transformers":
        if model is None and model_path:
            from transformers import AutoTokenizer

            device_map = kwargs.pop("device_map", "auto")
            dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", torch.bfloat16))
            # Quantization (e.g. transformers.BitsAndBytesConfig(load_in_4bit=True))
            # to cut memory usage for the model_path convenience path -- previously
            # only reachable by pre-loading the model yourself and passing
            # model=/tokenizer= instead (see the vLLM backend's model_init_kwargs
            # passthrough in vllm_engine.py, which already supported this).
            quantization_config = kwargs.pop("quantization_config", None)
            # Checkpoints shipping custom modeling code (e.g.
            # microsoft/Phi-mini-MoE-instruct's PhiMoEForCausalLM) need this to
            # load at all -- load_causal_or_multimodal_model already reads it
            # from kwargs, this path just never forwarded it into load_kwargs.
            trust_remote_code = kwargs.pop("trust_remote_code", False)

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )
            load_kwargs = {
                "device_map": device_map,
                "dtype": dtype,
                "trust_remote_code": trust_remote_code,
            }
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
            model = load_causal_or_multimodal_model(model_path, **load_kwargs)
        from .transformers_engine import TransformersRolloutEngine

        return TransformersRolloutEngine(model, tokenizer, **kwargs)

    # if backend == "api":
    #     from .api_engine import APIRolloutEngine
    #     return APIRolloutEngine(
    #         provider=api_provider, model=api_model, api_key=api_key, **kwargs
    #     )
    if backend == "api":
        from .api_engine import APIRolloutEngine

        return APIRolloutEngine(
            model=api_model or "gpt-4o-mini",
            api_key=api_key,
            **kwargs,
        )

    raise ValueError(f"Unknown backend: '{backend}'")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def create_rollout_fn(
    rollout_engine: RolloutEngine | None = None,
    rollout_backend: str | None = None,
    model=None,
    tokenizer=None,
    model_path: str | None = None,
    api_provider: str | None = None,
    api_model: str | None = None,
    api_key: str | None = None,
    api_base_url: str | None = None,
    engine_kwargs: dict | None = None,
    tools: list | None = None,
    max_steps: int = 20,
    reward_fn: Callable | None = None,
    custom_rollout_fn: Callable | None = None,
    pre_step_hook: Callable | None = None,
    post_step_hook: Callable | None = None,
    on_trajectory_end: Callable | None = None,
    enable_thinking: bool = False,
    system_prompt: str | None = None,
    async_rollouts: bool = False,
    force_final_answer: bool = False,
    force_action_on_stall: bool = False,
    context_length: int | None = None,
    tools_fallback_prompt: str | None = DEFAULT_TOOLS_FALLBACK_PROMPT,
    tool_result_format: str = DEFAULT_TOOL_RESULT_FORMAT,
) -> Callable:
    """Build a tool-calling rollout function (see ``_execute_trajectory``).

    ``tools_fallback_prompt``: when the chat template ignores ``tools`` (Tiny
    Aya, Aya Expanse, Aya Vision, North), this text (``{tools}`` = the JSON
    schemas) is added to the system prompt and tool results are folded into
    user turns as ``tool_result_format`` (``{name}``, ``{content}``). Pass
    ``None`` to raise a ``ValueError`` instead.
    """
    if custom_rollout_fn is not None:

        def _wrapped(prompts, *args, **kw):
            result = custom_rollout_fn(prompts, *args, **kw)
            if on_trajectory_end and "trajectories" in result:
                for t in result["trajectories"]:
                    on_trajectory_end(t)
            return result

        return _wrapped

    # ── Eagerly build engine if backend string given ──────────────────────────
    if rollout_engine is None and rollout_backend is not None:
        rollout_engine = create_rollout_engine(
            backend=rollout_backend,
            model=model,
            tokenizer=tokenizer,
            model_path=model_path,
            api_provider=api_provider,
            api_model=api_model,
            api_key=api_key,
            base_url=api_base_url,
            **(engine_kwargs or {}),
        )

    resolved_tools = tools or []
    tools_enabled = len(resolved_tools) > 0

    if not tools_enabled:
        warnings.warn(
            "No tools provided — running in single-turn mode.",
            UserWarning,
            stacklevel=2,
        )

    def rollout_fn(prompts: Any, trainer: Any = None, *args, **gen_kwargs) -> dict[str, Any]:
        if isinstance(prompts, str):
            prompts = [prompts]
        elif isinstance(prompts, dict):
            prompts = [prompts]
        elif (
            isinstance(prompts, list)
            and prompts
            and isinstance(prompts[0], dict)
            and prompts[0].get("role") is not None
        ):
            prompts = [prompts]

        engine = rollout_engine  # ← captures from outer scope

        # ── Lazy init — only if no engine AND no trainer ──────────────────────
        if engine is None and trainer is None:
            engine = create_rollout_engine(
                backend=rollout_backend or "auto",
                model=model,
                tokenizer=tokenizer,
                model_path=model_path,
                api_provider=api_provider,
                api_model=api_model,
                api_key=api_key,
                base_url=api_base_url,  # ← was missing here too
                **(engine_kwargs or {}),
            )

        trajectories = []
        # for prompt in prompts:
        #     traj = _execute_trajectory(
        #         prompt=prompt,
        #         tools=resolved_tools,
        #         tools_enabled=tools_enabled,
        #         engine=engine,
        #         max_steps=max_steps,
        #         pre_step_hook=pre_step_hook,
        #         post_step_hook=post_step_hook,
        #         gen_kwargs=gen_kwargs,
        #         system_prompt=system_prompt,
        #         trainer=trainer,
        #     )
        #     trajectories.append(traj)

        trajectories = _run_trajectories(
            prompts=prompts,
            tools=resolved_tools,
            tools_enabled=tools_enabled,
            engine=engine,
            max_steps=max_steps,
            pre_step_hook=pre_step_hook,
            post_step_hook=post_step_hook,
            gen_kwargs=gen_kwargs,
            system_prompt=system_prompt,
            trainer=trainer,
            enable_thinking=enable_thinking,
            async_rollouts=async_rollouts,
            force_final_answer=force_final_answer,
            force_action_on_stall=force_action_on_stall,
            context_length=context_length,
            tools_fallback_prompt=tools_fallback_prompt,
            tool_result_format=tool_result_format,
        )

        if reward_fn:
            _reward = _wrap_reward_fn(reward_fn)
            responses = [t.final_response for t in trajectories]
            # Keep dict prompts (e.g. {"prompt": ..., "gold_answer": ...}) intact --
            # _wrap_reward_fn's _wrapped() already branches on isinstance(p, dict) to
            # pull out extra dataset columns, but stringifying every non-str prompt
            # here first made that branch unreachable, silently dropping every column
            # beyond the prompt text before reward_fn ever saw it.
            prompts_list = list(prompts)
            scores = _reward(responses, prompts_list, trajectories)
            for t, score in zip(trajectories, scores, strict=False):
                t.reward = score

        if on_trajectory_end:
            for t in trajectories:
                on_trajectory_end(t)

        return _format_for_grpo(trajectories, prompts, tools_enabled)

    return rollout_fn


# ─────────────────────────────────────────────────────────────────────────────
# Core trajectory execution
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Trajectory dispatcher — sync (default) or async
# ─────────────────────────────────────────────────────────────────────────────


def _run_trajectories(
    prompts,
    tools,
    tools_enabled,
    engine,
    max_steps,
    pre_step_hook,
    post_step_hook,
    gen_kwargs,
    system_prompt=None,
    trainer=None,
    async_rollouts: bool = False,
    enable_thinking: bool = False,
    force_final_answer: bool = False,
    force_action_on_stall: bool = False,
    context_length: int | None = None,
    tools_fallback_prompt: str | None = DEFAULT_TOOLS_FALLBACK_PROMPT,
    tool_result_format: str = DEFAULT_TOOL_RESULT_FORMAT,
) -> list:
    if not async_rollouts or trainer is not None:
        # ── Synchronous — always safe, required when trainer is present ───
        trajectories = []
        for prompt in prompts:
            traj = _execute_trajectory(
                prompt=prompt,
                tools=tools,
                tools_enabled=tools_enabled,
                engine=engine,
                max_steps=max_steps,
                pre_step_hook=pre_step_hook,
                post_step_hook=post_step_hook,
                gen_kwargs=gen_kwargs,
                system_prompt=system_prompt,
                trainer=trainer,
                enable_thinking=enable_thinking,
                force_final_answer=force_final_answer,
                context_length=context_length,
                force_action_on_stall=force_action_on_stall,
                tools_fallback_prompt=tools_fallback_prompt,
                tool_result_format=tool_result_format,
            )
            trajectories.append(traj)
        return trajectories

    # ── Async — only when async_rollouts=True AND trainer is None ─────────
    # Best for: API engines, vLLM standalone, external tool calls.
    # trainer=None guard: model.generate() is not thread-safe under DDP/FSDP.
    from concurrent.futures import ThreadPoolExecutor

    _executor = ThreadPoolExecutor(max_workers=min(len(prompts), 8))

    async def _one(prompt):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: _execute_trajectory(
                prompt=prompt,
                tools=tools,
                tools_enabled=tools_enabled,
                engine=engine,
                max_steps=max_steps,
                pre_step_hook=pre_step_hook,
                post_step_hook=post_step_hook,
                gen_kwargs=gen_kwargs,
                system_prompt=system_prompt,
                trainer=None,
                enable_thinking=enable_thinking,
                force_final_answer=force_final_answer,
                force_action_on_stall=force_action_on_stall,
                tools_fallback_prompt=tools_fallback_prompt,
                tool_result_format=tool_result_format,
            ),
        )

    async def _gather():
        return await asyncio.gather(*[_one(p) for p in prompts])

    try:
        asyncio.get_running_loop()
        from concurrent.futures import ThreadPoolExecutor as _TPE

        with _TPE(max_workers=1) as pool:
            return pool.submit(asyncio.run, _gather()).result()
    except RuntimeError:
        return asyncio.run(_gather())


# Mid-trajectory stall-recovery nudge (see force_action_on_stall in
# _execute_trajectory). A generic "take an action" phrasing is not enough —
# verified interactively that a fine-tuned M1 model, when nudged this way,
# still re-emits a <state> block and stops again (the pattern is heavily
# reinforced by training). Explicitly forbidding the <state> tag, tested the
# same way, reliably produces a clean tool call or answer instead.
_STALL_NUDGE_TEXT = (
    "You did not take an action. In your NEXT message, do NOT include a "
    "<state> tag at all. Output ONLY one of: a tool call (to search for the "
    "next open question), or your final answer wrapped in "
    "<answer></answer> tags."
)


def _execute_trajectory(
    prompt: str,
    tools: list,
    tools_enabled: bool,
    engine: RolloutEngine | None,
    max_steps: int,
    pre_step_hook: Callable | None,
    post_step_hook: Callable | None,
    gen_kwargs: dict,
    system_prompt: str | None = None,
    trainer: Any = None,
    enable_thinking: bool = False,
    force_final_answer: bool = False,
    force_action_on_stall: bool = False,
    context_length: int | None = None,
    tools_fallback_prompt: str | None = DEFAULT_TOOLS_FALLBACK_PROMPT,
    tool_result_format: str = DEFAULT_TOOL_RESULT_FORMAT,
) -> Trajectory:
    gold = None
    if isinstance(prompt, dict) and prompt.get("role") is None and "prompt" in prompt:
        gold = prompt.get("answer")
        prompt = prompt["prompt"]

    # ── Resolve tokenizer ─────────────────────────────────────────────────────
    if trainer is not None:
        tokenizer = trainer.processing_class
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer
    elif engine is not None:
        tokenizer = engine._get_tokenizer()
    else:
        raise ValueError("Either `trainer` or `engine` must be provided.")

    use_vllm = engine is not None and _is_vllm_engine(engine)

    # ── Tool setup ────────────────────────────────────────────────────────────
    tool_callable_map: dict[str, Callable] = {}
    for t in tools:
        name = t.name if hasattr(t, "name") else t.__name__
        tool_callable_map[name] = _get_callable(t)

    tool_schemas = []
    for t in tools:
        schema = None
        if hasattr(t, "to_schema"):
            schema = t.to_schema()
        elif callable(t):
            schema = _build_tool_schema(t, use_vllm=use_vllm)
        if schema:
            tool_schemas.append(schema)

    all_completion_ids: list[int] = []
    all_logprobs: list[tuple] = []
    all_tool_mask: list[int] = []

    # ── Prompt-id helper ──────────────────────────────────────────────────────
    def _compute_prompt_ids(conversation: list[dict], schemas: list[dict]) -> list[int]:
        if tokenizer is None:
            return []
        try:
            ids = _render_chat_template(
                tokenizer,
                conversation,
                schemas,
                enable_thinking=enable_thinking,
                tool_result_format=tool_result_format,
                tokenize=True,
                return_tensors=None,
            )
        except Exception:
            ids = tokenizer.encode(
                prompt if isinstance(prompt, str) else str(prompt),
                add_special_tokens=True,
            )
        if isinstance(ids, dict):
            ids = ids.get("input_ids", [])
        return _ids_to_list(ids)

    # ── Generation helper ─────────────────────────────────────────────────────
    def _gen(conversation: list[dict], schemas: list[dict]):
        # ── Hard prompt-length backstop ───────────────────────────────────────
        # Never send an over-window prompt to the generator: it deadlocks the
        # vLLM server's rendering step (a 20K-token prompt froze at "Rendering
        # prompts 0%"). Resolve the prompt budget from the vLLM window minus the
        # completion budget (training rollout has no context_length passed, so
        # fall back to trainer.args). The old `_check_context_length` guard
        # fails OPEN (returns True) when its tokenizer is unresolved — which it
        # always is on the trainer-based vLLM path (no standalone engine) — so
        # it never fired here. This truncation uses a reliably-resolved
        # tokenizer and drops oldest messages until the applied template fits.
        _gen_budget: int | None = None
        if context_length is not None:
            _gen_budget = max(int(context_length) - 256, 512)
        elif trainer is not None and getattr(trainer, "use_vllm", False):
            _targs = getattr(trainer, "args", None)
            _vllm_len = getattr(_targs, "vllm_max_model_length", None) if _targs else None
            _compl = getattr(_targs, "max_completion_length", None) if _targs else None
            if _vllm_len:
                _gen_budget = max(int(_vllm_len) - (int(_compl) if _compl else 2048) - 64, 512)
        if _gen_budget is not None:
            _tok = tokenizer
            if _tok is None and trainer is not None:
                _pc = getattr(trainer, "processing_class", None)
                _tok = getattr(_pc, "tokenizer", None) or _pc
            if _tok is not None:
                _truncate_conversation_for_vllm(
                    conversation,
                    _tok,
                    schemas,
                    _gen_budget,
                    enable_thinking=enable_thinking,
                    tool_result_format=tool_result_format,
                )

        # ── Path A: trainer-based generation ─────────────────────────────────
        if trainer is not None:
            prompt_text = _render_chat_template(
                tokenizer,
                conversation,
                schemas,
                enable_thinking=enable_thinking,
                tool_result_format=tool_result_format,
                tokenize=False,
            )

            # ── vLLM trainer path — custom rollout via TRL's colocated vLLM ──
            # Uses trl.experimental.openenv.generate_rollout_completions, the
            # supported helper for custom agentic rollouts under use_vllm: it
            # generates via trainer.vllm_generation (kept weight-synced by the
            # _generate_single_turn patch's sync_weights() call) and returns
            # prompt_ids/completion_ids/logprobs/text. Imported lazily so the
            # module imports cleanly on trl builds lacking trl.experimental.
            if getattr(trainer, "use_vllm", False):
                from trl.experimental.openenv import generate_rollout_completions

                try:
                    rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
                except Exception as e:
                    if "maximum context length" not in str(e):
                        raise
                    # Safety net for guard/tokeniser mismatch (the context
                    # guard estimates prompt length itself; vLLM counts the
                    # rendered string — a hair's difference can slip through).
                    # Return an EMPTY completion: no tool calls in empty text,
                    # so the multi-turn loop exits naturally and the trajectory
                    # terminates (zero reward) instead of killing the whole
                    # training run with a VLLMValidationError mid-step.
                    return (
                        "",
                        {"role": "assistant", "content": ""},
                        [],
                        {
                            "prompt_ids": [],
                            "completion_ids": [],
                        },
                    )

                raw_vis = rollout_outputs.get("text") or tokenizer.decode(
                    rollout_outputs["completion_ids"], skip_special_tokens=True
                )
                try:
                    raw = tokenizer.decode(
                        rollout_outputs["completion_ids"], skip_special_tokens=False
                    )
                except Exception:
                    raw = raw_vis
                comp_ids = _ids_to_list(rollout_outputs["completion_ids"])
                raw_lps = rollout_outputs["logprobs"]
                lp_tuples = _ensure_logprob_tuples(_normalise_logprobs(raw_lps, comp_ids))
                raw_prompt_ids = rollout_outputs.get("prompt_ids", [])
                meta = {
                    "prompt_ids": _ids_to_list(raw_prompt_ids),
                    "completion_ids": comp_ids,
                }

            # ── Normal transformers trainer path (NEW) ────────────────────────
            else:
                from contextlib import nullcontext

                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from trl.models import unwrap_model_for_generation

                enc = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                )
                enc = {k: v.to(trainer.accelerator.device) for k, v in enc.items()}
                prompt_len = enc["input_ids"].shape[1]

                with (
                    unwrap_model_for_generation(
                        # trainer.model_wrapped,
                        getattr(trainer, "model_wrapped", trainer.model),
                        trainer.accelerator,
                        gather_deepspeed3_params=getattr(
                            trainer.args, "ds3_gather_for_generation", False
                        ),
                        generation_kwargs=getattr(trainer, "generation_kwargs", {}),
                    ) as unwrapped_model,
                    torch.no_grad(),
                    (
                        FSDP.summon_full_params(trainer.model_wrapped, recurse=False)
                        if trainer.is_fsdp_enabled
                        else nullcontext()
                    ),
                ):
                    gen_config = getattr(trainer, "generation_config", None)
                    # Use the already-unwrapped `tokenizer` (resolved at the top
                    # of _execute_trajectory, which handles processing_class ->
                    # .tokenizer unwrapping for VL processors like Qwen3VLProcessor
                    # that lack pad_token_id directly). trainer.processing_class
                    # itself may be a VL processor without pad_token_id.
                    _pad = getattr(tokenizer, "pad_token_id", None) or getattr(
                        tokenizer, "eos_token_id", None
                    )
                    generate_kwargs = dict(
                        **enc,
                        max_new_tokens=getattr(trainer, "_max_new_tokens", 256),
                        temperature=getattr(trainer, "_temperature", 0.7),
                        do_sample=True,
                        pad_token_id=_pad,
                        disable_compile=True,
                    )
                    if gen_config is not None:
                        generate_kwargs["generation_config"] = gen_config

                    # prompt_completion_ids = unwrapped_model.generate(**generate_kwargs)
                    prompt_completion_ids = getattr(
                        unwrapped_model, "policy", unwrapped_model
                    ).generate(**generate_kwargs)

                device = trainer.accelerator.device
                comp = prompt_completion_ids[:, prompt_len:]  # (1, T) strip prompt

                # mask after first EOS — same logic as GRPOTrainer
                # is_eos = comp == trainer.eos_token_id
                # Use the unwrapped tokenizer's eos_token_id (processing_class
                # may be a VL processor without eos_token_id directly).
                _eos = getattr(trainer, "eos_token_id", None) or getattr(
                    tokenizer, "eos_token_id", None
                )
                is_eos = comp == _eos
                eos_idx = torch.full((1,), comp.size(1), dtype=torch.long, device=device)
                if is_eos.any():
                    eos_idx[0] = is_eos[0].int().argmax()
                seq_indices = torch.arange(comp.size(1), device=device)
                comp_mask = (seq_indices <= eos_idx[0]).int()

                comp_ids = comp[0][comp_mask.bool()].tolist()
                raw_vis = tokenizer.decode(comp_ids, skip_special_tokens=True)
                raw = tokenizer.decode(comp_ids, skip_special_tokens=False)

                # zeros fine — GRPOTrainer recomputes logprobs in forward pass
                lp_tuples = [(0.0, tid) for tid in comp_ids]
                meta = {
                    "prompt_ids": enc["input_ids"][0].tolist(),
                    "completion_ids": comp_ids,
                }

            # ── shared tail for both trainer paths ────────────────────────────
            all_completion_ids.extend(comp_ids)
            all_logprobs.extend(lp_tuples)
            all_tool_mask.extend([1] * len(comp_ids))

            structured = {"role": "assistant", "content": _strip_special_tokens(raw_vis, tokenizer)}
            return raw, structured, lp_tuples, meta

        # ── Path B: standalone engine-based generation (unchanged) ───────────
        assert engine is not None
        # Enforce the context cap at the ENGINE level too. The _ctx_max_len
        # guard above stops the tool LOOP when the conversation exceeds the
        # budget, but the standalone transformers engine defaults its own
        # `max_length` to 40960 regardless — so a single over-long conversation
        # slips past the guard and SDPA materialises a full attention matrix
        # (97GB OOM on the eval path). Thread the cap through so the engine's
        # own left-truncation at max_length becomes the hard backstop.
        _engine_gen_cfg = dict(gen_kwargs) if gen_kwargs else {}
        if context_length is not None:
            _engine_gen_cfg.setdefault("max_length", context_length)
        # create_rollout_fn's own `enable_thinking` param never reached this
        # gen_cfg dict before -- only the trainer-based _gen path (above)
        # threaded it through, so a standalone engine (transformers or
        # vllm) always saw the engine's own hardcoded default regardless of
        # what the caller asked for.
        _engine_gen_cfg.setdefault("enable_thinking", enable_thinking)
        gen_result = engine.generate(prompts=conversation, tools=schemas, gen_cfg=_engine_gen_cfg)
        raw_vis = gen_result["completions"][0]
        # Prefer the special-token-preserving completion when the engine
        # provides one (TransformersRolloutEngine does). Some tool-call
        # formats are delimited by tokens the tokenizer treats as "special"
        # (e.g. Mistral's mistral_common backend renders a call's "[" / "]"
        # brackets as special tokens) -- decoding with skip_special_tokens
        # (the plain "completions" text) silently erases exactly the
        # delimiters _extract_tool_calls needs, so every such call would be
        # dropped even though the model generated it correctly.
        raws = gen_result.get("completions_raw") or []
        raw = (raws[0] if raws else raw_vis) or raw_vis
        structured = gen_result.get(
            "completions_structured", [{"role": "assistant", "content": raw_vis}]
        )[0]
        if isinstance(structured, dict) and not structured.get("content"):
            structured = {**structured, "content": raw_vis}
        if isinstance(structured.get("content"), str):
            content = _strip_special_tokens(structured["content"], tokenizer)
            structured = {**structured, "content": content}
        logprobs_raw = gen_result.get("logprobs", [None])[0]
        meta = gen_result.get("metadata", {})

        comp_ids_raw = meta.get("completion_ids")
        flat_ids: list[int] = []
        if comp_ids_raw is not None:
            nested = comp_ids_raw[0] if isinstance(comp_ids_raw[0], list) else comp_ids_raw
            flat_ids = _ids_to_list(nested)

        if logprobs_raw:
            lp_tuples = _ensure_logprob_tuples(_normalise_logprobs(logprobs_raw, flat_ids))
        else:
            lp_tuples = [(0.0, 0)] * len(flat_ids)

        all_completion_ids.extend(flat_ids)
        all_logprobs.extend(lp_tuples)
        all_tool_mask.extend([1] * len(flat_ids))

        return raw, structured, lp_tuples, meta

    # ── Single-turn (no tools) ────────────────────────────────────────────────
    if not tools_enabled:
        conversation = _build_conversation(prompt, system_prompt)
        initial_prompt_ids = _compute_prompt_ids(conversation, [])

        raw_output, _, lp_tuples, meta = _gen(conversation, [])
        raw_output = _strip_harmony_markup(raw_output)

        step = Step(
            step_number=0,
            state=prompt,
            action={},
            observation=raw_output,
            thought=raw_output,
        )
        return Trajectory(
            task=prompt,
            steps=[step],
            final_response=_strip_special_tokens(raw_output, tokenizer),
            logprobs=list(all_logprobs),
            metadata={
                "prompt_ids": initial_prompt_ids,
                "completion_ids": list(all_completion_ids),
                "tool_mask": list(all_tool_mask),
                "tool_call_count": 0,
                "conversation": conversation,
                "answer": gold,
            },
        )

    # ── Multi-turn tool loop ──────────────────────────────────────────────────
    conversation = _build_conversation(prompt, system_prompt)
    fold_tool_results = False
    if (
        tool_schemas
        and tokenizer is not None
        and not _template_renders_tools(tokenizer, tool_schemas)
    ):
        name = getattr(tokenizer, "name_or_path", "this model")
        if tools_fallback_prompt is None:
            raise ValueError(
                f"The chat template of {name} ignores `tools`, so the model never sees the "
                "tool schemas. Use a model whose template supports tools (e.g. Command R7B), "
                "set a tool-capable `tokenizer.chat_template`, or pass `tools_fallback_prompt`."
            )
        warnings.warn(
            f"The chat template of {name} ignores `tools`; adding the tool schemas to the "
            "system prompt and folding tool results into user turns (tools_fallback_prompt).",
            UserWarning,
            stacklevel=2,
        )
        try:
            fallback = tools_fallback_prompt.format(tools=json.dumps(tool_schemas, indent=2))
        except (KeyError, ValueError, IndexError):
            fallback = tools_fallback_prompt.replace("{tools}", json.dumps(tool_schemas, indent=2))
        conversation = list(conversation)  # don't edit the caller's prompt list
        if conversation and conversation[0].get("role") == "system":
            conversation[0] = {
                **conversation[0],
                "content": f"{conversation[0]['content']}\n\n{fallback}",
            }
        else:
            conversation.insert(0, {"role": "system", "content": fallback})
        # Such templates also drop (North) or reject (Aya Expanse) "tool" turns.
        fold_tool_results = True
    initial_prompt_ids = _compute_prompt_ids(conversation, tool_schemas)

    # Effective context cap for the length guard: under the vLLM colocate path
    # the binding constraint is the engine's configured context
    # (vllm_max_model_length), NOT the tokenizer's model_max_length (262k for
    # Qwen3.5) — without this, retrieval-heavy FinDER conversations outgrow the
    # engine mid-trajectory and vLLM raises VLLMValidationError. Leave room for
    # one completion plus a small tokenisation-mismatch buffer.
    _ctx_max_len: int | None = context_length
    if _ctx_max_len is None and trainer is not None and getattr(trainer, "use_vllm", False):
        _vllm_len = getattr(getattr(trainer, "args", None), "vllm_max_model_length", None)
        _compl_budget = getattr(getattr(trainer, "args", None), "max_completion_length", 640) or 640
        if _vllm_len:
            _ctx_max_len = max(int(_vllm_len) - int(_compl_budget) - 64, 512)

    completions_list: list[dict] = []
    steps: list[Step] = []
    tool_call_count = 0
    tool_failure_count = 0
    # FinDER golden-chunk-recall plumbing (E1): chunk ids surfaced by
    # search_corpus results are collected across the whole episode and stored
    # in trajectory metadata, so `_format_for_grpo` can forward them to reward
    # fns as the `retrieved_chunk_ids` kwarg (same path as tool_call_counts).
    # Additive: empty for tool outputs without chunk_id markers.
    retrieved_chunk_ids: list[str] = []
    _seen_chunk_ids: set = set()

    if pre_step_hook:
        pre_step_hook(0, conversation, tools)

    raw_output, completion_msg, _, _ = _gen(conversation, tool_schemas)
    # Strip <think> from content before appending to conversation
    if isinstance(completion_msg.get("content"), str):
        completion_msg = dict(completion_msg)
        completion_msg["content"] = re.sub(
            r"<think>.*?</think>", "", completion_msg["content"], flags=re.DOTALL
        ).strip()

    tool_calls = _extract_tool_calls(completion_msg, raw_output)
    if tool_calls:
        completion_msg = dict(completion_msg)
        completion_msg["tool_calls"] = _assign_tool_call_ids(tool_calls)

    conversation.append(completion_msg)
    completions_list.append(completion_msg)

    # ── Sprint 2: mid-trajectory stall recovery ────────────────────────────────
    # A turn with no parseable tool call AND no <answer> tag is a "stall" — the
    # `while tool_calls` loop below would exit immediately, treating the stall
    # as the trajectory's final response (this is what happened under M1: the
    # model often writes only a <state> block after a tool result and stops,
    # since state-writing "feels" like a complete turn to it). force_final_answer
    # only recovers this at the very END of the trajectory (one shot, answer
    # only) — it can't get the model back into the search loop. This nudge
    # fires INSIDE the loop, at the point of stall, and asks for either a
    # search or an answer, so a stall after turn 1 doesn't strand the
    # trajectory with 5 unused steps of budget. Gated by
    # `force_action_on_stall=True` (opt-in, default off, additive).
    if force_action_on_stall and not tool_calls and "<answer>" not in (raw_output or "").lower():
        if _check_context_length(
            conversation,
            tool_schemas,
            engine,
            tokenizer,
            max_len_override=_ctx_max_len,
            tool_result_format=tool_result_format,
        ):
            nudge_conv = list(conversation) + [
                {
                    "role": "user",
                    "content": _STALL_NUDGE_TEXT,
                }
            ]
            raw_output, completion_msg, _, _ = _gen(nudge_conv, tool_schemas)
            if isinstance(completion_msg.get("content"), str):
                completion_msg = dict(completion_msg)
                completion_msg["content"] = re.sub(
                    r"<think>.*?</think>", "", completion_msg["content"], flags=re.DOTALL
                ).strip()
            tool_calls = _extract_tool_calls(completion_msg, raw_output)
            if tool_calls:
                completion_msg = dict(completion_msg)
                completion_msg["tool_calls"] = _assign_tool_call_ids(tool_calls)
            conversation = nudge_conv + [completion_msg]
            completions_list.append(nudge_conv[-1])
            completions_list.append(completion_msg)

    iteration_num = 0

    while tool_calls and iteration_num < max_steps:

        # ── Execute tool calls ────────────────────────────────────────────────
        calls_to_run = []
        unknown_calls = []

        for tc in tool_calls:
            tool_call_count += 1

            fn_info = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn_info.get("name") or tc.get("name")
            arguments = fn_info.get(
                "arguments",
                fn_info.get("parameters", tc.get("arguments", tc.get("parameters", {}))),
            )

            if not name:
                tool_failure_count += 1
                unknown_calls.append(
                    (tc.get("name", "unknown"), {"error": f"Unsupported call: {tc!r}"})
                )
                continue

            if name not in tool_callable_map:
                tool_failure_count += 1
                unknown_calls.append((name, {"error": f"Tool '{name}' not found"}))
                continue

            calls_to_run.append((name, tool_callable_map[name], arguments))

        tool_results: list[tuple] = unknown_calls.copy()
        if calls_to_run:
            tool_results.extend(_run_tools_parallel(calls_to_run))

        for _, result in tool_results:
            if isinstance(result, dict) and "error" in result:
                tool_failure_count += 1

        # ── Append tool result messages ───────────────────────────────────────
        # for name, result in tool_results:
        #     tool_message = {"role": "tool", "name": name, "content": str(result)}
        #     conversation.append(tool_message)
        #     completions_list.append(tool_message)
        for tc, (name, result) in zip(tool_calls, tool_results, strict=False):
            tool_call_id = tc.get("id") or f"call_{name}"
            tool_message = {
                "role": "tool",
                "name": name,
                "content": str(result),
                "tool_call_id": tool_call_id,  # ← Groq requires this
            }
            conversation.append(tool_message)
            completions_list.append(tool_message)

            # Collect retrieved chunk ids for the golden-chunk-recall reward
            # (search_corpus results carry `[chunk_id=... doc_id=...]` markers).
            # Capture up to the ` doc_id=` marker (NOT whitespace): contract
            # titles contain spaces, so `[^\s\]]+` truncates `...Agreement::66`
            # to `...Agreement`, which can never equal the gold chunk id and
            # silently zeroes golden_chunk_recall.
            for _cid in re.findall(r"chunk_id=(.+?)(?:\s+doc_id=|\])", str(result)):
                if _cid not in _seen_chunk_ids:
                    _seen_chunk_ids.add(_cid)
                    retrieved_chunk_ids.append(_cid)

            if tokenizer is not None:
                tool_ids = _ids_to_list(tokenizer.encode(str(result), add_special_tokens=False))
                all_completion_ids.extend(tool_ids)
                all_logprobs.extend([(0.0, tid) for tid in tool_ids])
                all_tool_mask.extend([0] * len(tool_ids))

        if fold_tool_results:
            conversation = fold_tool_messages_into_user(conversation, tool_result_format)

        # ── Record step ───────────────────────────────────────────────────────
        step = Step(
            step_number=iteration_num,
            state=f"[turn={iteration_num}, conv_len={len(conversation)}]",
            action={"tool_calls": tool_calls},
            observation=str(dict(tool_results)),
            thought=raw_output,
        )
        step.metadata = {
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "tool_call_count": tool_call_count,
            "tool_failure_count": tool_failure_count,
        }
        if post_step_hook:
            # M1 (Sprint 2) core edit: thread `conversation` into the hook so a
            # rewrite hook (rag.memory.m1_rewrite.mem1_post_step_hook) can
            # return a rewritten conversation (MEM1's cur_obs="" wipe).
            # Backward-compatible: old single-arg hooks (TraceLogger) ignore
            # the extra kwarg via **kwargs or raise TypeError, which we catch
            # and retry with the legacy single-arg call. If the hook returns a
            # list, it REPLACES `conversation` from here on.
            try:
                _rewritten = post_step_hook(step, conversation=conversation)
            except TypeError:
                _rewritten = post_step_hook(step)
            if isinstance(_rewritten, list):
                conversation = _rewritten
        steps.append(step)

        iteration_num += 1
        if iteration_num >= max_steps:
            break

        if not _check_context_length(
            conversation,
            tool_schemas,
            engine,
            tokenizer,
            max_len_override=_ctx_max_len,
            tool_result_format=tool_result_format,
        ):
            warnings.warn(
                "Context length exceeded max model length. Stopping tool loop.",
                UserWarning,
            )
            break

        if pre_step_hook:
            pre_step_hook(iteration_num, conversation, tools)

        # Next model turn
        raw_output, completion_msg, _, _ = _gen(conversation, tool_schemas)
        # Strip <think> from content before appending to conversation
        if isinstance(completion_msg.get("content"), str):
            completion_msg = dict(completion_msg)
            completion_msg["content"] = re.sub(
                r"<think>.*?</think>", "", completion_msg["content"], flags=re.DOTALL
            ).strip()

        tool_calls = _extract_tool_calls(completion_msg, raw_output)
        if tool_calls:
            completion_msg = dict(completion_msg)
            completion_msg["tool_calls"] = _assign_tool_call_ids(tool_calls)

        conversation.append(completion_msg)
        completions_list.append(completion_msg)

        # Mid-trajectory stall recovery (see the matching block before the
        # loop for the full rationale) — this is the common case: a stall
        # after a tool result (e.g. M1's rewritten state-only turn), not just
        # the first turn.
        if (
            force_action_on_stall
            and not tool_calls
            and "<answer>" not in (raw_output or "").lower()
            and iteration_num < max_steps
        ):
            if _check_context_length(
                conversation,
                tool_schemas,
                engine,
                tokenizer,
                max_len_override=_ctx_max_len,
                tool_result_format=tool_result_format,
            ):
                nudge_conv = list(conversation) + [
                    {
                        "role": "user",
                        "content": _STALL_NUDGE_TEXT,
                    }
                ]
                raw_output, completion_msg, _, _ = _gen(nudge_conv, tool_schemas)
                if isinstance(completion_msg.get("content"), str):
                    completion_msg = dict(completion_msg)
                    completion_msg["content"] = re.sub(
                        r"<think>.*?</think>", "", completion_msg["content"], flags=re.DOTALL
                    ).strip()
                tool_calls = _extract_tool_calls(completion_msg, raw_output)
                if tool_calls:
                    completion_msg = dict(completion_msg)
                    completion_msg["tool_calls"] = _assign_tool_call_ids(tool_calls)
                conversation = nudge_conv + [completion_msg]
                completions_list.append(nudge_conv[-1])
                completions_list.append(completion_msg)

    # ── FIX: Final generation if last message is a tool result ────────────────
    # Mirrors TRL's _tool_call_loop which always ends with a model generation
    # after all tool results are appended. Without this, final_response would
    # be the tool-call text, not the model's answer.
    _last_role = conversation[-1].get("role") if conversation else None
    _last_is_tool = _last_role == "tool" or (
        fold_tool_results and _last_role == "user" and iteration_num > 0 and bool(tool_results)
    )
    if conversation and _last_is_tool:
        if not _check_context_length(
            conversation,
            tool_schemas,
            engine,
            tokenizer,
            max_len_override=_ctx_max_len,
            tool_result_format=tool_result_format,
        ):
            warnings.warn(
                "Context length exceeded before final generation after tool result.",
                UserWarning,
                stacklevel=2,
            )
        else:
            raw_output, completion_msg, _, _ = _gen(conversation, tool_schemas)
            if isinstance(completion_msg.get("content"), str):
                completion_msg = dict(completion_msg)
                completion_msg["content"] = re.sub(
                    r"<think>.*?</think>", "", completion_msg["content"], flags=re.DOTALL
                ).strip()
            conversation.append(completion_msg)
            completions_list.append(completion_msg)

    # ── Sprint 2: forced-answer fallback (Search-R1 pattern) ──────────────────
    # If the model never emitted an <answer> tag (looped on tool calls or
    # stopped after a state block), append an explicit "now answer" user turn
    # and generate once more. Small models reliably produce answer tags when
    # explicitly told to (verified zero-shot) but don't self-transition from
    # searching to answering. This guarantees answer-tag production so the
    # reward has real variance (answer vs no-answer) for GRPO to learn from.
    # Gated by `force_final_answer=True` (opt-in, default off for back-compat).
    if force_final_answer and conversation:
        final_text = raw_output or ""
        if "<answer>" not in final_text.lower():
            nudge_msg = (
                "You have gathered enough information. Now answer the original "
                "question. Respond with ONLY the answer, wrapped in "
                "<answer></answer> tags. No reasoning, no explanation, no "
                "other text. Example format: <answer>Paris</answer>"
            )
            forced_conv = list(conversation)
            if fold_tool_results and forced_conv[-1].get("role") == "user":
                forced_conv[-1] = {
                    **forced_conv[-1],
                    "content": f"{forced_conv[-1]['content']}\n\n{nudge_msg}",
                }
            else:
                forced_conv.append({"role": "user", "content": nudge_msg})
            if _check_context_length(
                forced_conv,
                tool_schemas,
                engine,
                tokenizer,
                max_len_override=_ctx_max_len,
                tool_result_format=tool_result_format,
            ):
                raw_output, completion_msg, _, _ = _gen(forced_conv, tool_schemas)
                if isinstance(completion_msg.get("content"), str):
                    completion_msg = dict(completion_msg)
                    completion_msg["content"] = re.sub(
                        r"<think>.*?</think>",
                        "",
                        completion_msg["content"],
                        flags=re.DOTALL,
                    ).strip()
                conversation.append(completion_msg)
                completions_list.append(completion_msg)

    # ── Terminal step ─────────────────────────────────────────────────────────
    raw_output = _strip_harmony_markup(raw_output)
    terminal_step = Step(
        step_number=iteration_num,
        state=f"[terminal, conv_len={len(conversation)}]",
        action={},
        observation=raw_output,
        thought=raw_output,
    )
    terminal_step.metadata = {
        "is_terminal": True,
        "tool_call_count": tool_call_count,
        "tool_failure_count": tool_failure_count,
    }
    if post_step_hook:
        # M1 core edit (see note above at the tool-result step call site):
        # thread conversation through, backward-compatible with single-arg hooks.
        # The terminal step's hook return is NOT applied (no next turn to
        # rewrite for), but we still pass conversation so logging hooks can
        # record the final state.
        try:
            post_step_hook(terminal_step, conversation=conversation)
        except TypeError:
            post_step_hook(terminal_step)
    steps.append(terminal_step)

    return Trajectory(
        task=prompt,
        steps=steps,
        final_response=_strip_special_tokens(raw_output, tokenizer),
        logprobs=list(all_logprobs),
        metadata={
            "conversation": conversation,
            "completions_list": completions_list,
            "tool_mask": list(all_tool_mask),
            "tool_call_count": tool_call_count,
            "tool_failure_count": tool_failure_count,
            "prompt_ids": initial_prompt_ids,
            "completion_ids": list(all_completion_ids),
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "answer": gold,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Context length guard
# ─────────────────────────────────────────────────────────────────────────────


def _render_conversation_length(
    conversation: list[dict],
    tokenizer,
    schemas: list[dict] | None,
    enable_thinking: bool = False,
    tool_result_format: str = DEFAULT_TOOL_RESULT_FORMAT,
) -> int:
    """Token length of the applied chat template, measured reliably.

    IMPORTANT: `tokenizer.apply_chat_template(..., tokenize=True)` is BROKEN on
    the pinned transformers (returns ~2 tokens regardless of content — the
    `tokenize=False` string is correct and the template tokenizes to thousands).
    So we render the string then tokenize it directly — this is exactly how the
    rollout measures prompt length, and it is what actually reflects an
    over-window prompt (observed: 20K-token prompt froze vLLM rendering).
    """
    s = _render_chat_template(
        tokenizer,
        conversation,
        schemas,
        enable_thinking=enable_thinking,
        tool_result_format=tool_result_format,
        tokenize=False,
    )
    return len(tokenizer(s).input_ids)


def _truncate_conversation_for_vllm(
    conversation: list[dict],
    tokenizer,
    schemas: list[dict] | None,
    max_prompt_len: int,
    enable_thinking: bool = False,
    tool_result_format: str = DEFAULT_TOOL_RESULT_FORMAT,
) -> None:
    """Drop the OLDEST tool-call/result messages until the applied chat template
    fits `max_prompt_len` tokens. Mutates `conversation` in place.

    Why: an over-window prompt sent to the vLLM server deadlocks its rendering
    step — observed a 20K-token prompt (one tool result injected ~18K tokens)
    freeze "Rendering prompts" at 0% with the trainer stuck in do_poll forever.
    The training loop already truncates the combined prompt+completion to the
    vLLM window after generation, so the effective training context is capped
    regardless; this only keeps the prompt SENT TO vLLM inside the window.

    Drops whole (assistant, tool-result) pairs from the front so the chat
    template stays structurally valid; the system prompt (idx 0) and user
    question (idx 1) are never dropped.
    """
    if tokenizer is None or max_prompt_len is None:
        return
    try:
        while len(conversation) > 3:
            if (
                _render_conversation_length(
                    conversation, tokenizer, schemas, enable_thinking, tool_result_format
                )
                <= max_prompt_len
            ):
                return
            # Drop the oldest non-(system|user) message; if it is an assistant
            # turn whose tool result follows, drop that pair together.
            is_tool_turn = conversation[3].get("role") == "tool" or (
                conversation[3].get("role") == "user"
                and _is_folded_tool_result(conversation[3].get("content", ""), tool_result_format)
            )
            if (
                len(conversation) >= 4
                and conversation[2].get("role") == "assistant"
                and is_tool_turn
            ):
                del conversation[2]
                del conversation[2]
            else:
                del conversation[2]
    except Exception:
        return


def _check_context_length(
    conversation: list[dict],
    tool_schemas: list[dict],
    engine: RolloutEngine | None,
    tokenizer=None,
    max_len_override: int | None = None,
    tool_result_format: str = DEFAULT_TOOL_RESULT_FORMAT,
) -> bool:
    """
    Returns True if conversation is within model's max length.
    Fails open (returns True) if tokenizer is unavailable.

    max_len_override: the vLLM engine's configured max (vllm_max_model_length
    minus completion budget) when running the colocate path — the binding
    constraint there is the engine's KV-cache context, NOT the tokenizer's
    model_max_length (262k for Qwen3.5), so without this override the guard
    never fires and vLLM raises VLLMValidationError mid-trajectory.
    """
    try:
        tok = tokenizer

        if tok is None and engine is not None:
            if _is_vllm_engine(engine):
                pc = engine.vllm_generation.processing_class
                tok = pc.tokenizer if hasattr(pc, "tokenizer") else pc
            elif hasattr(engine, "tokenizer"):
                tok = engine.tokenizer

        if tok is None:
            return True

        _n = _render_conversation_length(
            conversation, tok, tool_schemas, tool_result_format=tool_result_format
        )
        max_len = getattr(tok, "model_max_length", 8192)
        if max_len_override:
            max_len = min(max_len, max_len_override) if max_len else max_len_override
        return _n < max_len

    except Exception:
        return True  # fail open


def create_dpo_rollout_fn(
    reward_fn: Callable,
    rollout_engine=None,
    tools: list | None = None,
    max_steps: int = 20,
    system_prompt: str | None = None,
    num_generations: int = 2,  # must be >= 2 for DPO
    **rollout_kwargs,
) -> Callable:
    """
    Wraps create_rollout_fn to produce DPO-compatible paired outputs.

    Returns a callable with signature:
        fn(prompts: list, trainer=None) -> DPOBatch

    where DPOBatch is a dict with keys:
        prompt_ids       : list[list[int]]   — one entry per prompt
        chosen_ids       : list[list[int]]
        rejected_ids     : list[list[int]]
        chosen_logprobs  : list[list[tuple]]
        rejected_logprobs: list[list[tuple]]
        chosen_mask      : list[list[int]]
        rejected_mask    : list[list[int]]
        responses        : list[dict]        — {"chosen": str, "rejected": str}
    """
    # Build the underlying rollout (generates num_generations completions per prompt)
    _base_rollout = create_rollout_fn(
        rollout_engine=rollout_engine,
        tools=tools,
        max_steps=max_steps,
        reward_fn=reward_fn,
        system_prompt=system_prompt,
        **rollout_kwargs,
    )

    def dpo_rollout_fn(prompts: list[Any], trainer=None) -> dict[str, Any]:
        # Run num_generations rollouts per prompt
        # Duplicate prompts: [p0, p0, p1, p1, ...] for num_generations=2
        expanded_prompts = []
        for p in prompts:
            for _ in range(num_generations):
                expanded_prompts.append(p)

        batch = _base_rollout(expanded_prompts, trainer=trainer)

        # batch["rewards"] is list[float], length = len(prompts) * num_generations
        # batch["completion_ids"], batch["prompt_ids"], etc. — same length
        rewards = batch["rewards"]
        prompt_ids = batch["prompt_ids"]
        comp_ids = batch["completion_ids"]
        logprobs = batch["logprobs"]
        responses = batch["responses"]  # list[str]

        n = len(prompts)
        chosen_ids, rejected_ids = [], []
        chosen_logprobs, rejected_logprobs = [], []
        chosen_mask, rejected_mask = [], []
        chosen_responses, rejected_responses = [], []
        out_prompt_ids = []

        skipped = 0
        for i in range(n):
            # Indices into the expanded batch for this prompt
            indices = list(range(i * num_generations, (i + 1) * num_generations))
            best_idx = max(indices, key=lambda j: rewards[j])
            worst_idx = min(indices, key=lambda j: rewards[j])
            if best_idx == worst_idx or rewards[best_idx] == rewards[worst_idx]:
                skipped += 1
                continue

            out_prompt_ids.append(prompt_ids[best_idx])  # same prompt for both

            chosen_ids.append(comp_ids[best_idx])
            rejected_ids.append(comp_ids[worst_idx])

            chosen_logprobs.append(logprobs[best_idx])
            rejected_logprobs.append(logprobs[worst_idx])

            # mask: 1 for real tokens
            chosen_mask.append([1] * len(comp_ids[best_idx]))
            rejected_mask.append([1] * len(comp_ids[worst_idx]))

            chosen_responses.append(responses[best_idx])
            rejected_responses.append(responses[worst_idx])

        return {
            "prompt_ids": out_prompt_ids,
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "chosen_logprobs": chosen_logprobs,
            "rejected_logprobs": rejected_logprobs,
            "chosen_mask": chosen_mask,
            "rejected_mask": rejected_mask,
            "responses": [
                {"chosen": c, "rejected": r}
                for c, r in zip(chosen_responses, rejected_responses, strict=False)
            ],
            # pass through for reward logging
            "rewards": rewards,
            "trajectories": batch.get("trajectories", []),
            "skipped_ties": skipped,
        }

    return dpo_rollout_fn
