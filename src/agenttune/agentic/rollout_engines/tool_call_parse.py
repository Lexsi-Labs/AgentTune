"""Extract tool calls from model text, regardless of chat-template family.

The rollout loop only runs a call if this module returns a normalised
OpenAI-style list. Models emit many surface formats (JSON tags, XML
function/parameter blocks, Mistral [TOOL_CALLS], GLM arg_key, ReAct,
fenced JSON, Cohere START_ACTION, OpenAI Harmony channels). Missing any of
those used to look like "the model never called a tool."
"""

from __future__ import annotations

import json
import re
from typing import Any

# Reasoning/thinking wrappers stripped before parsing: Qwen-style <think>,
# Cohere's <|START_THINKING|> (Command R7B/A, North Mini Code, Aya), and
# gpt-oss's Harmony "analysis" channel.
_THINK_BLOCK_RE = re.compile(
    r"<think>.*?</think>"
    r"|<\|START_THINKING\|>.*?<\|END_THINKING\|>"
    r"|<\|channel\|>analysis<\|message\|>.*?<\|end\|>",
    flags=re.DOTALL | re.IGNORECASE,
)


# Cohere's "no tool needed" pseudo-call (Command R / Aya Expanse); not a call.
_DIRECTLY_ANSWER = {"directly-answer", "directly_answer"}


def _as_calls(items: Any) -> list[dict] | None:
    if items is None:
        return None
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return None
    out: list[dict] = []
    for call in items:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else None
        if fn:
            name = fn.get("name") or call.get("name")
            args = fn.get("arguments", fn.get("parameters", fn.get("args", {})))
        else:
            name = (
                call.get("name")
                or call.get("tool")
                or call.get("tool_name")
                or call.get("function_name")
            )
            # Small models sometimes write {"tool_names": ["NAME"], ...}. One
            # name is unambiguous; anything else is not a call.
            names = call.get("tool_names")
            if name is None and isinstance(names, list) and len(names) == 1:
                name = names[0]
            args = call.get(
                "arguments",
                call.get("parameters", call.get("args", call.get("input", {}))),
            )
        if not isinstance(name, str) or not name.strip() or name.strip() in _DIRECTLY_ANSWER:
            continue
        if args is None:
            args_s = "{}"
        elif isinstance(args, dict | list):
            args_s = json.dumps(args)
        elif isinstance(args, str):
            args_s = args
        else:
            args_s = json.dumps(args)
        out.append(
            {
                "type": "function",
                "function": {"name": name.strip(), "arguments": args_s},
            }
        )
        if call.get("id"):
            out[-1]["id"] = call["id"]
    return out or None


def _looks_like_tool_dict(d: dict, *, require_args: bool) -> bool:
    if not isinstance(d, dict):
        return False
    if d.get("type") == "function" and isinstance(d.get("function"), dict):
        return bool(d["function"].get("name"))
    name = d.get("name") or d.get("tool") or d.get("tool_name")
    if not isinstance(name, str):
        return False
    if "arguments" in d or "parameters" in d or "args" in d or "input" in d:
        return True
    return (not require_args) and "name" in d


def _json_objects(text: str) -> list[Any]:
    dec = json.JSONDecoder()
    found: list[Any] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        found.append(obj)
        i = max(end, i + 1)
    return found


def _calls_from_json_blob(text: str, *, require_args: bool) -> list[dict] | None:
    dicts: list[dict] = []
    for obj in _json_objects(text):
        # A list with anything but dicts in it (e.g. an answer like [2, 3])
        # is data, not a batch of calls.
        if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
            dicts.extend(obj)
        elif isinstance(obj, dict):
            dicts.append(obj)
    usable = [d for d in dicts if _looks_like_tool_dict(d, require_args=require_args)]
    return _as_calls(usable)


def _coerce_param(val: str) -> Any:
    val = val.strip()
    if len(val) >= 2 and ((val[0] == "{" and val[-1] == "}") or (val[0] == "[" and val[-1] == "]")):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            pass
    return val


def _params_from_body(body: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for m in re.finditer(
        r"<parameter\s*=\s*([^>\s/]+)\s*>\s*(.*?)(?="
        r"\s*<parameter\s*=|\s*</parameter>|\s*</function>|\s*</tool_call>|\s*</toolcall>|\s*$)",
        body,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        args[m.group(1).strip()] = _coerce_param(m.group(2))
    for m in re.finditer(
        r"<parameter\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>\s*(.*?)\s*</parameter>",
        body,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        args[m.group(1).strip()] = _coerce_param(m.group(2))
    for m in re.finditer(
        r"<arg\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>\s*(.*?)\s*</arg>",
        body,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        args[m.group(1).strip()] = _coerce_param(m.group(2))
    keys = re.findall(r"<arg_key>\s*(.*?)\s*</arg_key>", body, flags=re.DOTALL | re.IGNORECASE)
    vals = re.findall(r"<arg_value>\s*(.*?)\s*</arg_value>", body, flags=re.DOTALL | re.IGNORECASE)
    for k, v in zip(keys, vals, strict=False):
        args[k.strip()] = _coerce_param(v)
    return args


def parse_xml_function_calls(text: str) -> list[dict] | None:
    """`<function=name>` blocks, with or without closing tags / JSON bodies."""
    if not isinstance(text, str):
        text = str(text)
    lower = text.lower()
    if "<function" not in lower and "<invoke" not in lower:
        return None
    calls: list[dict] = []
    for m in re.finditer(
        r"<function\s*=\s*([^>\s/]+)\s*>",
        text,
        flags=re.IGNORECASE,
    ):
        name = m.group(1).strip().strip("\"'")
        rest = text[m.end() :]
        end_m = re.search(
            r"</function>|<function\s*=|</tool_call>|</toolcall>",
            rest,
            flags=re.IGNORECASE,
        )
        body = rest[: end_m.start()] if end_m else rest
        args: dict[str, Any] = {}
        if re.search(r"<parameter\s*=|<parameter\s+name=|<arg_key>|<arg\s+name=", body, re.I):
            args = _params_from_body(body)
        else:
            blob = _calls_from_json_blob(body, require_args=False)
            if blob and len(blob) == 1 and blob[0]["function"]["name"] != name:
                # Body is `{name, arguments}` rather than a raw args object.
                calls.extend(blob)
                continue
            js = None
            stripped = body.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    js = json.JSONDecoder().raw_decode(stripped)[0]
                except json.JSONDecodeError:
                    # A JSON-looking body that isn't JSON: skip the call rather
                    # than run it with no arguments.
                    continue
            if isinstance(js, dict) and not _looks_like_tool_dict(js, require_args=True):
                args = js
            elif isinstance(js, dict):
                more = _as_calls(js)
                if more:
                    calls.extend(more)
                    continue
        calls.append(
            {
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
    for m in re.finditer(
        r"<function\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>\s*(.*?)\s*</function>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        name = m.group(1).strip()
        body = m.group(2)
        args = _params_from_body(body)
        if not args:
            try:
                js = json.loads(body.strip()) if body.strip().startswith("{") else None
            except json.JSONDecodeError:
                js = None
            if isinstance(js, dict):
                args = js
        calls.append(
            {
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
    for m in re.finditer(
        r"<invoke\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>\s*(.*?)\s*</invoke>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        calls.append(
            {
                "type": "function",
                "function": {
                    "name": m.group(1).strip(),
                    "arguments": json.dumps(_params_from_body(m.group(2))),
                },
            }
        )
    return calls or None


def _parse_tool_call_tagged(text: str) -> list[dict] | None:
    calls: list[dict] = []
    blocks = re.findall(
        r"<tool_?call\b([^>]*)>(.*?)</tool_?call>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    dangling = re.findall(
        r"<tool_?call\b([^>]*)>(.*)$",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not blocks and dangling:
        blocks = dangling
    for attrs, body in blocks:
        name_attr = re.search(r"\bname\s*=\s*[\"']([^\"']+)[\"']", attrs or "", re.I)
        xml = parse_xml_function_calls(f"<tool_call>{body}</tool_call>")
        if xml:
            calls.extend(xml)
            continue
        js = _calls_from_json_blob(body, require_args=False)
        if js:
            calls.extend(js)
            continue
        glm_name = re.match(r"\s*([A-Za-z_][\w\.\-]*)\s*(?:<|\n|$)", body)
        glm_args = _params_from_body(body)
        if name_attr:
            calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": name_attr.group(1).strip(),
                        "arguments": json.dumps(glm_args),
                    },
                }
            )
        elif glm_name and (glm_args or "<arg_key>" in body.lower()):
            calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": glm_name.group(1),
                        "arguments": json.dumps(glm_args),
                    },
                }
            )
    return calls or None


def _parse_mistral(text: str) -> list[dict] | None:
    m = re.search(r"\[TOOL_CALLS\]\s*", text)
    if not m:
        return None
    return _calls_from_json_blob(text[m.end() :], require_args=False)


def _parse_harmony_tool_calls(text: str) -> list[dict] | None:
    """OpenAI gpt-oss "Harmony" format.

    Raw shape: `<|channel|>commentary to=functions.NAME<|constrain|>json
    <|message|>{...}<|call|>`. Keyed off the literal `to=functions.NAME`
    text rather than the surrounding `<|...|>` tokens, since that text is
    ordinary decoded content and survives even when the engine strips
    special tokens (the tags themselves don't).
    """
    dec = json.JSONDecoder()
    calls: list[dict] = []
    for m in re.finditer(r"\bto=functions?\.([A-Za-z_][\w\.\-]*)", text):
        name = m.group(1)
        rest = text[m.end() :]
        brace = re.search(r"[\{\[]", rest)
        if not brace:
            continue
        junk = rest[: brace.start()]
        if not junk.strip() and name.endswith("json") and len(name) > 4:
            # `<|constrain|>json<|message|>` with tokens stripped leaves a
            # bare "json" fused onto the name with no separator.
            name = name[:-4]
        try:
            args, _ = dec.raw_decode(rest, brace.start())
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict):
            continue
        calls.append(
            {
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
    return calls or None


def _parse_cohere_action(text: str) -> list[dict] | None:
    """Cohere's `<|START_ACTION|>[...]<|END_ACTION|>` block.

    Used by Command R7B/A and North Mini Code. Calls are a JSON array of
    `{"tool_call_id", "tool_name", "parameters"}` dicts; `_as_calls` already
    understands the `tool_name`/`parameters` key names.
    """
    m = re.search(r"<\|START_ACTION\|>\s*", text, flags=re.IGNORECASE)
    if not m:
        return None
    rest = text[m.end() :]
    end_m = re.search(r"<\|END_ACTION\|>", rest, flags=re.IGNORECASE)
    body = rest[: end_m.start()] if end_m else rest
    return _calls_from_json_blob(body, require_args=False)


def _parse_cohere_json(text: str) -> list[dict] | None:
    """Cohere-style calls without their markers.

    Command R7B's action list once decoding drops the special tokens
    (``plan[{"tool_call_id": ..., "tool_name": ...}]``), a bare
    ``{"tool_name": ..., "parameters": ...}`` object, and the list the
    rollout's ``tools_fallback_prompt`` asks tool-less templates (Tiny Aya)
    for. Only a list made entirely of such dicts counts, so an answer like
    ``[2, 3]`` is not a call.
    """
    if "tool_name" not in text:
        return None
    for obj in _json_objects(_THINK_BLOCK_RE.sub("", text)):
        items = obj if isinstance(obj, list) else [obj]
        if items and all(
            isinstance(c, dict) and ("tool_name" in c or "tool_names" in c) for c in items
        ):
            return _as_calls(items)
    return None


def _parse_cohere_legacy_action(text: str) -> list[dict] | None:
    """Classic Command R / Aya Expanse `Action:` fenced-JSON block.

    Renders as `Action:\\n\\`\\`\\`json\\n[{"tool_name": ..., "parameters": ...}]\\n\\`\\`\\``
    with no `Action Input:` line, so it never matches `_parse_react`.
    """
    m = re.search(
        r"Action\s*:\s*```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    return _calls_from_json_blob(m.group(1), require_args=False)


def _parse_react(text: str) -> list[dict] | None:
    calls: list[dict] = []
    for m in re.finditer(
        r"Action\s*:\s*([A-Za-z_][\w\.\-]*)\s*\nAction\s*Input\s*:\s*(\{.*?\})",
        text,
        flags=re.DOTALL,
    ):
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {"input": m.group(2).strip()}
        if not isinstance(args, dict):
            args = {"input": args}
        calls.append(
            {
                "type": "function",
                "function": {"name": m.group(1), "arguments": json.dumps(args)},
            }
        )
    return calls or None


def _parse_fenced_after_name(text: str) -> list[dict] | None:
    calls: list[dict] = []
    for m in re.finditer(
        r"(?:tool[_ ]?call|function)\s*(?:[:\|]|sep)?\s*([A-Za-z_][\w\.\-]*)\s*```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict):
            continue
        calls.append(
            {
                "type": "function",
                "function": {"name": m.group(1), "arguments": json.dumps(args)},
            }
        )
    return calls or None


def extract_tool_calls_from_text(raw_text: str) -> list[dict] | None:
    if not raw_text or not str(raw_text).strip():
        return None
    text = str(raw_text)
    text = _THINK_BLOCK_RE.sub("", text)
    text = text.strip()
    if not text:
        return None

    for parser in (
        parse_xml_function_calls,
        _parse_tool_call_tagged,
        _parse_cohere_action,
        _parse_cohere_json,
        _parse_harmony_tool_calls,
        _parse_mistral,
        _parse_react,
        _parse_cohere_legacy_action,
        _parse_fenced_after_name,
    ):
        got = parser(text)
        if got:
            return got

    tagged = bool(re.search(r"tool_?call|<\|start_action\|>|to=functions?\.", text, re.I))
    return _calls_from_json_blob(text, require_args=not tagged)


def extract_tool_calls(completion_msg: Any, raw_text: str) -> list[dict] | None:
    texts: list[str] = []
    msg = completion_msg if isinstance(completion_msg, dict) else {}
    tc = msg.get("tool_calls") if msg else None
    if tc:
        normalised = _as_calls(tc if isinstance(tc, list) else [tc])
        if normalised:
            return normalised
    if isinstance(msg.get("content"), str):
        content = _THINK_BLOCK_RE.sub("", msg["content"]).strip()
        if content:
            texts.append(content)
    if raw_text:
        stripped = _THINK_BLOCK_RE.sub("", str(raw_text)).strip()
        if stripped and stripped not in texts:
            texts.append(stripped)

    for t in texts:
        got = extract_tool_calls_from_text(t)
        if got:
            return got
    return None
