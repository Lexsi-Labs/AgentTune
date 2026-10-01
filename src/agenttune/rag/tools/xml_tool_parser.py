"""XML-style tool-call parser used by the RAG scripts.

The actual formats live in `rollout_engines.tool_call_parse` so every eval and
inference path sees the same extractor — not a Qwen-only monkeypatch.
"""

from typing import Any

from agenttune.agentic.rollout_engines.tool_call_parse import parse_xml_function_calls


def parse_xml_tool_calls(text: str):
    return parse_xml_function_calls(text)


def patch_xml_tool_call_parser():
    """Idempotent wrap of `_extract_tool_calls`. Kept so existing RAG scripts
    that call this at import time keep working. The core extractor already
    handles XML; the wrap is a no-op fallback.
    """
    from agenttune.agentic.rollout_engines import rollout_factory

    if getattr(rollout_factory._extract_tool_calls, "_xml_patched", False):
        return

    original = rollout_factory._extract_tool_calls

    def _patched(completion_msg: Any, raw_text: str):
        result = original(completion_msg, raw_text)
        if result:
            return result
        return parse_xml_tool_calls(raw_text)

    _patched._xml_patched = True
    _patched._original = original
    rollout_factory._extract_tool_calls = _patched
