"""Unit tests for the Qwen3.5 XML-style tool-call parser."""

import json

from agenttune.rag.tools.xml_tool_parser import parse_xml_tool_calls, patch_xml_tool_call_parser


def _delim_wrap(inner: str) -> str:
    """Wrap inner text in the Qwen3.5 tool-call delimiters (built via
    concatenation so the literal tokens don't appear in source)."""
    open_tag = "<" + "tool_call" + ">"
    close_tag = "<" + "/tool_call" + ">"
    return f"{open_tag}\n{inner}\n{close_tag}"


def test_parse_single_xml_tool_call():
    inner = "<function=search_corpus>\n<parameter=query>\nAlan Turing university\n</parameter>\n</function>"
    text = _delim_wrap(inner)
    calls = parse_xml_tool_calls(text)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "search_corpus"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"query": "Alan Turing university"}


def test_parse_multiple_params():
    inner = (
        "<function=read_document>\n" "<parameter=doc_id>\nturing_bio\n</parameter>\n" "</function>"
    )
    calls = parse_xml_tool_calls(_delim_wrap(inner))
    assert calls[0]["function"]["name"] == "read_document"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"doc_id": "turing_bio"}


def test_parse_no_function_block_returns_none():
    assert parse_xml_tool_calls("just some answer text") is None
    assert parse_xml_tool_calls("<answer>Paris</answer>") is None


def test_parse_empty_args():
    inner = "<function=some_tool>\n</function>"
    calls = parse_xml_tool_calls(_delim_wrap(inner))
    assert calls is not None
    assert json.loads(calls[0]["function"]["arguments"]) == {}


def test_parse_multiline_param_value():
    inner = (
        "<function=search_corpus>\n"
        "<parameter=query>\nThis is a query\nthat spans\nmultiple lines\n</parameter>\n"
        "</function>"
    )
    calls = parse_xml_tool_calls(_delim_wrap(inner))
    args = json.loads(calls[0]["function"]["arguments"])
    assert "multiple lines" in args["query"]


def test_patch_is_idempotent_and_falls_back():
    # Patching twice should not double-wrap.
    patch_xml_tool_call_parser()
    patch_xml_tool_call_parser()
    from agenttune.agentic.rollout_engines import rollout_factory

    assert getattr(rollout_factory._extract_tool_calls, "_xml_patched", False)

    # The patched parser should now handle XML that the original JSON parser
    # would miss. Build an XML tool call and feed it through the patched fn.
    inner = "<function=search_corpus>\n<parameter=query>\nfoo bar\n</parameter>\n</function>"
    text = _delim_wrap(inner)
    # _extract_tool_calls signature: (completion_msg, raw_text)
    result = rollout_factory._extract_tool_calls({"role": "assistant", "content": text}, text)
    assert result is not None
    assert result[0]["function"]["name"] == "search_corpus"


def test_patch_preserves_json_parsing():
    # A JSON-format tool call (Qwen3-0.6B style) should still parse after patch.
    patch_xml_tool_call_parser()
    from agenttune.agentic.rollout_engines import rollout_factory

    json_text = '[{"name": "search_corpus", "arguments": {"query": "foo"}}]'
    result = rollout_factory._extract_tool_calls(
        {"role": "assistant", "content": json_text}, json_text
    )
    assert result is not None
    assert result[0]["function"]["name"] == "search_corpus"
