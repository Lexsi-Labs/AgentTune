"""Tool-call extraction across template families — not one model dialect."""

import json

from agenttune.agentic.rollout_engines.rollout_factory import _extract_tool_calls
from agenttune.agentic.rollout_engines.tool_call_parse import (
    extract_tool_calls,
    extract_tool_calls_from_text,
    parse_xml_function_calls,
)


def _names(calls):
    return [c["function"]["name"] for c in calls]


def _args(calls, i=0):
    return json.loads(calls[i]["function"]["arguments"])


def test_json_tool_call_tag():
    text = '<tool_call>{"name": "lookup_order", "arguments": {"order_id": "ORD-1"}}</tool_call>'
    calls = extract_tool_calls_from_text(text)
    assert _names(calls) == ["lookup_order"]
    assert _args(calls) == {"order_id": "ORD-1"}


def test_xml_closed_tags():
    text = (
        "<tool_call>\n<function=search_corpus>\n"
        "<parameter=query>\nAlan Turing\n</parameter>\n</function>\n</tool_call>"
    )
    calls = parse_xml_function_calls(text)
    assert _names(calls) == ["search_corpus"]
    assert _args(calls) == {"query": "Alan Turing"}


def test_xml_unclosed_official_template():
    """Several chat templates never emit </function> / </parameter>."""
    text = (
        "<tool_call>\n<function=lookup_order>\n" "<parameter=order_id>\nORD-4101\n \n</tool_call>"
    )
    calls = _extract_tool_calls(None, text)
    assert calls is not None
    assert _names(calls) == ["lookup_order"]
    assert _args(calls) == {"order_id": "ORD-4101"}


def test_xml_two_params_unclosed():
    text = (
        "<tool_call>\n<function=create_ticket>\n"
        "<parameter=title>\nLate delivery\n"
        "<parameter=body>\nWhere is my order\n"
        "</tool_call>"
    )
    calls = extract_tool_calls_from_text(text)
    assert _names(calls) == ["create_ticket"]
    assert _args(calls)["title"] == "Late delivery"
    assert "order" in _args(calls)["body"]


def test_llama_function_json_body():
    text = '<function=get_weather>{"city": "Paris", "unit": "c"}</function>'
    calls = extract_tool_calls_from_text(text)
    assert _names(calls) == ["get_weather"]
    assert _args(calls) == {"city": "Paris", "unit": "c"}


def test_mistral_tool_calls():
    text = '[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Paris"}}]'
    calls = extract_tool_calls_from_text(text)
    assert _names(calls) == ["get_weather"]
    assert _args(calls) == {"city": "Paris"}


def test_glm_arg_key():
    text = (
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>\n"
        "<arg_value>Beijing</arg_value>\n"
        "</tool_call>"
    )
    calls = extract_tool_calls_from_text(text)
    assert _names(calls) == ["get_weather"]
    assert _args(calls) == {"city": "Beijing"}


def test_react_action_input():
    text = 'Action: lookup_order\nAction Input: {"order_id": "ORD-9"}'
    calls = extract_tool_calls_from_text(text)
    assert _names(calls) == ["lookup_order"]
    assert _args(calls) == {"order_id": "ORD-9"}


def test_cohere_start_action():
    """Command R7B/A and North Mini Code: <|START_ACTION|>[...]<|END_ACTION|>."""
    text = (
        "<|START_THINKING|>I should look this up.<|END_THINKING|>"
        "<|START_ACTION|>[\n"
        '    {"tool_call_id": "0", "tool_name": "get_weather", "parameters": {"city": "Paris"}}\n'
        "]<|END_ACTION|>"
    )
    calls = extract_tool_calls_from_text(text)
    assert _names(calls) == ["get_weather"]
    assert _args(calls) == {"city": "Paris"}


def test_cohere_start_action_parallel_calls():
    text = (
        "<|START_ACTION|>[\n"
        '    {"tool_call_id": "0", "tool_name": "get_weather", "parameters": {"city": "Paris"}},\n'
        '    {"tool_call_id": "1", "tool_name": "get_time", "parameters": {"city": "Paris"}}\n'
        "]<|END_ACTION|>"
    )
    calls = extract_tool_calls_from_text(text)
    assert _names(calls) == ["get_weather", "get_time"]


def test_cohere_legacy_action_fence():
    """Classic Command R / Aya Expanse: Action: ```json [...] ```."""
    text = (
        "Action:\n```json\n"
        '[{"tool_name": "internet_search", "parameters": {"query": "penguins"}}]\n'
        "```"
    )
    calls = extract_tool_calls_from_text(text)
    assert _names(calls) == ["internet_search"]
    assert _args(calls) == {"query": "penguins"}


def test_harmony_tool_call():
    """gpt-oss Harmony format: <|channel|>commentary to=functions.NAME..."""
    text = (
        "<|channel|>analysis<|message|>I should check the weather.<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.get_weather"
        '<|constrain|>json<|message|>{"location": "Paris"}<|call|>'
    )
    calls = extract_tool_calls_from_text(text)
    assert _names(calls) == ["get_weather"]
    assert _args(calls) == {"location": "Paris"}


def test_harmony_tool_call_special_tokens_stripped():
    """Same call, but with skip_special_tokens=True style decoding -- the
    <|...|> markers are gone but the plain-text `to=functions.NAME` and the
    JSON body survive."""
    text = 'commentary to=functions.get_weatherjson{"location": "Tokyo"}'
    calls = extract_tool_calls_from_text(text)
    assert _names(calls) == ["get_weather"]
    assert _args(calls) == {"location": "Tokyo"}


def test_harmony_parallel_tool_calls():
    text = (
        "<|start|>assistant<|channel|>commentary to=functions.get_weather"
        '<|constrain|>json<|message|>{"location": "Paris"}<|call|>'
        "<|start|>assistant<|channel|>commentary to=functions.get_time"
        '<|constrain|>json<|message|>{"location": "Paris"}<|call|>'
    )
    calls = extract_tool_calls_from_text(text)
    assert _names(calls) == ["get_weather", "get_time"]


def test_plain_text_is_not_a_call():
    assert extract_tool_calls_from_text("I don't know the answer, let me think.") is None
    assert extract_tool_calls_from_text("<answer>Paris</answer>") is None
    assert extract_tool_calls_from_text("the value was {approximately 5} maybe") is None
    assert extract_tool_calls_from_text('{"ok": true}') is None


def test_structured_message_tool_calls_are_normalised():
    msg = {
        "role": "assistant",
        "tool_calls": [{"name": "lookup_order", "arguments": {"order_id": "x"}}],
    }
    calls = extract_tool_calls(msg, "")
    assert calls[0]["type"] == "function"
    assert _names(calls) == ["lookup_order"]


def test_rollout_factory_export_parses_xml_without_patch():
    text = "<function=ping>\n<parameter=x>\n1\n</parameter>\n</function>"
    assert _extract_tool_calls(None, text) is not None


# ── Cohere cases ported from rollout_factory (one parser for both paths) ─────


def test_cohere_markers_stripped_by_decode():
    """Command R7B's action list after skip_special_tokens=True: "plan[...]"."""
    text = (
        'I will look it up.[\n    {"tool_call_id": "0", "tool_name": "get_weather", '
        '"parameters": {"city": "Paris"}}\n]'
    )
    for calls in (extract_tool_calls_from_text(text), _extract_tool_calls(None, text)):
        assert _names(calls) == ["get_weather"]
        assert _args(calls) == {"city": "Paris"}


def test_cohere_bare_object():
    text = '{"tool_name": "get_weather", "parameters": {"city": "Oslo"}}'
    for calls in (extract_tool_calls_from_text(text), _extract_tool_calls(None, text)):
        assert _names(calls) == ["get_weather"]
        assert _args(calls) == {"city": "Oslo"}


def test_cohere_single_tool_names():
    text = '[{"tool_names": ["get_weather"], "parameters": {"city": "Rome"}}]'
    calls = _extract_tool_calls(None, text)
    assert _names(calls) == ["get_weather"]
    assert _args(calls) == {"city": "Rome"}
    ambiguous = '[{"tool_names": ["a", "b"], "parameters": {}}]'
    assert _extract_tool_calls(None, ambiguous) is None


def test_cohere_directly_answer_is_not_a_call():
    only = 'Action: ```json\n[{"tool_name": "directly-answer", "parameters": {}}]\n```'
    assert extract_tool_calls_from_text(only) is None
    assert _extract_tool_calls(None, only) is None
    mixed = (
        '<|START_ACTION|>[{"tool_name": "directly-answer", "parameters": {}}, '
        '{"tool_name": "get_time", "parameters": {"tz": "UTC"}}]<|END_ACTION|>'
    )
    assert _names(_extract_tool_calls(None, mixed)) == ["get_time"]


def test_json_list_with_non_dicts_is_not_a_call():
    for text in ("[2, 3]", '[{"name": "add", "arguments": {}}, 3]', '[{"tool_name": "a"}, 1]'):
        assert extract_tool_calls_from_text(text) is None
        assert _extract_tool_calls(None, text) is None


def test_function_block_with_malformed_json_body_is_skipped():
    assert extract_tool_calls_from_text("<function=broken>{not json}</function>") is None
    calls = extract_tool_calls_from_text('<function=f>{"a": 1} trailing</function>')
    assert _names(calls) == ["f"]
    assert _args(calls) == {"a": 1}
