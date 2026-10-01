"""
Unit tests for M1 rewrite logic (pure-Python, no GPU, no model).

Tests extract_internal_state, compress_tool_output, and the rewrite hooks
against hand-built conversations/steps.
"""

from agenttune.agentic.trajectory.dataset import Step
from agenttune.rag.memory.m1_rewrite import (
    compress_tool_output,
    extract_internal_state,
    mem1_post_step_hook,
    recent_k_post_step_hook,
)

# ── extract_internal_state ───────────────────────────────────────────────────


def test_extract_state_block():
    resp = "some preamble <state>running_summary: X\nopen_questions: Y</state> more"
    assert extract_internal_state(resp, tag="state") == "running_summary: X\nopen_questions: Y"


def test_extract_think_block_fallback():
    # No <state>, but a think block present (Qwen native thinking tag).
    # Built via concatenation to avoid the literal tag tokens in source.
    open_tag = "<" + "think" + ">"
    close_tag = "<" + "/think" + ">"
    resp = f"preamble {open_tag}thinking here{close_tag} rest"
    assert extract_internal_state(resp, tag="think") == "thinking here"


def test_extract_no_block_returns_none():
    assert extract_internal_state("just plain text", tag="state") is None


def test_extract_non_string_input():
    assert extract_internal_state(123, tag="state") is None or isinstance(
        extract_internal_state(123, tag="state"), str
    )


# ── compress_tool_output ─────────────────────────────────────────────────────


def test_compress_preserves_chunk_ids():
    tool_text = (
        "[chunk_id=c1 doc_id=d1 score=0.900]\n"
        "Alan Turing attended King's College Cambridge. He was a brilliant student.\n\n"
        "[chunk_id=c2 doc_id=d2 score=0.800]\n"
        "King's College was founded in 1441 by Henry VI.\n"
    )
    out = compress_tool_output(tool_text)
    assert "chunk_id=c1" in out
    assert "chunk_id=c2" in out
    assert "doc_id=d1" in out
    # First sentence only, not the whole chunk
    assert "brilliant student" not in out


def test_compress_no_results():
    assert compress_tool_output("No results found.") == "No results found."


def test_compress_caps_chunks():
    # 10 chunks, max_chunks=3 → only 3 evidence lines
    chunks = "\n\n".join(
        f"[chunk_id=c{i} doc_id=d{i} score=0.{900-i}]\nChunk {i} content here." for i in range(10)
    )
    out = compress_tool_output(chunks, max_chunks=3)
    assert out.count("evidence[") == 3


def test_compress_passthrough_no_headers():
    # read_document output has no chunk headers → pass through with cap
    out = compress_tool_output("Just a plain document body with no headers.", max_chunks=5)
    assert "plain document body" in out


# ── mem1_post_step_hook ──────────────────────────────────────────────────────


def _make_step(thought, tool_results, is_terminal=False):
    s = Step(step_number=0, state="x", action={}, observation="obs", thought=thought)
    s.metadata = {"tool_results": tool_results, "is_terminal": is_terminal}
    return s


def test_mem1_hook_rewrites_when_state_present():
    thought = "<state>running_summary: found Turing's college\nopen_questions: when founded</state>"
    tool_results = [
        ("search_corpus", "[chunk_id=c1 doc_id=d1 score=0.9]\nKing's College founded 1441.")
    ]
    step = _make_step(thought, tool_results)
    conv = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "When was Turing's college founded?"},
        {"role": "assistant", "content": "old stuff"},
        {"role": "tool", "content": "old result"},
    ]
    new_conv = mem1_post_step_hook(step, conversation=conv)
    assert new_conv is not None
    # Rewritten conv: system + user + state(assistant) + tool(compressed)
    assert len(new_conv) == 4
    assert new_conv[0]["role"] == "system"
    assert new_conv[1]["role"] == "user"
    assert new_conv[2]["role"] == "assistant"
    assert "<state>" in new_conv[2]["content"]
    assert new_conv[3]["role"] == "tool"
    assert "chunk_id=c1" in new_conv[3]["content"]  # compressed, source IDs kept


def test_mem1_hook_no_state_returns_none():
    # Model didn't emit a <state> block → don't rewrite (safe zero-shot fallback)
    step = _make_step("just a tool call, no state", [("search_corpus", "result")])
    conv = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    assert mem1_post_step_hook(step, conversation=conv) is None


def test_mem1_hook_terminal_returns_none():
    step = _make_step("<state>x</state>", [], is_terminal=True)
    conv = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    assert mem1_post_step_hook(step, conversation=conv) is None


def test_mem1_hook_no_conversation_returns_none():
    step = _make_step("<state>x</state>", [("search_corpus", "r")])
    assert mem1_post_step_hook(step, conversation=None) is None


def test_mem1_hook_backward_compat_single_arg():
    # Old-style hook call (no conversation kwarg) must not crash — the hook
    # accepts **kwargs and returns None when conversation is None.
    step = _make_step("<state>x</state>", [("search_corpus", "r")])
    # Called without conversation= → defaults to None → returns None
    assert mem1_post_step_hook(step) is None


# ── recent_k_post_step_hook ──────────────────────────────────────────────────


def test_recent_k_truncates_to_k_pairs():
    step = _make_step("call", [("search_corpus", "r3")])
    conv = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "call1"},
        {"role": "tool", "content": "r1"},
        {"role": "assistant", "content": "call2"},
        {"role": "tool", "content": "r2"},
        {"role": "assistant", "content": "call3"},
        {"role": "tool", "content": "r3"},
    ]
    new_conv = recent_k_post_step_hook(step, conversation=conv, k=1)
    assert new_conv is not None
    # system + user + 1 (assistant, tool) pair = 4 messages
    assert len(new_conv) == 4
    assert new_conv[2]["content"] == "call3"
    assert new_conv[3]["content"] == "r3"


def test_recent_k_k2_keeps_two_pairs():
    step = _make_step("call", [("search_corpus", "r3")])
    conv = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "call1"},
        {"role": "tool", "content": "r1"},
        {"role": "assistant", "content": "call2"},
        {"role": "tool", "content": "r2"},
        {"role": "assistant", "content": "call3"},
        {"role": "tool", "content": "r3"},
    ]
    new_conv = recent_k_post_step_hook(step, conversation=conv, k=2)
    assert new_conv is not None
    # system + user + 2 pairs = 6 messages
    assert len(new_conv) == 6
