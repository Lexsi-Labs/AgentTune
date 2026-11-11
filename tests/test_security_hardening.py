"""Regression tests for the security-baseline fixes."""

import json
import os
import time

from agenttune.agentic.rollout_engines.rollout_factory import _GLM_CALL_RE, _extract_tool_calls


def test_glm_call_regex_still_matches_known_shapes():
    newline = "<tool_call>f\n<arg_key>a</arg_key>\n<arg_value>1</arg_value></tool_call>"
    compact = "<tool_call>f<arg_key>a</arg_key><arg_value>x\ny</arg_value></tool_call>"
    assert _GLM_CALL_RE.findall(newline)[0][0] == "f"
    assert _GLM_CALL_RE.findall(compact)[0][0] == "f"
    calls = _extract_tool_calls(
        None,
        "<tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value>"
        "<arg_key>b</arg_key><arg_value>two</arg_value></tool_call>",
    )
    assert calls[0]["function"]["name"] == "f"
    assert json.loads(calls[0]["function"]["arguments"]) == {"a": 1, "b": "two"}


def test_glm_call_regex_adversarial_input_is_fast():
    starts = "<tool_call>!<arg_key></arg_key><arg_value>"
    adversarial = starts + "</arg_value><arg_key></arg_key><arg_value>" * 5000
    t0 = time.perf_counter()
    assert _GLM_CALL_RE.findall(adversarial) == []
    _extract_tool_calls(None, adversarial)
    assert time.perf_counter() - t0 < 2.0


def test_sqlite_backend_temp_file_is_private():
    from agenttune.rag.trainer import _build_backend

    path = _build_backend("sqlite_fts", None, False).db_path
    try:
        assert os.stat(path).st_mode & 0o077 == 0
    finally:
        os.unlink(path)


def test_deepseek_detected_by_host_not_substring():
    import pytest

    pytest.importorskip("openai")
    from agenttune.rag.synthesis.llm_client import OpenAICompatLLMClient

    def is_ds(url):
        return OpenAICompatLLMClient("m", "k", url)._is_deepseek

    assert is_ds("https://api.deepseek.com/v1")
    assert not is_ds("https://evil.example/?x=deepseek.com")
    assert not is_ds("https://notdeepseek.com/v1")
