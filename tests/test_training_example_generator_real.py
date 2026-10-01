from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _real_backends import real_litellm_response

from agenttune.decide.closed_loop.contracts import ClassifiedFailure, Failure, TrainingExample
from agenttune.decide.closed_loop.replay_validator import ReplayValidator
from agenttune.decide.closed_loop.training_example_generator import (
    TrainingExampleGenerator,
    _parse_tool_call,
)

# Loads Qwen2.5-0.5B via _real_backends (3-5GB RSS + ~1GB download);
# not for the 7.8GB CPU CI runner. Runs under -m qwen_e2e.
pytestmark = pytest.mark.qwen_e2e


async def _real_acompletion(**kwargs):
    """Drop-in for litellm.acompletion backed by a real local HF model instead
    of a canned mock response. `generator.generate_batch` calls litellm with
    kwargs={"model": ..., "messages": [{"role": "user", "content": prompt}], ...}
    """
    prompt = kwargs["messages"][-1]["content"]
    return real_litellm_response(prompt, max_new_tokens=48)


@pytest.fixture
def validator():
    v = MagicMock(spec=ReplayValidator)
    v.validate_example_with_trace = AsyncMock(return_value=(True, ["ok"]))
    return v


@pytest.fixture
def generator(validator):
    gen = TrainingExampleGenerator(validator=validator, model_name="gpt-4o-mini")
    gen.evaluator._calculate_tac = MagicMock(return_value=0.8)
    gen.evaluator._calculate_ter = MagicMock(return_value=0.6)
    return gen


def _mock_response(text: str) -> MagicMock:
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.asyncio
async def test_generate_example_success(generator):
    failure = Failure(
        trajectory_id="traj-001",
        failure_type="wrong_tool",
        failed_stage_name="tool_call",
        context={"tool_used": "search", "tool_expected": "calculator"},
        judge_score=0.2,
    )
    classified = ClassifiedFailure(
        failure=failure,
        root_cause="wrong_tool",
        confidence=0.9,
        analysis="Used search instead of calculator.",
    )

    # REAL model call: HuggingFaceTB/SmolLM2-135M-Instruct generates the
    # completion instead of a canned mock string (see _real_backends.py).
    with patch("litellm.acompletion", new=AsyncMock(side_effect=_real_acompletion)):
        results = await generator.generate_batch([classified])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, TrainingExample)
    assert result.trajectory_id == "traj-001"
    assert result.original_failure_type == "wrong_tool"
    assert result.root_cause == "wrong_tool"
    assert result.completions
    assert result.rewards


@pytest.mark.asyncio
async def test_generate_skips_unrecognized_root_cause(generator):
    failure = Failure(
        trajectory_id="traj-002",
        failure_type="hallucinated_output",
        failed_stage_name="llm_call",
        context={"claim": "Berlin is the capital of France"},
        judge_score=0.0,
    )
    classified = ClassifiedFailure(
        failure=failure,
        root_cause="hallucinated_output",
        confidence=0.8,
        analysis="Hallucinated a geography fact.",
    )

    # Real acompletion wired in too, to prove the skip happens before any model
    # call is made — if this test ever regressed into calling the model, it
    # would just be slow, not silently pass on an unused mock.
    with patch("litellm.acompletion", new=AsyncMock(side_effect=_real_acompletion)) as mocked:
        results = await generator.generate_batch([classified])

    assert results == [], "hallucinated_output should be skipped by the generator"
    mocked.assert_not_called()


# ---------------------------------------------------------------------------
# _parse_tool_call — the load-bearing pure parser behind the TAC reward's
# schema-scoring path. CPU-only; the LLM example (self_heal_reward_real.py)
# exercises it end-to-end, but these lock in the shapes it must accept so the
# fix can't silently revert to the lenient raw-string fallback.
# ---------------------------------------------------------------------------
class TestParseToolCall:
    def test_canonical_name_and_arguments(self):
        call = _parse_tool_call('{"name": "lookup_order", "arguments": {"order_id": 42}}')
        assert call == {"name": "lookup_order", "arguments": {"order_id": 42}}

    @pytest.mark.parametrize("name_key", ["name", "tool", "tool_name", "action"])
    def test_alternate_name_keys(self, name_key):
        call = _parse_tool_call(f'{{"{name_key}": "lookup_order", "arguments": {{"order_id": 1}}}}')
        assert call["name"] == "lookup_order"
        assert call["arguments"] == {"order_id": 1}

    @pytest.mark.parametrize("arg_key", ["arguments", "args", "parameters", "params"])
    def test_alternate_arg_keys(self, arg_key):
        call = _parse_tool_call(f'{{"name": "lookup_order", "{arg_key}": {{"order_id": 7}}}}')
        assert call["arguments"] == {"order_id": 7}

    def test_function_as_string_with_implicit_args(self):
        # {"function": "lookup_order", "order_id": 42} — name is a string, args are
        # the remaining top-level keys. This is the shape the live LLM emitted.
        call = _parse_tool_call('{"function": "lookup_order", "order_id": 42}')
        assert call == {"name": "lookup_order", "arguments": {"order_id": 42}}

    def test_openai_style_nested_function(self):
        call = _parse_tool_call(
            '{"function": {"name": "lookup_order", "arguments": {"order_id": 5}}}'
        )
        assert call == {"name": "lookup_order", "arguments": {"order_id": 5}}

    def test_non_json_returns_none(self):
        assert _parse_tool_call("I think you should call lookup_order(42)") is None

    def test_json_non_object_returns_none(self):
        assert _parse_tool_call("[1, 2, 3]") is None
        assert _parse_tool_call('"just a string"') is None

    def test_object_without_name_returns_none(self):
        # No name key and no string-valued function → falls back to raw string.
        assert _parse_tool_call('{"order_id": 42}') is None
