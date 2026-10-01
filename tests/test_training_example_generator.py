from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agenttune.decide.closed_loop.contracts import ClassifiedFailure, Failure, TrainingExample
from agenttune.decide.closed_loop.replay_validator import ReplayValidator
from agenttune.decide.closed_loop.training_example_generator import (
    TrainingExampleGenerator,
    _parse_tool_call,
)


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

    mock_resp = _mock_response("Use the calculator tool instead of search.")

    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
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

    with patch("litellm.acompletion", new=AsyncMock()):
        results = await generator.generate_batch([classified])

    assert results == [], "hallucinated_output should be skipped by the generator"


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
