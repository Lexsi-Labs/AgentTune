from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agenttune.decide.closed_loop.contracts import ClassifiedFailure, Failure
from agenttune.decide.closed_loop.failure_classifier import FailureClassifier


@pytest.fixture
def classifier():
    return FailureClassifier(model_name="gpt-4o-mini")


def _mock_response(json_str: str) -> MagicMock:
    msg = MagicMock()
    msg.content = json_str
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.asyncio
async def test_failure_classifier_success(classifier):
    failure = Failure(
        trajectory_id="traj-123",
        failure_type="wrong_tool",
        failed_stage_name="tool_call",
        context={"tool_used": "search", "tool_expected": "calculator"},
        error_message="Wrong tool selected",
        judge_score=0.2,
    )

    mock_resp = _mock_response(
        '{"root_cause": "wrong_tool", "confidence": 0.92, "analysis": "Agent used search instead of calculator."}'
    )

    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
        results = await classifier.classify_batch([failure])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, ClassifiedFailure)
    assert result.root_cause == "wrong_tool"
    assert result.failure is failure
    assert result.confidence == pytest.approx(0.92)
    assert result.analysis != ""


@pytest.mark.asyncio
async def test_failure_classifier_fallback(classifier):
    failure = Failure(
        trajectory_id="traj-456",
        failure_type="loop_collapse",
        failed_stage_name="tool_call",
        context={"loop_count": 7},
        judge_score=0.1,
    )

    with patch("litellm.acompletion", new=AsyncMock(side_effect=Exception("API completely down"))):
        results = await classifier.classify_batch([failure])

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, ClassifiedFailure)
    assert result.root_cause == "incomplete_reasoning"
    assert result.confidence == pytest.approx(0.0)
    assert result.failure is failure
