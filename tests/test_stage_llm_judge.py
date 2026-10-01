"""Test LLM judge stage in isolation."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agenttune.decide.stages.llm_judge import LLMJudgeStage
from agenttune.decide.state import PipelineState


def _mock_llm_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(total_tokens=50)
    return resp


@patch("agenttune.decide.stages.llm_judge.LLMJUDGE_AVAILABLE", False)
@patch("agenttune.decide.stages.base.litellm")
@pytest.mark.asyncio
async def test_llm_judge_stage(mock_litellm):
    """Test LLMJudgeStage independently (mocked LLM)."""
    mock_litellm.acompletion = AsyncMock(
        return_value=_mock_llm_response(json.dumps({"score": 8, "verdict": "PASS"}))
    )

    config = {
        "id": "test_judge",
        "type": "llm_judge",
        "prompt": "Rate this text (0-10): {input_text}.",
        "model": "claude-haiku-4-5",
        "output_schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
                "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
            },
            "required": ["score", "verdict"],
        },
    }

    stage = LLMJudgeStage(config)
    state = PipelineState(
        pipeline_id="test_002",
        template_id="test",
        template_version="1.0.0",
        input_text="This is a good quality text that should be rated highly.",
        input_hash="test_hash",
    )

    print("Testing LLMJudgeStage...")
    result = await stage.execute(state, stage_config=config)
    print(f"Result: {json.dumps(result, indent=2, default=str)}")

    assert "output" in result, "Missing 'output' in result"
    assert "score" in result["output"], "Missing 'score' in output"
    assert "verdict" in result["output"], "Missing 'verdict' in output"
    print("✓ LLMJudgeStage test passed")


if __name__ == "__main__":
    asyncio.run(test_llm_judge_stage())
