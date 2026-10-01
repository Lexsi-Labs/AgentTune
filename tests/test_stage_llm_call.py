import pytest

"""Test LLM call stage in isolation."""

import asyncio
import json

from agenttune.decide.stages.llm_call import LLMCallStage
from agenttune.decide.state import PipelineState


@pytest.mark.asyncio
async def test_llm_call_stage():
    """Test LLMCallStage independently."""
    config = {
        "id": "test_llm_call",
        "type": "llm_call",
        "prompt": 'Say hello and respond with JSON: {"greeting": "hello", "status": "PASS"}',
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "output_schema": {
            "type": "object",
            "properties": {
                "greeting": {"type": "string"},
                "status": {"type": "string"},
            },
            "required": ["greeting"],
        },
    }

    stage = LLMCallStage(config)
    state = PipelineState(
        pipeline_id="test_001",
        template_id="test",
        template_version="1.0.0",
        input_text="test input",
        input_hash="test_hash",
    )

    print("Testing LLMCallStage...")
    result = await stage.execute(state, stage_config=config)
    print(f"Result: {json.dumps(result, indent=2)}")

    # Verify structure
    assert "output" in result, "Missing 'output' in result"
    assert "latency_ms" in result, "Missing 'latency_ms' in result"
    print("✓ LLMCallStage test passed")


if __name__ == "__main__":
    asyncio.run(test_llm_call_stage())
