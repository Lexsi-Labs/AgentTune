import pytest

"""Test router stage in isolation."""

import asyncio
import json

from agenttune.decide.stages.router import RouterStage
from agenttune.decide.state import PipelineState


@pytest.mark.asyncio
async def test_router_stage_llm_based():
    """Test RouterStage with LLM-based routing."""
    config = {
        "id": "test_router",
        "type": "router",
        "prompt": 'Based on the input, classify as positive or negative. Respond with JSON: {"classification": "positive" | "negative"}',
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "output_schema": {
            "type": "object",
            "properties": {
                "classification": {"type": "string"},
            },
            "required": ["classification"],
        },
        "on_result": [
            {
                "condition": "classification == 'positive'",
                "goto": "positive_path",
            },
            {
                "condition": "classification == 'negative'",
                "goto": "negative_path",
            },
        ],
        "default": "unknown",
    }

    stage = RouterStage(config)
    state = PipelineState(
        pipeline_id="test_004",
        template_id="test",
        template_version="1.0.0",
        input_text="This is wonderful!",
        input_hash="test_hash",
    )

    print("Testing RouterStage (LLM-based)...")
    result = await stage.execute(state, stage_config=config)
    print(f"Result: {json.dumps(result, indent=2, default=str)}")

    # Verify structure
    assert "output" in result, "Missing 'output' in result"
    assert "goto" in result, "Missing 'goto' in result"
    print("✓ RouterStage LLM-based test passed")


@pytest.mark.asyncio
async def test_router_stage_rules_based():
    """Test RouterStage with rules-based routing."""
    config = {
        "id": "test_rules_router",
        "type": "router",
        "on_result": [
            {
                "condition": "s0.score > 7",
                "goto": "high_quality",
            },
            {
                "condition": "s0.score <= 7",
                "goto": "low_quality",
            },
        ],
        "default": "unknown",
    }

    stage = RouterStage(config)
    state = PipelineState(
        pipeline_id="test_005",
        template_id="test",
        template_version="1.0.0",
        input_text="test",
        input_hash="test_hash",
    )
    state.stage_outputs = {"s0": {"score": 8}}

    print("\nTesting RouterStage (rules-based)...")
    result = await stage.execute(state, stage_config=config)
    print(f"Result: {json.dumps(result, indent=2, default=str)}")
    assert result.get("goto") == "high_quality", "Expected high_quality routing"
    print("✓ RouterStage rules-based test passed")


if __name__ == "__main__":
    asyncio.run(test_router_stage_llm_based())
    asyncio.run(test_router_stage_rules_based())
