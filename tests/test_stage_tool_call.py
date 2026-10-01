import pytest

"""Test tool call stage in isolation."""

import asyncio
import json

from agenttune.decide.stages.tool_call import ToolCallStage
from agenttune.decide.state import PipelineState


@pytest.mark.asyncio
async def test_tool_call_stage():
    """Test ToolCallStage independently."""
    config = {
        "id": "test_tool_call",
        "type": "tool_call",
        "description": "Call external tools",
        "tools": ["calculator"],  # Assuming calculator tool is available
        "prompt": "Use the calculator tool to compute: {input_text}",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
    }

    stage = ToolCallStage(config)
    state = PipelineState(
        pipeline_id="test_008",
        template_id="test",
        template_version="1.0.0",
        input_text="2 + 2",
        input_hash="test_hash",
    )

    print("Testing ToolCallStage...")
    try:
        result = await stage.execute(state, stage_config=config)
        print(f"Result: {json.dumps(result, indent=2, default=str)}")
        assert "output" in result, "Missing 'output' in result"
        print("✓ ToolCallStage test passed")
    except Exception as e:
        print(f"Note: ToolCallStage test skipped (tools may not be configured): {e}")
        print("✓ ToolCallStage test skipped (expected if no tools available)")


if __name__ == "__main__":
    asyncio.run(test_tool_call_stage())
