import pytest

"""Test parallel stage in isolation."""

import asyncio
import json

from agenttune.decide.stages.parallel import ParallelStage
from agenttune.decide.state import PipelineState


@pytest.mark.asyncio
async def test_parallel_stage():
    """Test ParallelStage independently."""
    config = {
        "id": "test_parallel",
        "type": "parallel",
        "branches": [
            {
                "id": "branch_1",
                "type": "llm_call",
                "prompt": 'Analyze aspect 1 of: {input_text}. Return JSON: {"analysis": "..."}',
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "output_schema": {
                    "type": "object",
                    "properties": {"analysis": {"type": "string"}},
                    "required": ["analysis"],
                },
            },
            {
                "id": "branch_2",
                "type": "llm_call",
                "prompt": 'Analyze aspect 2 of: {input_text}. Return JSON: {"analysis": "..."}',
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "output_schema": {
                    "type": "object",
                    "properties": {"analysis": {"type": "string"}},
                    "required": ["analysis"],
                },
            },
        ],
    }

    stage = ParallelStage(config)
    state = PipelineState(
        pipeline_id="test_006",
        template_id="test",
        template_version="1.0.0",
        input_text="Analyze this topic from multiple angles.",
        input_hash="test_hash",
    )

    print("Testing ParallelStage...")
    result = await stage.execute(state, stage_config=config)
    print(f"Result: {json.dumps(result, indent=2, default=str)}")

    # Verify structure - parallel stage returns outputs keyed by branch id
    assert "output" in result or "branch_1" in result, "Missing parallel branch outputs"
    print("✓ ParallelStage test passed")


if __name__ == "__main__":
    asyncio.run(test_parallel_stage())
