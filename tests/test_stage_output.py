import pytest

"""Test output stage in isolation."""

import asyncio
import json

from agenttune.decide.stages.output import OutputStage
from agenttune.decide.state import PipelineState


@pytest.mark.asyncio
async def test_output_stage():
    """Test OutputStage independently."""
    config = {
        "id": "test_output",
        "type": "output",
        "description": "Output the final result",
        "verdict_field": "s0.output.verdict",
        "confidence_field": "s0.output.confidence",
        "destinations": ["file"],
    }

    stage = OutputStage(config)
    state = PipelineState(
        pipeline_id="test_007",
        template_id="test",
        template_version="1.0.0",
        input_text="test",
        input_hash="test_hash",
    )
    # Add prior stage outputs (stored without the "output" wrapper key)
    state.stage_outputs = {
        "s0": {
            "verdict": "PASS",
            "confidence": 9,
            "result": "This is the final result",
        }
    }

    print("Testing OutputStage...")
    result = await stage.execute(state, stage_config=config)
    print(f"Result: {json.dumps(result, indent=2, default=str)}")

    # Verify verdict and confidence extraction (stage returns {"output": {...}})
    assert "output" in result, "Missing 'output' key in result"
    assert "verdict" in result["output"], "Missing 'verdict' in output"
    assert "confidence" in result["output"], "Missing 'confidence' in output"
    assert result["output"]["verdict"] == "PASS", "Expected verdict PASS"
    assert result["output"]["confidence"] == 9, "Expected confidence 9"
    print("✓ OutputStage test passed")


if __name__ == "__main__":
    asyncio.run(test_output_stage())
