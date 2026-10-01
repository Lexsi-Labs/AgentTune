import pytest

"""Test rules stage in isolation."""

import asyncio
import json

from agenttune.decide.stages.rules import RulesStage
from agenttune.decide.state import PipelineState


@pytest.mark.asyncio
async def test_rules_stage():
    """Test RulesStage independently."""
    config = {
        "id": "test_rules",
        "type": "rules",
        "rules": [
            {
                "condition": "score > 5",
                "on_failure": {
                    "goto": "low_quality",
                    "inject": "Score too low",
                },
            },
        ],
    }

    stage = RulesStage(config)
    state = PipelineState(
        pipeline_id="test_003",
        template_id="test",
        template_version="1.0.0",
        input_text="test",
        input_hash="test_hash",
    )
    # Add some stage outputs for rules to evaluate
    state.stage_outputs = {
        "s0": {"score": 8, "verdict": "PASS"},
    }

    print("Testing RulesStage (passing condition)...")
    result = await stage.execute(state, stage_config=config)
    print(f"Result: {json.dumps(result, indent=2)}")
    assert result.get("goto") is None, "Expected no routing on passing rule"
    print("✓ RulesStage passed test")

    # Test failing condition
    state.stage_outputs = {"s0": {"score": 3, "verdict": "FAIL"}}
    print("\nTesting RulesStage (failing condition)...")
    result = await stage.execute(state, stage_config=config)
    print(f"Result: {json.dumps(result, indent=2)}")
    assert result.get("goto") == "low_quality", "Expected routing on failing rule"
    print("✓ RulesStage failure test passed")


if __name__ == "__main__":
    asyncio.run(test_rules_stage())
