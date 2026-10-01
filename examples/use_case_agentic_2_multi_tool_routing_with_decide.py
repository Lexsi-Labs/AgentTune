"""
USE CASE AGENTIC-2: Multi-Tool Agent with DECIDE Tool Router

Combines agentic training with DECIDE for:
- Agent receives task but doesn't know which tool to use
- DECIDE analyzes task and recommends appropriate tool
- Uses templates to simulate tool evaluation
- Reward based on tool selection quality

This demonstrates intelligent tool routing where:
1. Agent receives task
2. DECIDE analyzes requirements
3. DECIDE recommends best tool(s)
4. Agent learns task-to-tool mapping through rewards
"""

import asyncio

from agenttune.decide.graph_runner import GraphRunner


async def create_tool_router_pipeline():
    """
    Create a DECIDE pipeline that routes tasks to appropriate tools.
    Uses the text_classify template adapted for tool selection.
    """
    try:
        # Load the existing text_classify template
        runner = GraphRunner.from_template("generic/text_classify")

        # Customize for tool routing
        for stage in runner.config.get("stages", []):
            if stage["id"] == "process":
                stage[
                    "prompt"
                ] = """Analyze this task and recommend which tool to use.

Task: {input_text}

Available tools:
- Database: For queries, lookups, data extraction
- API: For external service calls, integrations
- Computation: For calculations, analytics, processing

Return JSON: {{"recommended_tool": "database/api/computation", "confidence": "high/medium/low", "reasoning": "why"}}"""
                stage["output_schema"] = {
                    "type": "object",
                    "properties": {
                        "recommended_tool": {"type": "string"},
                        "confidence": {"type": "string"},
                        "reasoning": {"type": "string"},
                    },
                }

        return runner
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        raise


def extract_tool_reward(stage_outputs: dict) -> float:
    """
    Score tool selection decision based on confidence.

    High confidence → +1.0 (clear decision)
    Medium confidence → +0.6 (less certain)
    Low confidence → +0.3 (uncertain)
    """
    if "process" in stage_outputs:
        output = stage_outputs["process"]
        if isinstance(output, dict):
            confidence = output.get("confidence", "low").lower()
            rewards = {"high": 1.0, "medium": 0.6, "low": 0.3}
            return rewards.get(confidence, 0.0)
    return 0.0


async def main():
    print("\n" + "=" * 70)
    print("USE CASE AGENTIC-2: Multi-Tool Agent with DECIDE Router")
    print("=" * 70)

    # Create the DECIDE pipeline
    try:
        runner = await create_tool_router_pipeline()
        print("✅ DECIDE tool router pipeline created")
        print(f"   Model: {runner.config.get('default_model')}")
    except Exception as e:
        print(f"❌ Failed to create pipeline: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Simulate tasks that might need different tools
    tasks = [
        "Find all customers with orders over $1000",
        "Check the status of order #12345 via REST API",
        "Calculate total revenue for Q1 2024",
        "Get user profile information from database",
    ]

    print(f"\n[Step 1] Routing {len(tasks)} tasks to appropriate tools...")

    total_reward = 0
    high_confidence = 0

    for i, task in enumerate(tasks, 1):
        try:
            print(f"\n  Test {i}: {task[:50]}...")

            # Run DECIDE router
            state = await runner.run(task)

            # Extract tool selection details
            if "process" in state.stage_outputs:
                selection = state.stage_outputs["process"]
                if isinstance(selection, dict):
                    tool = selection.get("recommended_tool", "unknown")
                    selection.get("reasoning", "N/A")
                    confidence = selection.get("confidence", "low")
                    print(f"    🎯 Tool: {tool}")
                    print(f"    💡 Confidence: {confidence}")

                    # Calculate reward
                    reward = extract_tool_reward(state.stage_outputs)
                    total_reward += reward
                    print(f"    📊 Reward: {reward:.2f}")

                    if confidence == "high":
                        high_confidence += 1

        except Exception as e:
            print(f"    ❌ Error: {e}")

    print("\n" + "=" * 70)
    print("✅ Results:")
    print(f"   High Confidence: {high_confidence}/{len(tasks)}")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Avg Reward: {total_reward/len(tasks):.2f}")
    print("=" * 70)

    return len(tasks) > 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
