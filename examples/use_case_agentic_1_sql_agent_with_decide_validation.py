"""
USE CASE AGENTIC-1: SQL Query Agent with DECIDE Validation

Combines agentic training with DECIDE for:
- LLM generates SQL queries for database operations
- DECIDE validates the SQL syntax and correctness
- Rewards agent based on validation results + query correctness

This use case demonstrates a feedback loop where:
1. Agent generates SQL query based on natural language
2. DECIDE validates query syntax, safety, and relevance
3. Agent receives structured feedback for training
"""

import asyncio

from agenttune.decide.graph_runner import GraphRunner


async def create_sql_validation_pipeline():
    """
    Create a DECIDE pipeline for SQL validation.
    Uses the text_classify template and adapts it for SQL validation.
    """
    try:
        # Load the existing text_classify template
        runner = GraphRunner.from_template("generic/text_classify")

        # Customize for SQL validation
        for stage in runner.config.get("stages", []):
            if stage["id"] == "process":
                stage[
                    "prompt"
                ] = """Validate this SQL query and give a verdict.

SQL Query: {input_text}

Check:
1. Does it have SELECT clause?
2. Is syntax valid?
3. Is it safe (no injection risks)?

Return JSON: {{"is_valid": true/false, "issues": "list of issues", "verdict": "VALID/INVALID"}}"""
                stage["output_schema"] = {
                    "type": "object",
                    "properties": {
                        "is_valid": {"type": "boolean"},
                        "issues": {"type": "string"},
                        "verdict": {"type": "string"},
                    },
                }

        return runner
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        raise


def extract_reward_from_output(stage_outputs: dict) -> float:
    """
    Convert DECIDE output to agent reward.

    Valid SQL → +1.0
    Invalid SQL → +0.0
    """
    if "process" in stage_outputs:
        output = stage_outputs["process"]
        if isinstance(output, dict):
            is_valid = output.get("is_valid", False)
            return 1.0 if is_valid else 0.0
    return 0.0


async def main():
    print("\n" + "=" * 70)
    print("USE CASE AGENTIC-1: SQL Agent with DECIDE Validation")
    print("=" * 70)

    # Create the DECIDE pipeline
    try:
        runner = await create_sql_validation_pipeline()
        print("✅ DECIDE validation pipeline created")
        print(f"   Model: {runner.config.get('default_model')}")
    except Exception as e:
        print(f"❌ Failed to create pipeline: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Simulate agent outputs (what the agent would generate)
    agent_outputs = [
        "SELECT * FROM users WHERE age > 18;",
        "SELECT name, email FROM customers;",
        "SELECT COUNT(*) FROM orders WHERE status = 'completed';",
        "SELECT age FROM users",
    ]

    print(f"\n[Step 1] Validating {len(agent_outputs)} agent SQL queries...")

    total_reward = 0
    passed = 0

    for i, agent_output in enumerate(agent_outputs, 1):
        try:
            print(f"\n  Test {i}: {agent_output[:50]}...")

            # Run DECIDE validation
            state = await runner.run(agent_output)

            # Extract reward from stage outputs
            reward = extract_reward_from_output(state.stage_outputs)
            total_reward += reward

            print(f"    ✅ Completed in {state.elapsed_seconds:.2f}s")
            print(f"    📊 Reward: {reward:.2f}")

            if reward > 0.5:
                passed += 1

            # Show validation details
            if "process" in state.stage_outputs:
                output = state.stage_outputs["process"]
                if isinstance(output, dict):
                    print(f"    Valid: {output.get('is_valid', False)}")
                    print(f"    Issues: {output.get('issues', 'None')}")

        except Exception as e:
            print(f"    ❌ Error: {e}")

    print("\n" + "=" * 70)
    print("✅ Results:")
    print(f"   Passed: {passed}/{len(agent_outputs)}")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Avg Reward: {total_reward/len(agent_outputs):.2f}")
    print("=" * 70)

    return len(agent_outputs) > 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
