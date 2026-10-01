"""
USE CASE AGENTIC-3: Agent Training with DECIDE-Based Reward Function

Combines agentic training with DECIDE for:
- Agent generates response/completion
- DECIDE pipeline evaluates response quality
- Scores converted to RL training rewards
- Demonstrates integration with RL training loops

This demonstrates training loop integration where:
1. Agent generates completion
2. DECIDE scores: sentiment (positive/negative/neutral)
3. Reward converted from confidence score
4. Agent learns to generate better responses

Key insight: DECIDE pipelines become reward models for RL training!
"""

import asyncio

from agenttune.decide.graph_runner import GraphRunner


async def create_reward_evaluation_pipeline():
    """
    Create a DECIDE pipeline that evaluates agent responses.
    Uses sentiment_analysis template adapted for response evaluation.
    Returns scores for RL training.
    """
    try:
        # Load the sentiment analysis template
        runner = GraphRunner.from_template("generic/sentiment_analysis")

        # Customize for agent response evaluation
        for stage in runner.config.get("stages", []):
            if stage["id"] == "process":
                stage[
                    "prompt"
                ] = """Evaluate the quality of this agent response.

Response: {input_text}

Analyze:
- Relevance to question
- Accuracy and truthfulness
- Clarity and coherence
- Helpfulness

Return JSON: {{"quality": "positive/neutral/negative", "confidence": "high/medium/low", "issues": "any problems"}}"""
                stage["output_schema"] = {
                    "type": "object",
                    "properties": {
                        "quality": {"type": "string"},
                        "confidence": {"type": "string"},
                        "issues": {"type": "string"},
                    },
                }

        return runner
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        raise


def extract_rl_reward(stage_outputs: dict) -> float:
    """
    Extract the RL reward from DECIDE evaluation.
    Converts confidence + quality to a 0.0-1.0 reward signal.
    This reward is fed back to the RL trainer.
    """
    if "process" in stage_outputs:
        output = stage_outputs["process"]
        if isinstance(output, dict):
            quality = output.get("quality", "negative").lower()
            confidence = output.get("confidence", "low").lower()

            # Base reward from quality
            quality_reward = {"positive": 1.0, "neutral": 0.5, "negative": 0.0}.get(quality, 0.0)

            # Confidence multiplier
            confidence_mult = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(confidence, 0.4)

            return quality_reward * confidence_mult

    return 0.0


async def simulate_agent_training_step(runner, agent_output: str, gold_answer: str = None):
    """
    Simulate one training step:
    1. Agent generates response
    2. DECIDE evaluates
    3. Return reward for training
    """
    try:
        # Run evaluation pipeline
        state = await runner.run(agent_output)

        # Extract reward
        reward = extract_rl_reward(state)

        return {
            "verdict": state.verdict,
            "reward": reward,
            "stage_outputs": state.stage_outputs,
            "success": True,
        }
    except Exception as e:
        return {"verdict": "ERROR", "reward": 0.0, "error": str(e), "success": False}


async def main():
    print("\n" + "=" * 70)
    print("USE CASE AGENTIC-3: Agent Training with DECIDE Rewards")
    print("=" * 70)

    # Create the DECIDE reward pipeline
    try:
        runner = await create_reward_evaluation_pipeline()
        print("✅ DECIDE reward evaluation pipeline created")
        print(f"   Model: {runner.config.get('default_model')}")
    except Exception as e:
        print(f"❌ Failed to create pipeline: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Simulate agent responses (from different training steps)
    agent_responses = [
        "The capital of France is Paris. It's known for the Eiffel Tower and is a major cultural center.",
        "France: Capital is Paris. Population ~67M. Famous for culture, food, and monuments.",
        "The capital of Germany is Paris. Wait, that's wrong. France's capital is Paris.",
        "Where is Paris? It's in France. Paris is the capital of France and a beautiful city.",
    ]

    print(f"\n[Step 1] Evaluating {len(agent_responses)} agent responses...")

    total_reward = 0
    training_log = []

    for step, response in enumerate(agent_responses, 1):
        try:
            print(f"\n  Step {step}: {response[:50]}...")

            # Run DECIDE evaluation
            state = await runner.run(response)

            # Extract RL reward
            reward = extract_rl_reward(state.stage_outputs)
            total_reward += reward

            print("    ✅ Evaluated")
            print(f"    📊 RL Reward: {reward:.3f}")

            # Show evaluation details
            if "process" in state.stage_outputs:
                output = state.stage_outputs["process"]
                if isinstance(output, dict):
                    quality = output.get("quality", "unknown")
                    confidence = output.get("confidence", "unknown")
                    issues = output.get("issues", "none")
                    print(f"    Quality: {quality} | Confidence: {confidence}")
                    if issues and issues != "none":
                        print(f"    Issues: {issues}")

            training_log.append(
                {
                    "step": step,
                    "reward": reward,
                    "quality": (
                        state.stage_outputs.get("process", {}).get("quality", "unknown")
                        if isinstance(state.stage_outputs.get("process"), dict)
                        else "unknown"
                    ),
                }
            )

        except Exception as e:
            print(f"    ❌ Error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ Training Summary:")
    print(f"   Steps Evaluated: {len(training_log)}/{len(agent_responses)}")
    print(f"   Total Reward: {total_reward:.3f}")
    print(f"   Avg Reward/Step: {total_reward/len(agent_responses):.3f}")
    if training_log:
        print(
            f"   Quality Trajectory: {' → '.join(r['quality'][:3].upper() for r in training_log)}"
        )
    print("\n📝 Next: Feed these rewards to RL trainer (DPO/GRPO/PPO)")
    print("=" * 70)

    return len(training_log) >= 2


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
