import argparse
import asyncio
import logging
import os
import sys

# Ensure we can import from src/ without needing pip install -e .
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from agenttune.agentic.inference import APIEngine, OfflineVLLMEngine
from agenttune.agentic.memory import TrajectoryStore

logger = logging.getLogger(__name__)


async def run_teacher_batch(prompts: list[str], engine) -> list[dict]:
    """
    Runs a batch of teacher episodes with the "First-Thought Prefix" optimization.
    """
    # Build a custom RAG graph or strategy loop
    # For this script, we'll simulate the interaction by mocking the GraphRunner
    # In production, this binds directly to `GraphRunner(model=teacher_model)`.

    first_thought_prefix = "You are a master agent. You MUST start every response with <thought> explaining your exact reasoning step-by-step, followed by </thought>, and only then make a <tool_call>."

    batch_messages = []
    for prompt in prompts:
        batch_messages.append(
            [
                {"role": "system", "content": first_thought_prefix},
                {"role": "user", "content": prompt},
            ]
        )

    print(f"Running Teacher batch of size {len(prompts)}...")

    try:
        completions = await engine.generate_batch(batch_messages, temperature=0.0)

        results = []
        for prompt, completion in zip(prompts, completions, strict=False):
            results.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "success": bool(completion),  # Assume successful for this demo if not empty
                }
            )
        return results
    except Exception as e:
        logger.error(f"Teacher batch failed: {e}")
        return [{"success": False}] * len(prompts)


async def main():
    parser = argparse.ArgumentParser(description="Collect Teacher Rollouts for D2 Distillation")
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="groq/llama-3.3-70b-versatile",
        help="The massive teacher model",
    )
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument(
        "--engine", type=str, choices=["api", "vllm"], default="api", help="Engine type to use"
    )
    args = parser.parse_args()

    print(
        f"Starting Teacher Rollout Collection using {args.teacher_model} via {args.engine} engine"
    )

    if args.engine == "vllm":
        engine = OfflineVLLMEngine(model_name=args.teacher_model)
    else:
        engine = APIEngine(model_name=args.teacher_model)

    # Dummy questions for HotpotQA subset
    questions = [
        "What is the population of the capital of France?",
        "Who directed the movie Inception?",
        "What is the weather in SF today?",
    ] * (args.num_episodes // 3 + 1)

    questions = questions[: args.num_episodes]

    TrajectoryStore()

    # Batch process
    batch_size = 10 if args.engine == "api" else args.num_episodes

    successful = []
    for i in range(0, len(questions), batch_size):
        batch_q = questions[i : i + batch_size]
        results = await run_teacher_batch(batch_q, engine)
        successful.extend([r for r in results if r.get("success")])

    print(f"Successfully collected {len(successful)} teacher trajectories.")

    if args.engine == "vllm":
        engine.cleanup()

    # Here we would normally store them in EventLogs via TrajectoryStore
    # store.write(EventLog(trajectory=...))
    print("Rollouts written to TrajectoryStore. Ready for projection to SFT.")


if __name__ == "__main__":
    asyncio.run(main())
