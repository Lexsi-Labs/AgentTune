import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

# Ensure we can import from src/ without needing pip install -e .
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from agenttune.agentic.harness import DictToolHarness
from agenttune.agentic.inference import APIEngine, OfflineVLLMEngine, TransformersEngine
from agenttune.agentic.tools.builtin.web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)


def get_engine(engine_type: str, model_name: str):
    if engine_type == "api":
        return APIEngine(model_name=model_name)
    elif engine_type == "transformers":
        return TransformersEngine(model_name=model_name)
    else:
        return OfflineVLLMEngine(model_name=model_name)


async def run_batch_dagger(
    prompts: list[str],
    student_model: str,
    teacher_model: str,
    student_engine_type: str = "vllm",
    teacher_engine_type: str = "vllm",
) -> list[dict[str, Any]]:
    """
    Massive Batch DAgger (Dataset Aggregation) Loop.
    1. Loads Student model.
    2. Batches all N prompts simultaneously to generate trajectories.
    3. Evaluates all outputs offline to find failures.
    4. Unloads Student, Loads Teacher.
    5. Batches all failed states simultaneously to generate corrections.
    """
    print("\n--- Batch DAgger Pipeline Starting ---")
    print(f"Total Examples to Process: {len(prompts)}")

    # 1. Setup Common Environment Environment
    tool = WebSearchTool()
    harness = DictToolHarness({"search": tool.execute})
    system_prompt = 'You are an AI assistant with access to a WebSearchTool. To use it, output exactly: <tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>'

    # Pre-build all conversation payloads
    batch_conversations = []
    for p in prompts:
        batch_conversations.append(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
        )

    # --- PHASE 1: STUDENT BATCH GENERATION ---
    print(f"\n[Phase 1] Loading Student ({student_model}) with {student_engine_type} engine...")
    student_engine = get_engine(student_engine_type, student_model)

    print(f"[Phase 1] Generating {len(batch_conversations)} actions in massive parallel batch...")
    student_results = await student_engine.generate_batch(batch_conversations, temperature=0.7)

    print("[Phase 1] Unloading Student...")
    if hasattr(student_engine, "cleanup"):
        student_engine.cleanup()
    del student_engine

    # --- PHASE 2: OFFLINE EVALUATION ---
    print(f"\n[Phase 2] Evaluating {len(student_results)} student trajectories...")
    failed_states = []

    from agenttune.decide.closed_loop.training_example_generator import _parse_tool_call

    for i, student_action_str in enumerate(student_results):
        error_msg = None
        try:
            tool_call = _parse_tool_call(student_action_str)
            if tool_call:
                # Mock step just to see if it throws parsing/formatting errors
                observation, reward, done, info = harness.step(tool_call)
        except Exception as e:
            error_msg = str(e)

        if not error_msg and "<tool_call>" not in student_action_str:
            error_msg = "Student failed to output a valid tool call."

        if error_msg:
            # We found a failure!
            failed_state = f"{system_prompt}\n\nUser: {prompts[i]}\n\nStudent Action:\n{student_action_str}\n\nError:\n{error_msg}"
            failed_states.append(
                {"index": i, "failed_state": failed_state, "student_action": student_action_str}
            )

    print(f"[Phase 2] Found {len(failed_states)} failed trajectories requiring correction.")

    if not failed_states:
        return []

    # --- PHASE 3: TEACHER BATCH CORRECTION ---
    print(f"\n[Phase 3] Loading Teacher ({teacher_model}) with {teacher_engine_type} engine...")
    teacher_engine = get_engine(teacher_engine_type, teacher_model)

    correction_conversations = []
    for failure in failed_states:
        correction_conversations.append(
            [
                {
                    "role": "system",
                    "content": "You are a master teacher correcting an AI agent. The agent just made an error in the following state. Provide the optimal next action (using proper <tool_call> syntax) and thought process to correct it.",
                },
                {
                    "role": "user",
                    "content": f"State:\n{failure['failed_state']}\n\nCorrect this action:",
                },
            ]
        )

    print(
        f"[Phase 3] Generating {len(correction_conversations)} corrections in massive parallel batch..."
    )
    teacher_results = await teacher_engine.generate_batch(correction_conversations, temperature=0.0)

    print("[Phase 3] Unloading Teacher...")
    if hasattr(teacher_engine, "cleanup"):
        teacher_engine.cleanup()
    del teacher_engine

    # --- PHASE 4: AGGREGATE RESULTS ---
    dpo_pairs = []
    for j, failure in enumerate(failed_states):
        dpo_pairs.append(
            {
                "prompt": f"User: {prompts[failure['index']]}\n\n",
                "chosen": teacher_results[j],
                "rejected": failure["student_action"],
            }
        )

    return dpo_pairs


async def main():
    parser = argparse.ArgumentParser(description="Batch DAgger Correction Loop")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen1.5-0.5B-Chat")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen1.5-1.8B-Chat")
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Path to JSONL dataset of prompts"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="How many samples to process from dataset"
    )
    parser.add_argument(
        "--student_engine", type=str, default="vllm", choices=["vllm", "api", "transformers"]
    )
    parser.add_argument(
        "--teacher_engine", type=str, default="vllm", choices=["vllm", "api", "transformers"]
    )
    args = parser.parse_args()

    prompts = []
    if args.dataset_path and os.path.exists(args.dataset_path):
        print(f"Loading prompts from {args.dataset_path}...")
        with open(args.dataset_path) as f:
            for i, line in enumerate(f):
                if i >= args.num_samples:
                    break
                data = json.loads(line)
                # Fallback to multiple common prompt keys
                p = data.get("prompt", data.get("question", data.get("text", "")))
                if p:
                    prompts.append(p)
    else:
        print("No dataset provided, using dummy prompts.")
        prompts = [
            "Search for the current CEO of Microsoft and tell me their name.",
            "What is the weather in Tokyo right now?",
            "Who won the 2022 World Cup?",
            "Can you tell me a joke?",
            "Use the web search tool to find the population of Paris.",
        ]

    results = await run_batch_dagger(
        prompts, args.student_model, args.teacher_model, args.student_engine, args.teacher_engine
    )

    print(f"\nCollected {len(results)} on-policy DAgger corrections.")

    if results:
        out_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../data/batch_dagger_dpo.jsonl")
        )
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Saved DPO pairs to {out_file}")

        print("\nExample Pair:")
        print(f"Rejected (Student): {results[0]['rejected']}")
        print(f"Chosen (Teacher):\n{results[0]['chosen']}")


if __name__ == "__main__":
    asyncio.run(main())
