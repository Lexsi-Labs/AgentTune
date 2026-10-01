import asyncio
import json
import os
import time

from dotenv import load_dotenv

load_dotenv()  # Load keys for litellm
from agenttune.decide.graph_runner import GraphRunner
from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator


async def main():
    config_path = "config.yaml"
    template_id = "generic/text_classify"
    print(f"Loading real agent from {template_id}...")
    try:
        runner = GraphRunner.from_template(template_id, config_path)
    except Exception as e:
        print(f"Could not load runner: {e}")
        return

    sample_input = "The product arrived damaged and I demand a refund."

    print("\nRunning real agent inference to generate a trajectory...")
    start_time = time.time()
    final_state = await runner.run(sample_input)
    actual_latency_ms = (time.time() - start_time) * 1000
    print(f"Execution finished in {final_state.elapsed_seconds:.2f}s")

    # Extract the audit log for the real latency
    audit_file = "audit.jsonl"
    real_trajectory = None
    if os.path.exists(audit_file):
        with open(audit_file) as f:
            lines = f.readlines()
            tool_calls = []
            tool_outputs = []
            logged_latency = 0.0

            for line in lines:
                entry = json.loads(line)
                if entry.get("pipeline_id") == final_state.pipeline_id:
                    tool_calls.append(
                        {"name": entry.get("stage_id"), "arguments": entry.get("inputs", {})}
                    )
                    tool_outputs.append(entry.get("outputs", {}))
                    logged_latency += entry.get("latency_ms") or 0.0

            final_latency = actual_latency_ms if logged_latency == 0.0 else logged_latency

            if tool_calls:
                real_trajectory = {
                    "trajectory_id": final_state.pipeline_id,
                    "tool_calls": tool_calls,
                    "tool_outputs": tool_outputs,
                    "latency_ms": final_latency,  # TRUE ACTUAL LATENCY
                    "goal": "Classify text",
                    "status": "success" if final_state.is_complete else "error",
                    "final_answer": str(final_state.stage_outputs),
                }

    if real_trajectory:
        print(f"\nReal Latency Captured: {real_trajectory['latency_ms']:.2f} ms")
        print("\nPassing real trajectory to TrajectoryEvaluator...")
        evaluator = TrajectoryEvaluator(model_name="groq/llama-3.3-70b-versatile")
        res_list = await evaluator.evaluate_batch([real_trajectory])
        res = res_list[0]

        print(f"   -> {res.trajectory_id}:")
        print(f"      LLM Judge Goal Completion: {res.goal_completion_score}")
        print(f"      LLM Judge Tool Validity: {res.tool_sequence_validity}")
        print(f"      LLM Judge Unnecessary Steps: {res.unnecessary_steps_penalty}")
        print(f"      LLM Judge Error Recovery: {res.error_recovery_score}")
        print(f"      LLM Judge Intent-Action Alignment (IASA): {res.iasa_score}")
        print(f"      LLM Judge Evidence Grounding (EGS): {res.egs_score}")
        print(f"      Deterministic TAC Score: {res.tac_score:.2f}")
        print(f"      Deterministic TER Score: {res.ter_score:.2f}")
        print(f"      API Redundancy Ratio (ARR): {res.arr_score:.2f}")
        print(f"      Self-Correction Success Rate (SCSR): {res.scsr_score:.2f}")
        print(f"      Reasoning-to-Action Density (RAD): {res.rad_score:.2f}")
        print(f"      Loop Collapse Frequency (LCF): {res.lcf_score}")
        print(f"      Response Latency: {res.latency_ms:.2f} ms (ACTUAL)")
        print(f"      BLEU-1 (Overlap): {res.bleu_score:.2f}")
        print(f"      BERTScore (Semantic): {res.bert_score:.2f}")
    else:
        print("Could not extract real trajectory from audit.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
