import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv

from agenttune.decide.closed_loop.behavioral_diversity_monitor import BehavioralDiversityMonitor
from agenttune.decide.closed_loop.contracts import ClassifiedFailure, Failure
from agenttune.decide.closed_loop.pipeline import SelfHealingPipeline
from agenttune.decide.closed_loop.replay_validator import ReplayValidator
from agenttune.decide.closed_loop.training_example_generator import TrainingExampleGenerator
from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator

load_dotenv()


async def run_full_week_pipeline():
    audit_log = "audit_week.jsonl"
    classified_log = "classified_week.jsonl"

    if os.path.exists(audit_log):
        os.remove(audit_log)
    if os.path.exists(audit_log + ".offset"):
        os.remove(audit_log + ".offset")
    if os.path.exists(classified_log):
        os.remove(classified_log)

    # ---------------------------------------------------------
    # 1. LIVE AGENT FAILURE SIMULATION (Writes to Audit Log)
    # ---------------------------------------------------------
    print("1. [Agent Sandbox] Simulating agent failures...")
    trace_1 = {
        "trajectory_id": f"live_run_{int(time.time())}",
        "stage_name": "agent_execution",
        "stage_type": "tool_call",
        "status": "error",
        "error_details": "SchemaViolation: 'name' is not a valid parameter. Expected 'user_id'.",
        "state_snapshot": {
            "messages": [
                {"role": "user", "content": "Look up user John Doe."},
                {
                    "role": "assistant",
                    "content": "I will look them up.",
                    "tool_calls": [{"name": "search_db", "arguments": '{"name": "John Doe"}'}],
                },
            ]
        },
    }

    with open(audit_log, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace_1) + "\n")

    print(f"   -> Wrote trace {trace_1['trajectory_id']} to {audit_log}")

    # ---------------------------------------------------------
    # 2. PIPELINE DETECTION & CLASSIFICATION
    # ---------------------------------------------------------
    print("\n2. [Self-Healing Pipeline] Detecting and classifying failures...")
    pipeline = SelfHealingPipeline(
        audit_log_path=audit_log,
        output_file=classified_log,
        classifier_model="groq/llama-3.3-70b-versatile",
    )
    await pipeline.run_once()

    # ---------------------------------------------------------
    # 3. RL GENERATOR & REPLAY VALIDATOR (TAC / TER Scoring)
    # ---------------------------------------------------------
    print("\n3. [Training Generator] Generating GRPO RL completions...")
    classified_failures = []
    if os.path.exists(classified_log):
        with open(classified_log, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                classified_failures.append(
                    ClassifiedFailure(
                        failure=Failure(
                            trajectory_id=data["trajectory_id"],
                            failure_type=data["failure_type"],
                            failed_stage_name=data["stage_name"],
                            error_message=data["error_message"],
                            judge_score=data["judge_score"],
                            context=data["context"],
                        ),
                        root_cause=data.get("root_cause", "unknown"),
                        confidence=data.get("confidence", 0.0),
                        analysis=data.get("analysis", ""),
                    )
                )

    sandbox_path = os.path.join(os.path.dirname(__file__), "dummy_sandbox.py")
    validator = ReplayValidator(validation_script=f"{sys.executable} {sandbox_path}")
    generator = TrainingExampleGenerator(
        validator=validator, model_name="groq/llama-3.3-70b-versatile"
    )

    batch = await generator.generate_batch(classified_failures)

    for ex in batch:
        print(f"   -> {ex.trajectory_id} (Root Cause: {ex.root_cause})")
        for idx, (comp, reward) in enumerate(zip(ex.completions, ex.rewards, strict=False)):
            content = comp[0]["content"].replace("\n", " ")
            print(f"      [Path {idx+1}] Reward: {reward:.2f} | Action: {content[:50]}...")

    # ---------------------------------------------------------
    # 4. TRAJECTORY EVALUATOR (LLM Judge)
    # ---------------------------------------------------------
    print("\n4. [Trajectory Evaluator] Running batch LLM evaluations...")
    evaluator = TrajectoryEvaluator(model_name="groq/llama-3.3-70b-versatile")
    # Measure ACTUAL latency of a synthetic simulated workload
    start_time = time.time()
    # Simulate an organic tool execution sequence taking ~1.2 to 1.5 seconds
    import random

    time.sleep(random.uniform(1.2, 1.5))
    actual_latency = (time.time() - start_time) * 1000

    sample_trajectory = {
        "trajectory_id": f"eval_live_{int(time.time())}",
        "tool_calls": [
            {"name": "search_db", "arguments": '{"query": "John"}'},
            {"name": "fetch_record", "arguments": '{"id": 123}'},
            {"name": "fetch_record", "arguments": '{"id": 123}'},
        ],
        "tool_outputs": ["query_success", "error: rate limit", "record_data"],
        "reasoning_trace": ["I need to search.", "Rate limited, trying again.", "Got it!"],
        "golden_trajectory": [{"name": "search_db"}, {"name": "fetch_record"}],
        "initial_plan": ["Search database", "Fetch record"],
        "goal": "Find user record",
        "status": "success",
        "final_answer": "John Doe's record was found.",
        "reference_answer": "Record found for John Doe.",
        "latency_ms": actual_latency,  # DYNAMIC ACTUAL LATENCY
    }

    eval_results = await evaluator.evaluate_batch([sample_trajectory])
    for res in eval_results:
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
        print(
            f"      Plan Adherence Score (PAS): {res.pas_score:.2f}"
            if res.pas_score is not None
            else "      Plan Adherence Score (PAS): N/A"
        )
        print(
            f"      Path Min Edit Distance (PMED): {res.pmed_score}"
            if res.pmed_score is not None
            else "      Path Min Edit Distance (PMED): N/A"
        )
        print(
            f"      Action-State Efficiency (ASE): {res.ase_score:.2f}"
            if res.ase_score is not None
            else "      Action-State Efficiency (ASE): N/A"
        )
        print(f"      Loop Collapse Frequency (LCF): {res.lcf_score}")
        print(f"      Response Latency: {res.latency_ms} ms")
        print(f"      BLEU-1 (Overlap): {res.bleu_score:.2f}")
        print(f"      BERTScore (Semantic): {res.bert_score:.2f}")

    # ---------------------------------------------------------
    # 5. BEHAVIORAL DIVERSITY MONITOR
    # ---------------------------------------------------------
    print("\n5. [Diversity Monitor] Checking for behavioral collapse...")
    monitor = BehavioralDiversityMonitor(diversity_threshold=0.3)

    # Simulate an agent hitting a loop collapse
    for _ in range(12):
        monitor.observe_trajectory({"tool_calls": ["search_db", "fetch_record"]})

    monitor.check_diversity()
    print("   -> Completed diversity scan over 12 identical trajectories.")


if __name__ == "__main__":
    asyncio.run(run_full_week_pipeline())
