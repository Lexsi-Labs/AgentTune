import asyncio
import json
import os
import sys

from agenttune.decide.closed_loop.contracts import ClassifiedFailure, Failure
from agenttune.decide.closed_loop.replay_validator import ReplayValidator
from agenttune.decide.closed_loop.training_example_generator import TrainingExampleGenerator
from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator


async def run_presentation_demo():
    print("\n" + "=" * 60)
    print("   🚀 AgentTune Path A: GRPO Generation & Eval Demo")
    print("=" * 60 + "\n")

    # 1. Mock a classified failure
    print("[1/3] Receiving Classified Failure (Root Cause: wrong_tool)...")
    failure = ClassifiedFailure(
        failure=Failure(
            trajectory_id="demo_traj_999",
            failure_type="tool_schema_error",
            failed_stage_name="agent_execution",
            context={"messages": [{"role": "user", "content": "Find John Doe"}]},
        ),
        root_cause="wrong_tool",
        confidence=0.95,
        analysis="Agent used incorrect argument keys.",
    )

    # 2. Generator & Validator
    print("[2/3] Generating GRPO Multi-Completions & Replay Validation...")
    sandbox_path = os.path.join(os.path.dirname(__file__), "dummy_sandbox.py")
    validator = ReplayValidator(validation_script=f"{sys.executable} {sandbox_path}")
    generator = TrainingExampleGenerator(validator=validator, model_name="groq/openai/gpt-oss-20b")

    batch = await generator.generate_batch([failure])
    training_example = batch[0]

    # 3. Evaluator
    print("[3/3] Running Trajectory Evaluator (TAC, TER, BLEU, BERTScore)...\n")
    evaluator = TrajectoryEvaluator(model_name="groq/openai/gpt-oss-20b")

    sample_trajectory = {
        "trajectory_id": training_example.trajectory_id,
        "tool_calls": [{"name": "search_db", "arguments": '{"user_id": 123}'}],
        "tool_outputs": ["query_success"],
        "goal": "Find user record",
        "status": "success",
        "final_answer": "John Doe's record was found.",
        "reference_answer": "Record found for John Doe.",
        "latency_ms": 845.2,
    }

    eval_results = await evaluator.evaluate_batch([sample_trajectory])
    eval_res = eval_results[0]

    # 4. Beautiful JSON Output for Screenshots
    print("=" * 60)
    print("   📊 FINAL GRPO PAYLOAD & EVALUATION RESULTS")
    print("=" * 60)

    output_payload = {
        "trajectory_id": training_example.trajectory_id,
        "root_cause": training_example.root_cause,
        "grpo_completions_generated": len(training_example.completions),
        "rl_rewards_assigned": training_example.rewards,
        "evaluation_metrics": {
            "deterministic": {
                "TAC_Score_Schema_Match": eval_res.tac_score,
                "TER_Score_State_Novelty": eval_res.ter_score,
            },
            "llm_judge": {
                "Goal_Completion": eval_res.goal_completion_score,
                "Efficiency": eval_res.unnecessary_steps_penalty,
            },
            "nlp_proxies": {
                "Response_Latency_ms": eval_res.latency_ms,
                "BLEU_1_Overlap": round(eval_res.bleu_score, 2),
                "BERTScore_Semantic": round(eval_res.bert_score, 2),
            },
        },
    }

    print(json.dumps(output_payload, indent=4))
    print("\n✅ Process Complete!")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(run_presentation_demo())
