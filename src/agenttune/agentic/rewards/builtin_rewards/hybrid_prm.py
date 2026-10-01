import asyncio
import logging

from agenttune.utils.score_logger import log_score

logger = logging.getLogger(__name__)


def hybrid_prm_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Robust Hybrid Process Reward Model (PRM) for GRPO, PPO, and RLOO.
    Evaluates completions using the TrajectoryEvaluator's metrics and returns a composite scalar reward.

    Dynamic Configuration via kwargs:
      - use_llm_judge (bool): Default False. Set to True to enable soft metrics (IASA, SCSR).
      - eval_model_name (str): Default "groq/llama-3.3-70b-versatile".
      - arr_penalty_weight (float): Default 0.2
      - lcf_penalty_weight (float): Default 0.1
    """
    from agenttune.agentic.rollout_engines.rollout_factory import _extract_tool_calls
    from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator

    # 1. Dynamic Config
    use_llm_judge = kwargs.get("use_llm_judge", False)
    eval_model_name = kwargs.get("eval_model_name", "groq/llama-3.3-70b-versatile")
    arr_penalty_weight = kwargs.get("arr_penalty_weight", 0.2)
    lcf_penalty_weight = kwargs.get("lcf_penalty_weight", 0.1)

    # Check if WandB is active for granular logging
    wandb_run = None
    try:
        import wandb

        if wandb.run is not None:
            wandb_run = wandb
    except ImportError:
        pass

    # 2. Reconstruct robust trajectories
    trajectories = []
    for prompt, comp in zip(prompts, completions, strict=False):
        # Shared parser: recognises Qwen <tool_call>, Llama <function=...>,
        # Mistral [TOOL_CALLS], DeepSeek's <｜tool▁call▁begin｜>, and OpenAI-style
        # tool_calls, not just Qwen's tag.
        calls = _extract_tool_calls({"role": "assistant", "content": comp}, comp) or []
        parsed_tools = [
            {"name": c["function"].get("name"), "arguments": c["function"].get("arguments")}
            for c in calls
        ]

        trajectories.append(
            {
                "trajectory_id": "rl_rollout",
                "prompt": prompt,
                "completion": comp,
                "tool_calls": parsed_tools,
                "error": None,
                "latency_ms": 100.0,
            }
        )

    # 3. Evaluate Batch with Fault Tolerance
    model_name = eval_model_name if use_llm_judge else "offline-deterministic-only"

    try:
        evaluator = TrajectoryEvaluator(model_name=model_name)
        results = asyncio.run(evaluator.evaluate_batch(trajectories))
    except Exception as e:
        logger.error(
            f"[Hybrid PRM] TrajectoryEvaluator crashed: {e}. Defaulting batch to 0.0 reward."
        )
        log_score(
            "hybrid_prm_reward",
            0.0,
            reasons=[f"TrajectoryEvaluator crashed: {e} -> whole batch defaults to 0.0"],
            meta={"n": len(prompts)},
        )
        # Return base zeroes to prevent training crash
        return [0.0] * len(prompts)

    # 4. Compute Scores & Track Metrics
    rewards = []
    batch_metrics = {"tac": 0.0, "arr": 0.0, "lcf": 0.0, "composite": 0.0}
    if use_llm_judge:
        batch_metrics.update({"iasa": 0.0, "scsr": 0.0, "base_judge": 0.0})

    for i, r in enumerate(results):
        tac = getattr(r, "tac_score", 0.0)
        arr = getattr(r, "arr_score", 0.0)
        lcf = getattr(r, "lcf_score", 0)

        reasons = [f"tac_score={tac}"]
        score = tac - (arr * arr_penalty_weight)
        reasons.append(
            f"arr_score={arr} * arr_penalty_weight={arr_penalty_weight} subtracted -> {score}"
        )
        if lcf > 0:
            score -= lcf_penalty_weight * lcf
            reasons.append(f"lcf_score={lcf} > 0 -> -{lcf_penalty_weight * lcf} -> {score}")

        batch_metrics["tac"] += tac
        batch_metrics["arr"] += arr
        batch_metrics["lcf"] += lcf

        if use_llm_judge:
            base = getattr(r, "overall_judge_score", 0.0)
            scsr = getattr(r, "scsr_score", 0.0)
            iasa = getattr(r, "iasa_score", 0.0)
            score = (score + base + scsr + iasa) / 4.0
            reasons.append(
                f"use_llm_judge=True -> averaged with judge={base}, scsr={scsr}, iasa={iasa} -> {score}"
            )

            batch_metrics["base_judge"] += base
            batch_metrics["scsr"] += scsr
            batch_metrics["iasa"] += iasa

        final_reward = max(0.0, min(1.0, score))
        if final_reward != score:
            reasons.append(f"clamped to [0, 1] -> {final_reward}")
        log_score(
            "hybrid_prm_reward",
            final_reward,
            reasons=reasons,
            components={"tac": tac, "arr": arr, "lcf": lcf},
            meta={"index": i},
        )
        rewards.append(final_reward)
        batch_metrics["composite"] += final_reward

    # 5. Log metrics
    n = len(results)
    if n > 0:
        for k in batch_metrics:
            batch_metrics[k] /= n

        logger.info(f"[Hybrid PRM] Batch Means -> {batch_metrics}")

        if wandb_run:
            wandb_run.log({f"reward/prm_{k}": v for k, v in batch_metrics.items()})

    return rewards
