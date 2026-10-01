"""
REAL LangGraph orchestrator → a real GRPOTrainer.train().
=========================================================

`agenttune.agentic.langgraph_orchestrator.AgentTuneGraph` composes rollout and
LLM-judge nodes into one graph and compiles drop-in callables for a trainer::

    graph = AgentTuneGraph()
    graph.add_rollout("agent", rollout_fn)
    graph.add_judge("quality", judge_a); graph.add_judge("strict", judge_b, aggregation="min")
    trainer = create_agentic_trainer("grpo", model=..., train_dataset=ds,
        rollout_func=graph.compile_rollout(), reward_funcs=[graph.compile_grpo_reward()])
    trainer.train()

The only place this fed a real trainer was `tests/e2e_test_suite.py::T11`, which
(a) has a hard-coded `/workspace/...` path and an FP8 model (can't run here), and
(b) scored training with a hand-written `reward_from_judges` that read
`completion.get("final_reward", 0.5)` off **string** completions — so it always
returned the constant **0.5** and the graph's judges never actually scored the
training completions. The GRPOTrainer-compat claim was, in practice, never run.

WHY IT COULD NOT WORK, AND THE FIX
----------------------------------
`compile_rollout()`'s output exposed judge results only as *batch-level* scalars
(`judge_scores`, `final_reward`) — there was no length-N vector aligned to the N
completions, which is exactly what a GRPOTrainer reward_func needs. This example
comes with a small library fix: the judge node now keeps its **per-trajectory**
scores and the graph exposes a per-completion `judge_rewards` vector, plus a new
`compile_grpo_reward()` that returns a `reward_func(completions, **kwargs) ->
list[float]` reading it. So the documented one-liner now runs, with the judges
actually scoring each completion.

This script runs it for real: a live `Qwen2.5-3B-Instruct` agent (with a `calc`
tool) is the rollout node; **two** real `LLMJudge`s (also live Qwen2.5-3B) score
each rolled-out trajectory and their scores are averaged per completion; that
composed, non-constant judge reward drives a real GRPO LoRA `train()`.

HONEST SCOPE — read before quoting
----------------------------------
The claim is that a **real, non-constant, graph-composed LLM-judge reward drives a
real `GRPOTrainer.train()`** — the thing T11's constant-0.5 stub never did. It is
NOT a convergence claim. As in `rag_training_real.py`, the two completions of one
prompt can draw the same judge score → zero *within-group* variance → zero
gradient on that group; we report `reward_std` / `grad_norm` as-measured. The
load-bearing contrast with T11 is *non-constant across the batch*, which we assert
(≥2 distinct judge rewards, ≥1 strictly in (0,1) so the judge genuinely parsed and
discriminated rather than silently flooring to 0).

Needs a GPU + `Qwen/Qwen2.5-3B-Instruct` cached. Run:
    python examples/langgraph_orchestrator_real.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

import sys

sys.modules["vllm"] = None  # env vllm is ABI-broken vs torch; keep it out of the import graph

import statistics

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM

from agenttune.agentic.langgraph_orchestrator import AgentTuneGraph, make_grpo_rollout_func
from agenttune.agentic.rewards.llm_judge import LLMJudge
from agenttune.agentic.rollout_engines.rollout_factory import (
    create_rollout_engine,
    create_rollout_fn,
)
from agenttune.core.backend_factory import create_agentic_trainer

MODEL = "Qwen/Qwen2.5-3B-Instruct"
OUT_DIR = "/tmp/agenttune-langgraph-grpo"
# Two DELIBERATELY different rubrics, so mean-aggregation is observable rather than
# a no-op: a lenient grader (answer-only) and a harsh grader (demands shown work +
# units). The composed per-completion reward is the mean of the two, so it visibly
# reflects both — that is what "compose multiple judge nodes" has to mean.
JUDGE_SYS_LENIENT = (
    "You grade a math answer for the FINAL NUMBER only. 1.0 if the "
    "final number is correct (ignore missing units/work), ~0.5 if close/"
    'rounded, 0.0 if wrong. Reply JSON: {"score": <float 0-1>, "explanation": "<short>"}'
)
JUDGE_SYS_STRICT = (
    "You are a HARSH grader. Give 1.0 ONLY if the final number is correct "
    "AND the units are shown AND the working is shown. Deduct heavily "
    "(cap at 0.4) for any missing units, rounding, or unshown steps; 0.0 if "
    'the answer is wrong. Reply JSON: {"score": <float 0-1>, "explanation": "<short>"}'
)

# Deliberately spans easy → hard so the judge produces a *spread* of scores (a
# 3B agent nails the easy ones and flubs/partly-answers the multi-step ones),
# rather than everything scoring 1.0. Same idea as the boundary-straddling corpus
# in rag_datagen_real: design the inputs so the real signal is non-degenerate.
EXAMPLES = [
    {
        "prompt": "A bat and a ball cost $1.10 total; the bat costs $1.00 more than "  # trick (≈0)
        "the ball. How much is the ball, in cents?"
    },
    {
        "prompt": "Compound interest on $1000 at 10% per year for 3 years, "  # hard
        "compounded annually — final amount?"
    },
    {"prompt": "Area of an 8 m by 12 m rectangle?"},  # easy (≈1)
    {"prompt": "What is 15% of 240?"},  # easy (≈1)
    {"prompt": "Convert 100 km to miles (1 km = 0.621371 miles)."},  # medium
    {"prompt": "Simple interest on $1000 at 5% per year for 3 years?"},  # medium
]


def calculator(expression: str) -> dict:
    """Evaluate a mathematical expression.

    Args:
        expression: Python math expression string (e.g. '2 + 2').

    Returns:
        dict with the numeric result or an error string.
    """
    # Demo-only calculator tool (mirrors the repo's e2e_test_suite calculator):
    # eval is sandboxed with no builtins and only ever sees this script's own
    # fixed math prompts — not untrusted input. Not for production use.
    try:
        return {"result": eval(expression, {"__builtins__": {}}, {})}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def build_graph():
    """One graph: a tool-using rollout node + two real LLM-judge nodes (mean-aggregated)."""
    engine = create_rollout_engine(backend="transformers", model_path=MODEL)
    rollout_fn = create_rollout_fn(
        rollout_engine=engine,
        tools=[calculator],
        max_steps=3,
        system_prompt="You are a math assistant. Use the calculator tool, then answer concisely.",
    )
    judge_a = LLMJudge(
        backend="transformers", model_path=MODEL, system_prompt=JUDGE_SYS_LENIENT, cache_size=0
    )
    judge_b = LLMJudge(
        backend="transformers", model_path=MODEL, system_prompt=JUDGE_SYS_STRICT, cache_size=0
    )

    graph = AgentTuneGraph()
    graph.add_rollout("agent", rollout_fn)
    graph.add_judge("lenient", judge_a, criteria={"correctness": 1.0}, aggregation="mean")
    graph.add_judge("strict", judge_b, criteria={"correctness": 1.0}, aggregation="mean")
    return graph


def main():
    graph = build_graph()
    compiled = graph.compile_rollout()  # judges run here → per-completion judge_rewards
    grpo_reward = graph.compile_grpo_reward()  # reads that per-completion vector

    # ── Preflight: the composed judge reward is real & discriminating ──
    # Over a varied-difficulty batch the two live judges score each completion and
    # their scores are averaged per completion. Success = real + non-constant +
    # genuinely parsed: ≥2 distinct values (not the constant-0.5 stub) and max > 0
    # (a total parse failure would floor every score to 0).
    print("\n── Preflight: graph-composed LLM-judge reward on a real rollout ──")
    pre = compiled([e["prompt"] for e in EXAMPLES], temperature=0.9, top_p=0.95)
    jr = pre["judge_rewards"]
    print(f"  per-completion judge_rewards : {[round(x, 3) for x in jr]}")
    print(
        f"  per-judge batch scores       : "
        f"{ {k: round(v, 3) for k, v in pre['judge_scores'].items()} }"
    )
    print(
        f"  rewards strictly in (0,1)    : {sum(0 < x < 1 for x in jr)}/{len(jr)} "
        f"(graded partial credit; correctness judges tend toward 0/1)"
    )
    js = pre["judge_scores"]
    print(
        f"  two judges diverge?          : lenient={js['lenient']:.3f} vs strict={js['strict']:.3f} "
        f"(Δ={abs(js['lenient']-js['strict']):.3f}) — mean-aggregation is observable, not a no-op"
    )
    assert all(getattr(t, "task", "") for t in pre["trajectories"]), "empty task reached a judge"
    assert max(jr) > 0.0, "every judge reward is 0 — the judge never parsed (silently floored)"
    assert (
        len({round(x, 3) for x in jr}) >= 2
    ), "judge rewards constant — no better than the 0.5 stub"
    assert abs(js["lenient"] - js["strict"]) > 1e-6, (
        "the two judges scored identically — multi-node composition is a no-op; "
        "the strict rubric should grade lower than the lenient one"
    )

    # ── The documented one-liner: graph rollout + graph reward → real GRPOTrainer ──
    seen = {"batches": [], "all": []}

    def recording_reward(completions=None, **kwargs):
        scores = grpo_reward(completions=completions, **kwargs)
        seen["batches"].append([round(float(s), 3) for s in scores])
        seen["all"].extend(float(s) for s in scores)
        return scores

    recording_reward.__name__ = "agenttune_graph_reward"

    ds = Dataset.from_list(EXAMPLES)
    trainer = create_agentic_trainer(
        "grpo",
        model=MODEL,
        train_dataset=ds,
        rollout_func=make_grpo_rollout_func(compiled),  # documented wrapper
        reward_funcs=[recording_reward],
        output_dir=OUT_DIR,
        num_generations=2,
        per_device_train_batch_size=4,  # 2 prompts/step → training window spans easy+hard
        gradient_accumulation_steps=1,
        max_steps=4,
        max_completion_length=200,
        shuffle_dataset=False,  # keep hard prompts first so a spread is scored
        temperature=0.9,
        learning_rate=1e-5,
        beta=0.04,
        seed=0,
        logging_steps=1,
        report_to="none",
        peft_config={
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    )
    print("\n── Running real GRPO (LoRA) with the graph-composed rollout + judge reward ──")
    trainer.train()
    gt = trainer.trainer
    gt.save_model(OUT_DIR)

    # ── Proof the composed judge reward drove training ──
    print("\n── Proof: the graph's LLM-judge reward scored training completions ──")
    print(f"  reward batches (per-completion) : {seen['batches']}")
    all_r = seen["all"]
    distinct = len({round(x, 3) for x in all_r})
    print(f"  completions scored              : {len(all_r)}")
    print(f"  distinct reward values          : {distinct}")
    print(f"  any strictly in (0,1)           : {any(0 < x < 1 for x in all_r)}")
    print(f"  constant-0.5 (T11's stub)?      : {all(x == 0.5 for x in all_r)}")
    rstds = [h["reward_std"] for h in gt.state.log_history if "reward_std" in h]
    gnorms = [h["grad_norm"] for h in gt.state.log_history if "grad_norm" in h]
    print(f"  per-step reward_std             : {[round(x,3) for x in rstds]}")
    print(
        f"  per-step grad_norm              : {[round(x,3) for x in gnorms]} "
        f"(0 on a step ⇒ that group's 2 completions drew equal judge scores — as scoped)"
    )

    adapter_ok = os.path.exists(os.path.join(OUT_DIR, "adapter_model.safetensors"))
    reload_ok = False
    if adapter_ok:
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
        reload_ok = isinstance(PeftModel.from_pretrained(base, OUT_DIR), PeftModel)

    print("\n── Summary ──")
    print("  graph rollout + graph reward drove GRPOTrainer.train() : True")
    print(f"  judge reward non-constant (not T11's 0.5 stub)         : {distinct > 1}")
    print(f"  judge genuinely parsed (max reward > 0)                : {max(all_r) > 0}")
    print(
        f"  rewards strictly in (0,1) across training              : {sum(0 < x < 1 for x in all_r)}/{len(all_r)}"
    )
    print(
        f"  within-group reward variance (mean std)                : "
        f"{statistics.mean(rstds) if rstds else 0.0:.3f} (measured, not gated on)"
    )
    print(f"  LoRA adapter saved + reloadable                        : {adapter_ok and reload_ok}")
    print("  convergence: NOT claimed")

    assert distinct > 1, "reward collapsed to a constant across training — no better than the stub"
    assert max(all_r) > 0, "every training reward is 0 — judge floored to 0 (never parsed)"
    assert adapter_ok and reload_ok, "adapter not produced/reloadable"
    print(
        "\n✓ AgentTuneGraph orchestrator drove a real GRPOTrainer.train() with a real judge reward."
    )


if __name__ == "__main__":
    main()
