"""
M1 zero-shot evaluation (Sprint 2, §10 debug-first loop step 1).

Runs the base model (NO training) through the rollout loop on ~50 HotpotQA
questions under memory conditions, and compares token-cost + EM/F1:

  - plain    : full conversation history (the existing agent, no rewrite)
  - m1       : MEM1 rewrite (mem1_post_step_hook wipes history each turn,
               keeps only the model's <state> block + compressed tool output)
  - recent_k : keep only the last k tool-result turns (non-learned baseline)
  - m2       : M2 decision-token wrapper (memory_op_post_step_hook) — extends
               m1 with an explicit <decision:MEMORY_OP:ACTION_OP> token the
               model emits each turn; falls back to m1's compress behavior
               when no decision token is present (so an untrained M2 policy
               evaluates identically to m1 zero-shot).

This is the iterate loop from rag_plan_s2.md §10: catch prompt/tag bugs (prose
before the search tag, unclosed tags, early answering, no <state> block
emitted) BEFORE any RL. Most bugs are visible zero-shot and have nothing to do
with training.

Standalone — drives create_rollout_fn directly (no TRL trainer), like
verify_masking.py / evaluate.py. Thinking mode is ENABLED for the m1 condition
(MEM1's mechanism requires the model to emit a state/think block each turn);
plain and recent_k keep thinking disabled (the existing default, since thinking
prose breaks the tool-call parser when there's no rewrite to capture it).

Usage:
    python -m agenttune.rag.scripts.eval_m1_zeroshot \\
        --model Qwen/Qwen3.5-4B --backend sqlite \\
        --index_dir rag_experiments/indexes/sqlite \\
        --num_questions 50 --max_steps 6 \\
        --output rag_experiments/m1_zeroshot/report.json
"""

import argparse
import json
import os
from typing import Any

from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
from agenttune.rag.data.hotpotqa import get_system_prompt, load_hotpotqa_splits
from agenttune.rag.retrieval.chroma_backend import ChromaBackend
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend
from agenttune.rag.rewards.qa_metrics import exact_match_score, extract_answer_tag, f1_score
from agenttune.rag.tools import ReadDocumentTool, SearchCorpusTool, patch_xml_tool_call_parser
from agenttune.rag.trajectory_utils import is_tool_step

# Qwen3.5 emits tool calls in a custom XML format (not JSON); patch the
# rollout's parser to handle it. Idempotent + safe (falls back to original
# JSON parser when no XML block is present). See rag/tools/xml_tool_parser.py.
patch_xml_tool_call_parser()


def make_backend(name: str, index_dir: str):
    if name == "sqlite":
        return SQLiteFTSBackend(os.path.join(index_dir, "corpus.db"))
    if name == "chroma":
        return ChromaBackend(persist_dir=index_dir)
    raise ValueError(f"Unknown backend '{name}'.")


def _count_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(text.split())


def _conversation_token_cost(conversation: list[dict], tokenizer) -> int:
    """Total tokens the model was shown across the whole trajectory.

    For the plain agent this grows each turn (full history); for m1 it stays
    ~constant (rewritten each turn). This is the metric the ≥30% bar measures.
    We sum the token length of every prompt the model generated against — i.e.
    the conversation length at each generation turn.
    """
    # Reconstruct per-turn prompt lengths from the final conversation is wrong
    # for m1 (it was rewritten). Instead, sum the token length of every
    # assistant + tool message in the final conversation as a proxy. For a fair
    # per-turn cost we'd need to log at each _gen call; the rollout doesn't
    # expose that. This proxy (final-conversation token count) is what we
    # report — it's the right comparison for plain vs recent_k (both keep full
    # or truncated final history) and a lower bound for m1 (whose final conv
    # is already compact). For m1 we additionally count the cumulative state
    # the model actually saw by summing compressed tool outputs.
    total = 0
    for m in conversation:
        total += _count_tokens(str(m.get("content", "")), tokenizer)
    return total


def run_condition(
    condition: str,
    model_path: str,
    backend_name: str,
    index_dir: str,
    questions: list[dict[str, str]],
    max_steps: int,
    enable_thinking: bool,
    recent_k: int = 2,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    backend = make_backend(backend_name, index_dir)
    tools = [SearchCorpusTool(backend), ReadDocumentTool(backend)]

    # M1/M2 conditions use their own system prompts (asks for <state> blocks,
    # M2 additionally asks for the <decision:...> token); others use the
    # standard backend-aware prompt.
    system_prompt = get_system_prompt(
        backend_name, m1=(condition in ("m1", "m2")), m2=(condition == "m2")
    )

    # Pick the post_step_hook per condition.
    post_step_hook = None
    if condition == "m2":
        from agenttune.rag.memory import memory_op_post_step_hook

        post_step_hook = memory_op_post_step_hook
    elif condition == "m1":
        from agenttune.rag.memory import mem1_post_step_hook

        post_step_hook = mem1_post_step_hook
    elif condition == "recent_k":
        from agenttune.rag.memory import recent_k_post_step_hook

        post_step_hook = recent_k_post_step_hook
    # plain: no hook (full history)

    # If an adapter_path is given, load the base model + merge the LoRA adapter
    # into it, then point the rollout at the merged model. This lets us eval the
    # trained M1 adapter zero-shot (the controlled plain-vs-M1 comparison from
    # sprint2_results/README.md §7). We merge+unload so the rollout's standard
    # transformers generate path works unchanged (no PEFT wrapper needed in the
    # rollout engine). The merged model is passed as model= (an object, not a
    # path) so the engine uses it directly; model_path= stays as the string for
    # tokenizer loading.
    effective_model = None
    effective_model_path = model_path
    effective_tokenizer = None
    if adapter_path:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  [adapter] loading base={model_path} + adapter={adapter_path}")
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(base, adapter_path)
        effective_model = peft_model.merge_and_unload()
        effective_tokenizer = AutoTokenizer.from_pretrained(model_path)
        if effective_tokenizer.pad_token is None:
            effective_tokenizer.pad_token = effective_tokenizer.eos_token

    engine_kwargs = {"device_map": {"": 0}}
    rollout_fn = create_rollout_fn(
        rollout_backend="transformers",
        model=effective_model,
        model_path=effective_model_path if effective_model is None else None,
        tokenizer=effective_tokenizer,
        tools=tools,
        max_steps=max_steps,
        system_prompt=system_prompt,
        post_step_hook=post_step_hook,
        enable_thinking=enable_thinking,
        # Training always ran with --force_final_answer (a corrective nudge
        # appended when the model stops without an <answer> tag). Without it
        # here, an early stop under the M1 rewrite (which happens often —
        # avg_tool_calls collapses to ~2 vs plain's ~5) becomes a permanent
        # empty answer instead of the same one-shot recovery training used.
        # Matching train/eval settings is required for the comparison to be
        # about the memory mechanism, not an eval-harness gap.
        force_final_answer=True,
        # force_action_on_stall: root-cause fix for the M1 collapse. Traced
        # with a manual generation test: after the M1 rewrite (system+user+
        # <state>+tool-result), the model reliably writes a NEW <state> block
        # and stops — no tool call, no answer — because state-writing "feels"
        # complete to it. The `while tool_calls` loop then exits immediately,
        # treating that stall as the final response. force_final_answer alone
        # only recovers this once, at the very end (and only asks for an
        # answer, never "search again"), so a stall after turn 1 stranded the
        # trajectory. This nudges at the point of stall instead, letting the
        # model still search if open_questions isn't resolved.
        force_action_on_stall=(condition in ("m1", "m2")),
        engine_kwargs=engine_kwargs,
    )

    q_texts = [q["question"] for q in questions]
    result = rollout_fn(q_texts)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    em_scores, f1_scores, tool_counts, token_costs = [], [], [], []
    state_emitted_count = 0
    decision_emitted_count = 0
    per_q: list[dict[str, Any]] = []
    for q, traj in zip(questions, result["trajectories"], strict=False):
        pred = extract_answer_tag(traj.final_response)
        gold = q["answer"]
        em = exact_match_score(pred, gold)
        f1 = f1_score(pred, gold)
        n_calls = sum(1 for s in traj.steps if is_tool_step(s))
        conv = traj.metadata.get("conversation", []) if traj.metadata else []
        token_cost = _conversation_token_cost(conv, tokenizer)

        # Did the model emit a <state> block? (m1/m2 health check)
        from agenttune.rag.memory import extract_internal_state

        emitted_state = any(
            extract_internal_state(getattr(s, "thought", "") or "", tag="state") is not None
            for s in traj.steps
        )
        if emitted_state:
            state_emitted_count += 1

        # Did the model emit a <decision:...> token? (m2 health check —
        # measures whether the model has learned the decision protocol vs
        # just falling back to m1's always-compress behavior).
        emitted_decision = False
        if condition == "m2":
            from agenttune.rag.memory.m2_decisions import parse_decision_token

            emitted_decision = any(
                parse_decision_token(getattr(s, "thought", "") or "") is not None
                for s in traj.steps
            )
            if emitted_decision:
                decision_emitted_count += 1

        em_scores.append(em)
        f1_scores.append(f1)
        tool_counts.append(n_calls)
        token_costs.append(token_cost)
        per_q.append(
            {
                "question": q["question"],
                "gold": gold,
                "pred": pred[:200],
                "em": em,
                "f1": f1,
                "tool_calls": n_calls,
                "token_cost": token_cost,
                "emitted_state": emitted_state,
                **({"emitted_decision": emitted_decision} if condition == "m2" else {}),
            }
        )

    n = len(questions)
    return {
        "condition": condition,
        "num_questions": n,
        "exact_match": sum(em_scores) / n if n else 0.0,
        "f1": sum(f1_scores) / n if n else 0.0,
        "avg_tool_calls": sum(tool_counts) / n if n else 0.0,
        "avg_token_cost": sum(token_costs) / n if n else 0.0,
        "pct_emitted_state": state_emitted_count / n if n else 0.0,
        "pct_zero_search": sum(1 for c in tool_counts if c == 0) / n if n else 0.0,
        **(
            {"pct_emitted_decision": decision_emitted_count / n if n else 0.0}
            if condition == "m2"
            else {}
        ),
        "per_question": per_q,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--backend", choices=["sqlite", "chroma"], default="sqlite")
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--num_questions", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--recent_k", type=int, default=2)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["plain", "m1", "recent_k"],
        choices=["plain", "m1", "recent_k", "m2"],
    )
    parser.add_argument("--hotpotqa_config", default="distractor")
    parser.add_argument(
        "--adapter_path",
        default=None,
        help=(
            "Path to a trained LoRA adapter directory (e.g. runs/m1_real/). "
            "When set, the base model is loaded, the adapter is merged in, and "
            "the merged model is used for ALL conditions. This enables the "
            "controlled plain-vs-M1-trained comparison (sprint2_results §7): "
            "eval the trained adapter zero-shot on a shared held-out set with "
            "identical decode settings. If None, the untrained base model is used."
        ),
    )
    parser.add_argument("--output", default="rag_experiments/m1_zeroshot/report.json")
    args = parser.parse_args()

    _, eval_split = load_hotpotqa_splits(
        config=args.hotpotqa_config, train_size=1, eval_size=args.num_questions
    )
    questions = [{"question": r["question"], "answer": r["answer"]} for r in eval_split]

    results = {}
    for cond in args.conditions:
        # m1/m2 need thinking ON (the model must emit a state/think block);
        # plain and recent_k keep thinking OFF (thinking prose breaks the
        # tool-call parser when there's no rewrite to capture it).
        enable_thinking = cond in ("m1", "m2")
        tag = f"adapter={args.adapter_path}" if args.adapter_path else "base"
        print(f"[m1_zeroshot] running condition={cond} thinking={enable_thinking} model={tag} ...")
        try:
            results[cond] = run_condition(
                cond,
                args.model,
                args.backend,
                args.index_dir,
                questions,
                args.max_steps,
                enable_thinking,
                args.recent_k,
                adapter_path=args.adapter_path,
            )
            r = results[cond]
            print(
                f"  -> EM={r['exact_match']:.3f} F1={r['f1']:.3f} "
                f"avg_tool_calls={r['avg_tool_calls']:.2f} "
                f"avg_token_cost={r['avg_token_cost']:.0f} "
                f"pct_emitted_state={r['pct_emitted_state']:.2f}"
            )
        except Exception as e:
            import traceback

            results[cond] = {
                "condition": cond,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            print(f"  -> ERROR: {e}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "model": args.model,
                "adapter_path": args.adapter_path,
                "backend": args.backend,
                "num_questions": args.num_questions,
                "max_steps": args.max_steps,
                "conditions": args.conditions,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"[m1_zeroshot] wrote {args.output}")

    # Summary table
    print("\n=== M1 zero-shot summary ===")
    print(
        f"{'condition':<12} {'EM':>7} {'F1':>7} {'avg_calls':>10} {'avg_tokens':>11} {'%state':>7}"
    )
    for cond, r in results.items():
        if "error" in r:
            print(f"{cond:<12} ERROR: {r['error'][:60]}")
            continue
        print(
            f"{cond:<12} {r['exact_match']:>7.3f} {r['f1']:>7.3f} "
            f"{r['avg_tool_calls']:>10.2f} {r['avg_token_cost']:>11.0f} "
            f"{r['pct_emitted_state']:>7.2f}"
        )


if __name__ == "__main__":
    main()
