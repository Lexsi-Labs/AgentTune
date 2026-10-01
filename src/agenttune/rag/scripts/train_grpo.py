"""
Phase 1: GRPO LoRA training driver for the agentic RAG use case.

Explicitly builds the rollout function via `create_rollout_fn` and passes it
as `rollout_func=...` into `create_agentic_trainer`. This is deliberate, not
optional: passing `tools=[...]` alone does NOT guarantee agenttune's own
masking-aware rollout path (`_execute_trajectory`/`_format_for_grpo`,
producing `env_mask`) is what actually runs during training — TRL's
GRPOTrainer only patches in agenttune's rollout when `rollout_func` is
explicitly set (see TrlAgenticGrpo's `_generate_single_turn` monkeypatch).
Being explicit here is what makes Phase 0's masking guarantee actually hold
during real training, not just in the standalone verify_masking.py check.

Usage:
    python -m agenttune.rag.scripts.train_grpo \\
        --model Qwen/Qwen2.5-1.5B-Instruct --backend sqlite \\
        --index_dir rag_experiments/indexes/sqlite \\
        --output_dir rag_experiments/runs/qwen1.5b_sqlite \\
        --trace_log_path rag_experiments/runs/qwen1.5b_sqlite/trace.jsonl
"""

import argparse
import glob
import json
import os
from collections.abc import Callable

from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
from agenttune.core.backend_factory import create_agentic_trainer
from agenttune.rag.data.hotpotqa import get_system_prompt, load_hotpotqa_splits, to_grpo_dataset
from agenttune.rag.retrieval.chroma_backend import ChromaBackend
from agenttune.rag.retrieval.hybrid_backend import HybridBackend
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend
from agenttune.rag.tools import ReadDocumentTool, SearchCorpusTool, patch_xml_tool_call_parser
from agenttune.rag.trajectory_utils import extract_question_text, extract_tool_calls

# Qwen3.5 emits tool calls in a custom XML format (not JSON); patch the
# rollout's parser to handle it. WITHOUT this, training rollouts would miss
# every tool call → zero search_usage/correctness reward → no learning signal.
# Idempotent + safe (falls back to original JSON parser). See
# rag/tools/xml_tool_parser.py.
patch_xml_tool_call_parser()


def make_backend(name: str, index_dir: str, match_all: bool = True):
    if name == "sqlite":
        return SQLiteFTSBackend(os.path.join(index_dir, "corpus.db"), match_all=match_all)
    if name == "chroma":
        return ChromaBackend(persist_dir=index_dir)
    if name == "hybrid":
        # BM25 + BGE-M3 dense, RRF-fused. Requires prebuilt
        # bge_m3_embeddings.npy + bge_m3_chunk_ids.json in index_dir
        # (build them offline with a BGE-M3 encoder over the same chunks
        # used for the sqlite/chroma index).
        return HybridBackend(index_dir=index_dir)
    raise ValueError(f"Unknown backend '{name}'.")


class TraceLogger:
    """Appends one JSON line per completed trajectory: question, tool calls
    (query + retrieved text), final answer, reward, gold answer.

    The reward here is computed directly via `reward_fn` — deliberately NOT
    read from `trajectory.reward`. `rollout_factory.py`'s `rollout_fn` only
    ever sets `t.reward` inside its own `if reward_fn:` branch, which requires
    passing `reward_fn=` to `create_rollout_fn` itself; `build_rollout_fn`
    doesn't do that (the real training reward is computed later, inside TRL's
    `GRPOTrainer` via `reward_funcs=`, and never written back onto the
    `Trajectory` object). Confirmed empirically before this fix: a trajectory
    that searched, retrieved the correct fact, and produced an answer exactly
    matching gold still logged `reward: 0.0`. Calling `reward_fn` here
    ourselves — with the same kwargs TRL passes it (`prompts`, `completions`,
    `gold_answer`, `tool_call_counts`; see `combine_rewards` in
    `agentic/rewards/composite.py` and the reward fns in
    `rag/rewards/phase1_rewards.py` for the exact calling convention this
    mirrors) — reproduces the real training signal exactly, entirely from
    this script, with zero framework changes.
    """

    def __init__(
        self,
        path: str,
        gold_by_question: dict[str, str],
        reward_fn: Callable,
        component_getter: Callable = None,
        gold_chunks_by_question: dict[str, list] = None,
        domain_by_question: dict[str, str] = None,
    ):
        self.path = path
        self.gold_by_question = gold_by_question
        # FinDER/mixed only: question -> gold chunk ids, so the trace can score
        # golden-chunk recall per trajectory. None for HotpotQA runs.
        self.gold_chunks_by_question = gold_chunks_by_question
        # Mixed-domain only: question -> domain, so the trace's reward call
        # uses the same per-domain correctness the training loop does.
        self.domain_by_question = domain_by_question
        self.reward_fn = reward_fn
        # Component getter: phase1's get_last_component_scores or T3's
        # get_last_t3_component_scores. Defaults to phase1 (backward-compatible).
        if component_getter is None:
            from agenttune.rag.rewards.phase1_rewards import get_last_component_scores

            component_getter = get_last_component_scores
        self._component_getter = component_getter
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "a").close()

    def __call__(self, trajectory) -> None:
        question = extract_question_text(trajectory.task)
        tool_calls = [
            {"query": call["arguments"], "result": s.observation}
            for s in trajectory.steps
            for call in extract_tool_calls(s)
        ]
        gold = self.gold_by_question.get(question)
        # Authoritative count straight from rollout_factory.py's own
        # bookkeeping (rollout_factory.py:406-409) — not recomputed from
        # `tool_calls` above, so it exactly matches what real training sees.
        tool_call_count = (
            trajectory.metadata.get("tool_call_count", 0) if trajectory.metadata else 0
        )
        reward_kwargs = {
            "completions": [trajectory.final_response],
            "prompts": [question],
            "gold_answer": [gold],
            "tool_call_counts": [tool_call_count],
        }
        if self.domain_by_question is not None:
            reward_kwargs["domain"] = [self.domain_by_question.get(question, "finder")]
        retrieved = (
            trajectory.metadata.get("retrieved_chunk_ids", []) if trajectory.metadata else []
        )
        if self.gold_chunks_by_question is not None:
            reward_kwargs["retrieved_chunk_ids"] = [retrieved]
            reward_kwargs["gold_chunk_ids"] = [self.gold_chunks_by_question.get(question, [])]
        reward = self.reward_fn(**reward_kwargs)[0]
        # Per-component scores for the trace — captured by the logged reward
        # fn's stashed dict (phase1 or T3, depending on --t3).
        comp = self._component_getter()
        record = {
            "question": question,
            "tool_calls": tool_calls,
            "n_tool_calls": tool_call_count,
            "final_answer": trajectory.final_response,
            "has_answer_tag": "<answer>" in (trajectory.final_response or "").lower(),
            "reward": reward,
            "gold_answer": gold,
            "reward_components": {k: (v[0] if v else 0.0) for k, v in comp.items()} if comp else {},
        }
        if self.gold_chunks_by_question is not None:
            record["retrieved_chunk_ids"] = retrieved
            record["gold_chunk_ids"] = self.gold_chunks_by_question.get(question, [])
            # Unique-query ratio input (FINNLP_EXPERIMENTS §5): the distinct-
            # query count per episode, for the query-echo -> reformulation
            # analysis. Computed post-hoc from tool_calls in the analysis
            # scripts; stored here as the raw query list is already present.
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


def make_unmasked_rollout_fn(base_rollout_fn: Callable) -> Callable:
    """
    Ablation mechanism (see plan): rollout_factory.py has no parameter to
    disable tool-token masking (it's unconditional-by-role). This wraps the
    REAL rollout function's output and overwrites env_mask to all-ones, so
    GRPO computes gradients over every token including retrieved-chunk text.
    Entirely confined to this script's process — rollout_factory.py is never
    edited.
    """

    def _wrapped(prompts, *a, **kw):
        result = base_rollout_fn(prompts, *a, **kw)
        if result.get("env_mask") is not None:
            result["env_mask"] = [
                ([1] * len(m) if m is not None else m) for m in result["env_mask"]
            ]
        return result

    return _wrapped


def build_rollout_fn(
    model_path: str,
    tools: list,
    max_steps: int,
    ablation: str,
    on_trajectory_end: Callable,
    system_prompt: str,
    enable_thinking: bool = False,
    force_final_answer: bool = False,
    force_action_on_stall: bool = False,
    post_step_hook: Callable | None = None,
):
    # This engine's own model weights are dead weight in real GRPOTrainer
    # training: rollout_factory.py's `_gen` "Path A" (trainer is not None,
    # confirmed by reading it directly) generates exclusively via
    # `trainer.model`/`trainer.model_wrapped` — `engine.model` is never read
    # for generation once a trainer is attached (only `engine.tokenizer` is
    # used elsewhere). Loading it on GPU anyway (the default `device_map=
    # "auto"`) doubles VRAM usage for nothing and was the actual cause of a
    # real OOM/silent-CPU-offload-into-NaN-logits failure on an 8GB GPU when
    # both this copy and the trainer's own copy tried to fit at once. Force
    # this copy onto CPU — harmless (it's structurally required to exist for
    # engine.tokenizer / tool-schema wiring, but its weights are never used
    # for a forward/generate pass in this script's actual training path).
    engine_kwargs = {"device_map": "cpu"}
    if ablation == "none":
        return create_rollout_fn(
            rollout_backend="transformers",
            model_path=model_path,
            tools=tools,
            max_steps=max_steps,
            system_prompt=system_prompt,
            on_trajectory_end=on_trajectory_end,
            engine_kwargs=engine_kwargs,
            # enable_thinking: OFF for the plain agent (thinking prose before a
            # tool call broke the parser when there's no rewrite to capture it —
            # see README §7). ON for M1 (MEM1's mechanism requires
            # the model to emit a state/think block each turn — that block IS
            # the memory carried forward). The parser strips the think block
            # before tool-call parsing, so tool calls are still detected.
            enable_thinking=enable_thinking,
            # force_final_answer: Search-R1 pattern — if the model never emitted
            # an <answer> tag, append a "now answer" user turn and generate once
            # more. Small models don't self-transition from searching to
            # answering; this guarantees answer-tag production so the reward has
            # real variance for GRPO. See rollout_factory.py _execute_trajectory.
            force_final_answer=force_final_answer,
            # force_action_on_stall: mid-trajectory version of force_final_answer.
            # Fixes the M1 collapse (see rag/memory/m1_rewrite.py): under the M1
            # rewrite, the model often stops after writing only a <state> block
            # (no tool call, no answer) — the `while tool_calls` loop then exits
            # immediately, treating the stall as the trajectory's final response.
            # force_final_answer only recovers this once, at the very end, and
            # only asks for an answer (never "search again"), so a stall after
            # turn 1 stranded the trajectory with unused step budget and a
            # premature guess. This nudge fires at the point of stall instead.
            force_action_on_stall=force_action_on_stall,
            post_step_hook=post_step_hook,
        )
    if ablation == "unmasked":
        base_fn = create_rollout_fn(
            rollout_backend="transformers",
            model_path=model_path,
            tools=tools,
            max_steps=max_steps,
            system_prompt=system_prompt,
            engine_kwargs=engine_kwargs,
            enable_thinking=enable_thinking,
            force_final_answer=force_final_answer,
            force_action_on_stall=force_action_on_stall,
            post_step_hook=post_step_hook,
            # on_trajectory_end intentionally NOT set here — set on the outer
            # call below instead, to avoid double-logging each trajectory.
        )
        wrapped = make_unmasked_rollout_fn(base_fn)
        return create_rollout_fn(custom_rollout_fn=wrapped, on_trajectory_end=on_trajectory_end)
    raise ValueError(f"Unknown ablation mode '{ablation}'.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--dataset",
        choices=["hotpotqa", "finder", "mixed", "cuad"],
        default="hotpotqa",
        help=(
            "Training dataset. 'finder' (E1 headline): Linq-AI-Research/FinDER "
            "with the ticker-level split + gold-chunk artifacts produced by "
            "build_index_finder.py (index_dir must contain splits.json + "
            "gold_chunks.json + corpus.db). 'cuad': synthetic-CUAD only — trains "
            "entirely on --cuad_dataset (no FinDER), uses the CUAD relaxed-legal-F1 "
            "correctness reward. 'mixed': 50/50 FinDER + synthetic-CUAD "
            "— index_dir must be the combined index from build_index_mixed.py, and "
            "--cuad_dataset points at the CUAD GRPO dataset (--train_size / "
            "--eval_size apply per domain, so --train_size 2000 = 4K total). Uses "
            "the FinDER reward stack with per-domain correctness (numeric tolerance "
            "for finance rows, relaxed legal F1 for CUAD rows). 'hotpotqa' is the "
            "Sprint-2 default."
        ),
    )
    parser.add_argument(
        "--cuad_dataset",
        default=None,
        help="Required when --dataset mixed or cuad: CUAD synthetic GRPO dataset "
        "(.json array or .jsonl) — e.g. dataset_grpo.jsonl from "
        "build_cuad_dataset.py.",
    )
    parser.add_argument(
        "--cuad_eval_dataset",
        default=None,
        help="Optional CUAD val dataset for --dataset mixed or cuad "
        "(gold-chunk-disjoint from train, e.g. dataset_val_grpo.jsonl). If "
        "omitted, eval slices from the tail of --cuad_dataset.",
    )
    parser.add_argument("--backend", choices=["sqlite", "chroma", "hybrid"], default="sqlite")
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--train_size", type=int, default=2000)
    parser.add_argument("--eval_size", type=int, default=200)
    parser.add_argument("--hotpotqa_config", default="distractor")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument(
        "--max_rollout_steps", type=int, default=6, help="max tool-call turns per trajectory"
    )
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help=(
            "Doesn't raise peak memory (microbatches are processed and freed "
            "sequentially) — use this, not per_device_train_batch_size, to grow "
            "effective batch size on a memory-constrained GPU. With batch_size=1 "
            "and train_size=100, --gradient_accumulation_steps 8 gives ~12 "
            "optimizer steps/epoch (100/8, rounded down by HF Trainer)."
        ),
    )
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="AdamW LR for the LoRA adapter. FINNLP_EXPERIMENTS v2 §7 range: "
        "1e-6..5e-6 for the 4B GRPO runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="HF Trainer seed. E1 runs 3 seeds (42/43/44) for the headline "
        "arm's mean±std (FINNLP_EXPERIMENTS v2 §5).",
    )
    parser.add_argument(
        "--report_to",
        nargs="+",
        default=["tensorboard"],
        choices=["tensorboard", "wandb", "none"],
        help=(
            "Where HF Trainer reports loss/reward/kl/grad_norm (and, via the "
            "RewardSignalLogger callback, per-component reward/<name> curves). "
            "'wandb' requires WANDB_API_KEY in the environment (never pass a "
            "key on the command line — it ends up in shell history and "
            "/proc). Pass both to log to both simultaneously: "
            "--report_to tensorboard wandb."
        ),
    )
    parser.add_argument(
        "--wandb_project",
        default="finagent",
        help="W&B project name (only used when 'wandb' is in --report_to).",
    )
    parser.add_argument(
        "--wandb_run_name",
        default=None,
        help="W&B run name; defaults to the basename of --output_dir.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help=(
            "Push the LoRA adapter to the Hugging Face Hub periodically "
            "during training (every --hub_push_steps optimizer steps, via "
            "the HubPushCallback below). Requires HF_TOKEN in the environment "
            "(never pass a token on the command line). Repo defaults to "
            "private; each push uploads to a "
            "`<output_dir basename>/checkpoint-<step>` subfolder — so "
            "multiple seeds/runs can share ONE repo_id (e.g. 'finagent') "
            "without their checkpoints colliding, and the Hub's own commit "
            "history is the version list within each run's subtree."
        ),
    )
    parser.add_argument(
        "--hub_model_id",
        default=None,
        help=(
            "HF Hub repo id, e.g. 'yourname/finagent-e1-finder-seed42'. "
            "Required when --push_to_hub is set."
        ),
    )
    parser.add_argument(
        "--hub_push_steps",
        type=int,
        default=50,
        help="Push a checkpoint to the Hub every N optimizer steps.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in --output_dir "
        "(HF Trainer auto-detects; we keep exactly one resume checkpoint "
        "via save_total_limit=1, so resume is unambiguous).",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        default=True,
        help="Create the Hub repo as private (default: on).",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=0,
        help=(
            "Run the cheap in-training eval pass (EM/F1, gold-chunk recall, "
            "unique-query ratio, search count/tokens — no API key, no LLM "
            "judge) every N optimizer steps, on up to --eval_sample_size "
            "questions from eval_dataset. 0 (default) disables it — matches "
            "FINNLP_EXPERIMENTS.md's own scoping of full eval (judge accuracy, "
            "citation precision) as a separate post-hoc pass, not part of the "
            "training loop."
        ),
    )
    parser.add_argument(
        "--eval_sample_size",
        type=int,
        default=32,
        help="Max number of eval_dataset questions per periodic eval pass "
        "(kept small — this runs INSIDE the training loop and each "
        "question costs a full multi-turn rollout).",
    )
    parser.add_argument(
        "--eval_context_length",
        type=int,
        default=4096,
        help="Cap the in-loop eval rollout's conversation to this many tokens. "
        "The eval runs the LIVE HF model (not vLLM) so it evaluates the exact "
        "current LoRA weights, but the HF generate path uses non-flash SDPA — "
        "a 9B x 12288-token context materializes ~19GB of attention scores and "
        "OOMs. 4096 keeps eval attention ~2GB. Final eval on the full 1k set is "
        "a separate post-hoc pass.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5,
        help="Save a trainer RESUME checkpoint every N optimizer steps. "
        "save_total_limit=1 keeps only the newest (for resume-from-latest); "
        "the two best-by-eval checkpoints are preserved separately under "
        "<output_dir>/best/. For the 3k run (500 steps) eval_steps=75 is "
        "the 15%-of-progress cadence.",
    )
    parser.add_argument("--ablation", choices=["none", "unmasked"], default="none")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--trace_log_path", default=None)
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help=(
            "Enable Qwen3/3.5 thinking mode during rollouts. OFF by default "
            "(thinking prose broke the tool-call parser for the plain agent). "
            "Turn ON for M1 — MEM1's mechanism requires the model to emit a "
            "state/think block each turn (that block is the memory carried "
            "forward after the history wipe). The parser strips the think "
            "block before tool-call parsing, so tool calls still work."
        ),
    )
    parser.add_argument(
        "--m1",
        action="store_true",
        help=(
            "Use the M1 (MEM1-style) system prompt + mem1_post_step_hook "
            "(rewrites conversation each turn to keep only the model's <state> "
            "block + compressed tool output). Implies --enable_thinking."
        ),
    )
    parser.add_argument(
        "--m2",
        action="store_true",
        help=(
            "Use the M2 (trained memory decisions) system prompt + "
            "memory_op_post_step_hook. Extends M1: the model also emits an "
            "explicit <decision:MEMORY_OP:ACTION_OP> token each turn "
            "(keep/drop/compress x search-again/answer-now), executed by the "
            "hook and rewarded by decision_reward (get_m2_reward). "
            "Backward-compatible with M1 — falls back to M1's compress "
            "behavior whenever no decision token is present, so an "
            "undertrained M2 policy runs exactly like M1. Implies --m1 and "
            "--enable_thinking. Was conditional on M1 clearing its "
            "≥30%-token-cut / no-EM-loss bar (rag_plan_s2.md §7) — M1's "
            "collapse was traced to a rollout-loop stall bug and fixed (see "
            "--force_action_on_stall), so M2 is unblocked; recommended ON "
            "together with --force_action_on_stall and --combined-style "
            "termination dominance (get_m2_reward builds on the combined "
            "stack, not raw T3)."
        ),
    )
    parser.add_argument(
        "--force_final_answer",
        action="store_true",
        help=(
            "Search-R1 pattern: if the model never emitted an <answer> tag, "
            "append a 'now answer' user turn and generate once more. Guarantees "
            "answer-tag production so the reward has variance for GRPO. "
            "Recommended ON for small models that don't self-transition from "
            "searching to answering (verified zero-shot for Qwen3.5-4B)."
        ),
    )
    parser.add_argument(
        "--force_action_on_stall",
        action="store_true",
        help=(
            "Mid-trajectory nudge: if a turn produces neither a tool call nor "
            "an <answer> tag (a 'stall' — e.g. M1's model writing only a "
            "<state> block and stopping), append a 'take an action now' user "
            "turn and generate once more, at the POINT of stall rather than "
            "waiting for --force_final_answer at the very end. Fixes the M1 "
            "collapse where a stall after turn 1 stranded the trajectory with "
            "unused step budget. Recommended ON whenever --m1 is used."
        ),
    )
    parser.add_argument(
        "--t3",
        action="store_true",
        help=(
            "Use the T3 reward stack (necessity + frugality) instead of the "
            "Phase-1 stack (search_usage). T3 splits the old search_usage into "
            "two primitives: necessity (should-I-search-at-all, IKEA r_kb) + "
            "frugality (how-many-searches, FrugalRAG). The logged variant also "
            "includes termination (answer-tag signal) for GRPO variance — "
            "without it, all-looping groups have zero variance. See "
            "rag/rewards/t3_rewards.py get_logged_t3_reward. NOTE: at 150 "
            "steps / 40 questions this stack let the policy collapse into "
            "looping-without-answering (termination weight too diluted by "
            "necessity+frugality) — see --combined for the fix."
        ),
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help=(
            "Use the combined reward stack: phase-1's termination dominance "
            "(format+termination=0.4, the ratio that fixed the original "
            "loop-without-answering collapse) plus T3's necessity+frugality "
            "as smaller supplementary terms (0.1 each). Fixes the mode "
            "collapse seen with --t3 at longer step counts (m1_t3_long: "
            "EM=0.000, 40/40 no answer tag). Mutually exclusive with --t3. "
            "See rag/rewards/t3_rewards.py get_logged_combined_reward."
        ),
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help=(
            "Enable T1 curriculum sampling: order questions easy→hard by "
            "solve_difficulty (T1 prober) + retrieval_difficulty (GRADE), pace "
            "them across training steps, and re-probe mid-training to retire "
            "mastered examples (contamination gate). See "
            "rag/synthesis/curriculum.py CurriculumSampler."
        ),
    )
    parser.add_argument(
        "--curriculum_labels",
        default=None,
        help=(
            "Path to a JSON file with T1 solve_difficulty labels (output from "
            "run_t1_t4_probe.py). Required when --curriculum is set and you "
            "want precomputed labels. If not provided, all questions get "
            "solve_difficulty=1.0 (curriculum orders by retrieval_difficulty only)."
        ),
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help=(
            "Generate rollout completions via TRL's colocated vLLM instead of "
            "HF generate(). Faster generation. Still runs the full agentic "
            "tool-calling loop with env_mask masking — the rollout_func is "
            "called with the trainer, and _execute_trajectory takes its "
            "use_vllm branch (trl.experimental.openenv.generate_rollout_"
            "completions over trainer.vllm_generation). Requires vLLM + a "
            "GPU with enough VRAM to colocate the inference engine with the "
            "training model (the 0.6B smoke fits in 16GB; larger models may "
            "need --vllm_mode server)."
        ),
    )
    parser.add_argument(
        "--vllm_mode",
        choices=["colocate", "server"],
        default="colocate",
        help="TRL vLLM mode. 'colocate' runs the vLLM engine in-process (default, "
        "single-GPU). 'server' connects to an external vLLM server.",
    )
    parser.add_argument(
        "--vllm_server_base_url",
        type=str,
        default=None,
        help=(
            "Base URL of the external vLLM server (--vllm_mode server). e.g. "
            "http://localhost:8000. TRL falls back to http://{host}:{port} if unset."
        ),
    )
    parser.add_argument(
        "--vllm_max_model_length",
        type=int,
        default=2048,
        help=(
            "vLLM engine max sequence length (prompt+completion). Default 2048 "
            "is ample for RAG (max_completion_length default 1024 + a few hundred "
            "prompt tokens). Lowering this from the model's default (e.g. 40960 "
            "for Qwen3) shrinks the KV cache so the colocated vLLM engine fits "
            "alongside the training model on a memory-constrained GPU — without "
            "it vLLM raises 'KV cache needed is larger than available memory'."
        ),
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.45,
        help=(
            "Fraction of GPU memory vLLM's colocated engine may reserve. Default "
            "0.45 leaves the majority of VRAM for the training model + LoRA + "
            "activations on a 16GB single GPU; raise on larger GPUs."
        ),
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help=(
            "Number of GPUs the colocated vLLM engine is sharded across "
            "(TRL `vllm_tensor_parallel_size`, colocate mode only). Set to 2 on "
            "a 2-GPU box so generation splits across both GPUs; keep 1 for "
            "single-GPU. Must match the number of GPUs visible to the trainer."
        ),
    )
    parser.add_argument(
        "--device_map",
        choices=["single_gpu", "auto"],
        default="single_gpu",
        help=(
            "Placement for the trainer's own model (the one actually used for "
            "generation + the backward pass — see build_rollout_fn's docstring-"
            "comment for why the rollout engine's separate model copy is forced "
            "to CPU regardless of this flag). 'single_gpu' (default) pins it to "
            "cuda:0 via device_map={'':0}. On an 8GB single-GPU box, "
            "device_map='auto' was observed to silently offload some layers to "
            "CPU (a 'parameters are on the meta device' warning) and corrupt "
            "generation into NaN/inf logits — use 'auto' only on a multi-GPU box."
        ),
    )
    args = parser.parse_args()
    resolved_device_map = {"": 0} if args.device_map == "single_gpu" else "auto"

    # FinDER uses OR-semantics BM25 (standard ranking): its analyst queries are
    # abstract and their tokens don't all literally appear in the evidence, so
    # AND-semantics returns empty 90%+ of the time (measured). HotpotQA keeps
    # the original AND semantics (Sprint-2/3 behaviour unchanged). 'mixed' keeps
    # OR semantics (same abstract-query profile as FinDER).
    backend = make_backend(
        args.backend, args.index_dir, match_all=(args.dataset not in ("finder", "mixed", "cuad"))
    )
    tools = [SearchCorpusTool(backend, top_k=8), ReadDocumentTool(backend)]

    # Backend-aware system prompt: BM25 (sqlite) needs short keyword queries;
    # dense (chroma) matches on semantics so natural-language queries work.
    # --m1 swaps in the MEM1-style prompt (asks the model to emit a <state>
    # block each turn) and implies --enable_thinking. --m2 implies --m1 (adds
    # the decision-token instruction on top of the state instruction).
    # --dataset finder swaps in the finance-flavoured FinDER prompt instead
    # (M1/M2 prompts are HotpotQA-shaped; E1 doesn't use them — E6 is deferred).
    args.m1 = args.m1 or args.m2
    enable_thinking = args.enable_thinking or args.m1
    if args.dataset in ("finder", "mixed", "cuad"):
        from agenttune.rag.data.finder import get_finder_system_prompt

        # 'mixed': each row's `prompt` already carries its OWN system prompt
        # (CUAD rows carry the legal prompt from the synthesis; FinDER rows get
        # the finance prompt at dataset-build). This is the generic fallback
        # used only when a raw-string prompt reaches the rollout engine.
        system_prompt = get_finder_system_prompt(args.backend)
    else:
        system_prompt = get_system_prompt(args.backend, m1=args.m1, m2=args.m2)

    # Memory post_step_hook: --m2's memory_op_post_step_hook executes the
    # model's chosen memory operation, falling back to M1's compress when no
    # decision token is present. --m1 alone always compresses. None (full
    # history) for the plain agent.
    post_step_hook = None
    if args.m2:
        from agenttune.rag.memory import memory_op_post_step_hook as _m2_hook

        post_step_hook = _m2_hook
    elif args.m1:
        from agenttune.rag.memory import mem1_post_step_hook as _m1_hook

        post_step_hook = _m1_hook

    # Each branch builds `train_dataset`/`eval_dataset` (HF Datasets with
    # prompt/gold_answer/question_id/gold_chunk_ids/domain/optimal_search_count)
    # plus question-TEXT-keyed maps for the TraceLogger + PeriodicEvalCallback.
    # `domain_by_question` is None for hotpotqa (no per-domain correctness).
    if args.dataset == "finder":
        from agenttune.rag.data.finder import (
            load_finder_split_rows,
            load_gold_chunk_map,
            to_grpo_dataset_finder,
        )

        train_rows, val_rows = load_finder_split_rows(args.index_dir)
        train_split = train_rows[: args.train_size]
        eval_split = val_rows[: args.eval_size]
        gold_chunk_map = load_gold_chunk_map(args.index_dir)
        train_dataset = to_grpo_dataset_finder(
            train_split, gold_chunk_map, system_prompt=system_prompt
        )
        eval_dataset = to_grpo_dataset_finder(
            eval_split, gold_chunk_map, system_prompt=system_prompt
        )
        gold_by_question = {row["text"]: row["answer"] for row in train_split}
        gold_by_question.update({row["text"]: row["answer"] for row in eval_split})
        gold_chunks_by_question = {
            row["text"]: gold_chunk_map.get(row["_id"], []) for row in train_split
        }
        gold_chunks_by_question.update(
            {row["text"]: gold_chunk_map.get(row["_id"], []) for row in eval_split}
        )
        domain_by_question = {q: "finder" for q in gold_by_question}
        print(
            f"[train_grpo] FinDER: {len(train_split)} train / {len(eval_split)} eval rows "
            f"(ticker-level split from {args.index_dir}/splits.json)"
        )
    elif args.dataset == "mixed":
        if not args.cuad_dataset:
            raise ValueError(
                "--dataset mixed requires --cuad_dataset (and a combined "
                "index built by build_index_mixed.py)."
            )
        from agenttune.rag.data.cuad import load_cuad_grpo_rows, to_grpo_dataset_cuad
        from agenttune.rag.data.finder import (
            load_finder_split_rows,
            load_gold_chunk_map,
            to_grpo_dataset_finder,
        )
        from agenttune.rag.trajectory_utils import extract_question_text as _qtext

        finder_train, finder_val = load_finder_split_rows(args.index_dir)
        gold_chunk_map = load_gold_chunk_map(args.index_dir)
        finder_train = finder_train[: args.train_size]
        finder_val = finder_val[: args.eval_size]
        finder_ds = to_grpo_dataset_finder(
            finder_train, gold_chunk_map, system_prompt=system_prompt
        )
        finder_eval_ds = to_grpo_dataset_finder(
            finder_val, gold_chunk_map, system_prompt=system_prompt
        )

        cuad_rows = [
            r for r in load_cuad_grpo_rows(args.cuad_dataset) if r.get("answerable") is not False
        ]
        cuad_train_rows = cuad_rows[: args.train_size]
        if args.cuad_eval_dataset:
            cuad_eval_rows = [
                r
                for r in load_cuad_grpo_rows(args.cuad_eval_dataset)
                if r.get("answerable") is not False
            ][: args.eval_size]
        else:
            cuad_eval_rows = cuad_rows[args.train_size : args.train_size + args.eval_size]
        cuad_ds = to_grpo_dataset_cuad(cuad_train_rows)
        cuad_eval_ds = to_grpo_dataset_cuad(cuad_eval_rows)

        def _interleave(a: list, b: list) -> list:
            """Strict round-robin so every GRPO batch samples ~50/50 domains."""
            out, n = [], max(len(a), len(b))
            for i in range(n):
                if i < len(a):
                    out.append(a[i])
                if i < len(b):
                    out.append(b[i])
            return out

        from datasets import Dataset as _HF_Dataset

        train_dataset = _HF_Dataset.from_list(_interleave(list(finder_ds), list(cuad_ds)))
        eval_dataset = _HF_Dataset.from_list(_interleave(list(finder_eval_ds), list(cuad_eval_ds)))
        gold_by_question, gold_chunks_by_question, domain_by_question = {}, {}, {}
        for ds in (finder_ds, finder_eval_ds, cuad_ds, cuad_eval_ds):
            for row in ds:
                q = _qtext(row["prompt"])
                gold_by_question[q] = row["gold_answer"]
                gold_chunks_by_question[q] = row["gold_chunk_ids"]
                domain_by_question[q] = row["domain"]
        print(
            f"[train_grpo] mixed: {len(finder_train)} finder train / {len(finder_val)} finder eval + "
            f"{len(cuad_train_rows)} cuad train / {len(cuad_eval_rows)} cuad eval = "
            f"{len(train_dataset)} train / {len(eval_dataset)} eval rows"
        )
    elif args.dataset == "cuad":
        # CUAD-only: synthetic nllp-cuad-multihop GRPO rows (our derived data, NOT
        # raw CUAD). Same columns as FinDER (prompt/gold_answer/question_id/
        # gold_chunk_ids/domain/optimal_search_count); each row's `prompt` already
        # carries the legal-register CUAD system prompt baked in at synthesis.
        if not args.cuad_dataset:
            raise ValueError(
                "--dataset cuad requires --cuad_dataset (the synthetic "
                "nllp-cuad-multihop GRPO dataset, e.g. dataset_train_grpo.jsonl)."
            )
        from agenttune.rag.data.cuad import load_cuad_grpo_rows, to_grpo_dataset_cuad
        from agenttune.rag.trajectory_utils import extract_question_text as _qtext

        cuad_rows = [
            r for r in load_cuad_grpo_rows(args.cuad_dataset) if r.get("answerable") is not False
        ]
        cuad_train_rows = cuad_rows[: args.train_size]
        if args.cuad_eval_dataset:
            cuad_eval_rows = [
                r
                for r in load_cuad_grpo_rows(args.cuad_eval_dataset)
                if r.get("answerable") is not False
            ][: args.eval_size]
        else:
            cuad_eval_rows = cuad_rows[args.train_size : args.train_size + args.eval_size]
        train_dataset = to_grpo_dataset_cuad(cuad_train_rows)
        eval_dataset = to_grpo_dataset_cuad(cuad_eval_rows)
        gold_by_question, gold_chunks_by_question, domain_by_question = {}, {}, {}
        for ds in (train_dataset, eval_dataset):
            for row in ds:
                q = _qtext(row["prompt"])
                gold_by_question[q] = row["gold_answer"]
                gold_chunks_by_question[q] = row["gold_chunk_ids"]
                domain_by_question[q] = row["domain"]
        print(
            f"[train_grpo] cuad-only: {len(cuad_train_rows)} train / {len(cuad_eval_rows)} eval rows "
            f"(synthetic nllp-cuad-multihop, answerable only)"
        )
    else:
        train_split, eval_split = load_hotpotqa_splits(
            config=args.hotpotqa_config, train_size=args.train_size, eval_size=args.eval_size
        )
        gold_chunks_by_question = None
        domain_by_question = None

    # T1 curriculum: order questions easy→hard, pace across steps, retire mastered.
    # (HotpotQA-only today — FinDER rows key the question as "text", and the
    # E1 plan runs curriculum only as the E5 appendix ablation on HotpotQA-style
    # labels; wire --curriculum_labels from a FinDER probe before using here.)
    curriculum_sampler = None
    if args.curriculum:
        if args.dataset in ("finder", "mixed", "cuad"):
            raise NotImplementedError(
                "--curriculum is not wired for --dataset finder/mixed/cuad yet (E5)."
            )
        import json as _json

        from agenttune.rag.synthesis import CurriculumSampler, build_curriculum

        questions = [{"question": r["question"], "answer": r["answer"]} for r in train_split]
        # Load T1 labels if provided
        solve_labels = None
        if args.curriculum_labels and os.path.exists(args.curriculum_labels):
            with open(args.curriculum_labels) as f:
                label_data = _json.load(f)
            solve_labels = (
                label_data.get("labels", label_data) if isinstance(label_data, dict) else label_data
            )
            print(
                f"[train_grpo] loaded {len(solve_labels)} T1 difficulty labels from {args.curriculum_labels}"
            )
        curriculum = build_curriculum(questions, solve_difficulty_labels=solve_labels)
        curriculum_sampler = CurriculumSampler(
            curriculum,
            max_steps=args.max_steps,
            batch_size=args.per_device_train_batch_size * args.gradient_accumulation_steps,
        )
        print(
            f"[train_grpo] curriculum enabled: {len(curriculum)} questions, "
            f"warmup={curriculum_sampler.warmup_steps} steps, "
            f"full={curriculum_sampler.full_steps} steps"
        )

    if args.dataset == "hotpotqa":
        train_dataset = to_grpo_dataset(train_split, system_prompt=system_prompt)
        eval_dataset = to_grpo_dataset(eval_split, system_prompt=system_prompt)
        gold_by_question = {row["question"]: row["answer"] for row in train_split}
        gold_by_question.update({row["question"]: row["answer"] for row in eval_split})

    trace_path = args.trace_log_path or os.path.join(args.output_dir, "trace.jsonl")
    # Use the logged reward fn so per-component scores are captured for the
    # trace + per-step summary. FinDER gets its own assembled stack
    # (finder_rewards: format+termination+numeric correctness+golden-chunk
    # recall+conciseness+frugality — FINNLP_EXPERIMENTS v2 §3); --t3 switches
    # to the T3 stack (necessity + frugality + termination + correctness +
    # format); default is Phase-1 (format + termination + search_usage +
    # correctness).
    if args.dataset in ("finder", "mixed", "cuad") and not (args.t3 or args.combined or args.m2):
        from agenttune.rag.rewards.finder_rewards import (
            get_last_finder_component_scores,
            get_logged_finder_reward,
        )

        reward_fn = get_logged_finder_reward()
        print(
            "[train_grpo] reward stack: FinDER/mixed/CUAD "
            "(format(0.1)+termination(0.3)+correctness_numeric(0.6)+"
            "golden_chunk_recall(0.8)+conciseness(0.15)+frugality(0.25)); "
            "chunk-level GCR; per-domain correctness (finance numeric, CUAD relaxed-F1)"
        )
        trace_logger = TraceLogger(
            trace_path,
            gold_by_question,
            reward_fn,
            component_getter=get_last_finder_component_scores,
            gold_chunks_by_question=gold_chunks_by_question,
            domain_by_question=domain_by_question,
        )
    elif args.m2:
        from agenttune.rag.rewards.t3_rewards import (
            get_last_m2_component_scores,
            get_logged_m2_reward,
        )

        reward_fn = get_logged_m2_reward()
        print("[train_grpo] reward stack: M2 (combined + decision(0.05))")
        trace_logger = TraceLogger(
            trace_path,
            gold_by_question,
            reward_fn,
            component_getter=get_last_m2_component_scores,
            gold_chunks_by_question=gold_chunks_by_question,
        )
        from agenttune.rag.rewards.t3_rewards import (
            get_last_combined_component_scores,
            get_logged_combined_reward,
        )

        reward_fn = get_logged_combined_reward()
        print(
            "[train_grpo] reward stack: Combined (format+termination(0.3)+correctness+necessity(0.1)+frugality(0.1))"
        )
        trace_logger = TraceLogger(
            trace_path,
            gold_by_question,
            reward_fn,
            component_getter=get_last_combined_component_scores,
        )
    elif args.t3:
        from agenttune.rag.rewards.t3_rewards import (
            get_last_t3_component_scores,
            get_logged_t3_reward,
        )

        reward_fn = get_logged_t3_reward()
        print("[train_grpo] reward stack: T3 (format+termination+correctness+necessity+frugality)")
        trace_logger = TraceLogger(
            trace_path, gold_by_question, reward_fn, component_getter=get_last_t3_component_scores
        )
    else:
        from agenttune.rag.rewards.phase1_rewards import get_logged_training_reward

        reward_fn = get_logged_training_reward()
        print("[train_grpo] reward stack: Phase-1 (format+termination+search_usage+correctness)")
        trace_logger = TraceLogger(trace_path, gold_by_question, reward_fn)

    rollout_fn = build_rollout_fn(
        model_path=args.model,
        tools=tools,
        max_steps=args.max_rollout_steps,
        ablation=args.ablation,
        on_trajectory_end=trace_logger,
        system_prompt=system_prompt,
        enable_thinking=enable_thinking,
        force_final_answer=args.force_final_answer,
        force_action_on_stall=args.force_action_on_stall,
        post_step_hook=post_step_hook,
    )

    from peft import LoraConfig

    # NOTE: `tools` is deliberately NOT passed here. TRL's native GRPOTrainer
    # forwards a `tools=[...]` kwarg into its own experimental tool-calling
    # glue, which expects plain named callables (it indexes by `tool.__name__`)
    # — not our BaseTool instances. Since `rollout_func` already wires the
    # tools into agenttune's own `_execute_trajectory` loop (the masking-aware
    # path this whole design depends on), passing `tools=` again here would
    # only trip that unrelated, unused TRL code path.
    #
    # Explicitly pass a plain AutoTokenizer as processing_class. WITHOUT this,
    # GRPOTrainer auto-creates a processor from the model — and for Qwen3.5 it
    # picks Qwen3VLProcessor (a vision-language processor), which lacks
    # pad_token_id/eos_token_id directly and crashes the rollout's _gen. A plain
    # AutoTokenizer (Qwen2Tokenizer for Qwen3.5) has both. See rollout_factory.py
    # _gen's pad_token_id/eos_token_id access.
    from transformers import AutoTokenizer

    processing_class = AutoTokenizer.from_pretrained(args.model)
    if processing_class.pad_token is None:
        processing_class.pad_token = processing_class.eos_token

    # Sprint 2: custom callback to log per-component reward means + answer-tag
    # rate each step to tensorboard, and print a one-line step summary so the
    # console shows reward signal health live (not just loss).
    #
    # BUG FOUND 2026-08-08 (E1 seed42 30-step run): this callback and the
    # enable_input_require_grads() call below used to run AFTER
    # create_agentic_trainer(...) returned, reading `trainer.trainer` to reach
    # the underlying HF Trainer. But TrlAgenticGrpo.__init__ only stores
    # kwargs — it builds `self.trainer` inside `.train()` -> `setup_trainer()`,
    # which hasn't run yet at that point. So `trainer.trainer` was always
    # `None`: enable_input_require_grads() silently no-op'd (gradient
    # checkpointing likely ran with reduced effectiveness the whole time, not
    # just here) and add_callback() never attached the RewardSignalLogger — no
    # `[step N]` console lines and no `reward/<component>` tensorboard tags on
    # any run to date, including today's E1 30-step validation run. The
    # trace.jsonl-based logging (TraceLogger, computed independently via
    # reward_fn) was NOT affected — this bug only killed the *duplicate*
    # tensorboard/console view, not the actual training reward signal grad_norm
    # was already confirmed nonzero via a different code path (RewardSignalLogger
    # doesn't touch the reward computation, only re-reports it).
    #
    # Fix: GRPOTrainer's own __init__ accepts `callbacks=[...]` directly
    # (confirmed via introspection) and `_split_kwargs` already routes any
    # recognised key straight through — so pass the callback at CONSTRUCTION
    # time instead of trying to attach it after. `on_train_begin` (which DOES
    # receive `model=` in kwargs, per CallbackHandler.call_event) replaces the
    # old post-construction enable_input_require_grads() call.
    from transformers import TrainerCallback

    if args.dataset in ("finder", "mixed", "cuad") and not (args.t3 or args.combined or args.m2):
        from agenttune.rag.rewards.finder_rewards import (
            get_last_finder_component_scores as get_last_scores,
        )
    elif args.combined:
        from agenttune.rag.rewards.t3_rewards import (
            get_last_combined_component_scores as get_last_scores,
        )
    elif args.t3:
        from agenttune.rag.rewards.t3_rewards import get_last_t3_component_scores as get_last_scores
    else:
        from agenttune.rag.rewards.phase1_rewards import (
            get_last_component_scores as get_last_scores,
        )

    class RewardSignalLogger(TrainerCallback):
        def __init__(self):
            # Lazy SummaryWriter, separate from TensorBoardCallback's own —
            # see on_log's docstring-comment for why a second writer is
            # needed rather than mutating the shared `logs` dict.
            self._tb_writer = None

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            # See gradient_checkpointing=True's docstring-comment below: the
            # PEFT caveat this fixes requires the model object, which is only
            # available once GRPOTrainer has actually been constructed — this
            # hook fires right before the training loop starts, after that.
            if model is not None and hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

            # Point our SummaryWriter at the SAME directory TensorBoardCallback
            # just created for its own writer, so both sets of scalars land in
            # one `tensorboard --logdir` view. TensorBoardCallback resolves its
            # dir as `os.path.join(args.output_dir, default_logdir())` where
            # default_logdir() is `runs/<timestamp>_<host>` — non-deterministic
            # per call (confirmed: two calls a couple seconds apart differ), so
            # recomputing it ourselves would very likely pick a DIFFERENT dir
            # (verified failure mode: an earlier version of this fix wrote to
            # a bare `SummaryWriter(log_dir=args.logging_dir)`, which is None
            # at runtime — `logging_dir` is deprecated — and silently fell
            # back to `./runs/...` under the process CWD, nowhere near the run
            # dir at all). Default callbacks (including TensorBoardCallback)
            # fire BEFORE user callbacks, so its dir already exists when this
            # method runs — glob for it instead of re-resolving independently.
            if args.report_to and "tensorboard" in args.report_to:
                candidates = sorted(
                    glob.glob(os.path.join(args.output_dir, "runs", "*")),
                    key=os.path.getmtime,
                )
                if candidates:
                    from torch.utils.tensorboard import SummaryWriter

                    self._tb_writer = SummaryWriter(log_dir=candidates[-1])

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            comp = get_last_scores()
            comp_means = {}
            if comp:
                import statistics

                for name, vals in comp.items():
                    if vals:
                        comp_means[f"reward/{name}"] = statistics.mean(vals)

            # FOUND 2026-08-08 (callback_fix_check smoke): mutating `logs` here
            # does NOT reach training_log.json OR tensorboard, for two
            # independent reasons — (1) HF Trainer.log() does
            # `state.log_history.append({**logs, "step": ...})` BEFORE calling
            # on_log(), so the append is a snapshot copy made before this
            # method ever runs; (2) report_to="tensorboard"'s TensorBoardCallback
            # is a DEFAULT callback (Trainer.__init__: "callbacks = default_
            # callbacks + callbacks"), so it fires — and writes+flushes to its
            # own SummaryWriter — BEFORE any user callback in the chain, using
            # the very `logs` dict this method is about to mutate. Both are
            # confirmed by reading transformers/trainer.py + integration_utils.py
            # directly (not documented anywhere). Fix: write to state.log_history's
            # already-appended entry directly (mutable in place — training_log.json
            # is dumped straight from this list by main() below) for the JSON
            # artifact, and maintain a SECOND independent SummaryWriter pointed
            # at the same run's tensorboard dir for live viewing.
            if comp_means and state.log_history:
                state.log_history[-1].update(comp_means)
            if comp_means and self._tb_writer is not None:
                for k, v in comp_means.items():
                    self._tb_writer.add_scalar(k, v, state.global_step)
                self._tb_writer.flush()

            # Print a compact step summary.
            step = state.global_step
            r = logs.get("rewards", logs.get("reward", logs.get("reward_mean")))
            loss = logs.get("loss")
            print(
                f"[step {step}] loss={loss:.4f} "
                if isinstance(loss, int | float)
                else f"[step {step}] loss={loss} "
                f"reward={r if r is not None else '?'} "
                f"kl={logs.get('kl', '?')} "
                f"grad_norm={logs.get('grad_norm', '?')} "
                f"comp={ {k: round(v, 3) for k, v in comp_means.items()} if comp_means else {} }"
            )

    class PeriodicEvalCallback(TrainerCallback):
        """Cheap in-training eval: EM/F1, gold-chunk recall (reference- +
        chunk-level), unique-query ratio, search count/tokens — everything
        computable from a rollout + the dataset's own gold_answer/
        gold_chunk_ids columns, no API key, no LLM judge. Judge accuracy and
        citation precision are deliberately NOT here — FINNLP_EXPERIMENTS.md
        §4/E0 scopes those as a separate 1-2 day API/CPU pass, not part of the
        training loop (an LLM-judge call every N steps would add real
        latency+cost to every checkpoint of a multi-hour run for a metric the
        paper doesn't need until the final eval table).

        Runs on `state.global_step % eval_steps == 0`, sampling up to
        `sample_size` questions from `eval_dataset` (deterministic — same
        first-K each time, so evals across steps are comparable, not just
        across seeds). Uses a fresh TransformersRolloutEngine wrapping the
        LIVE trainer.model (LoRA weights as of this step) — no reload, no
        extra disk I/O — and reuses the SAME reward_fn `train_grpo` already
        built (so eval numbers use the identical FinDER numeric-tolerance
        correctness the training reward does, not a different EM path).
        """

        def __init__(
            self,
            eval_dataset,
            sample_size: int,
            eval_steps: int,
            reward_fn: Callable,
            tools: list,
            max_rollout_steps: int,
            system_prompt: str,
            enable_thinking: bool,
            force_final_answer: bool,
            has_gold_chunks: bool,
            context_length: int | None = None,
        ):
            self.eval_rows = eval_dataset.select(range(min(sample_size, len(eval_dataset))))
            self.eval_steps = eval_steps
            self.reward_fn = reward_fn
            self.tools = tools
            self.max_rollout_steps = max_rollout_steps
            self.system_prompt = system_prompt
            self.enable_thinking = enable_thinking
            self.force_final_answer = force_final_answer
            self.has_gold_chunks = has_gold_chunks
            # Cap the eval rollout's conversation to the same context window the
            # training path truncates to (vLLM max_model_length). Without this the
            # eval engine (trainer=None -> no length guard) lets retrieval-heavy
            # conversations grow to ~50K tokens and OOMs on 9B x 150K vocab.
            self.context_length = context_length
            # (step, eval_score) history used to prune local checkpoints to the
            # two BEST by eval quality (see on_log). The HF Hub retains all.
            self._eval_scores: list = []
            self._best: list = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            step = state.global_step
            if self.eval_steps <= 0 or step == 0 or step % self.eval_steps != 0:
                return
            model = kwargs.get("model")
            processing_class = kwargs.get("processing_class")
            if model is None or processing_class is None:
                return

            from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

            eval_rollout_fn = create_rollout_fn(
                rollout_backend="transformers",
                model=model,
                tokenizer=processing_class,
                tools=self.tools,
                max_steps=self.max_rollout_steps,
                system_prompt=self.system_prompt,
                enable_thinking=self.enable_thinking,
                force_final_answer=self.force_final_answer,
                context_length=self.context_length,
            )
            prompts = [row["prompt"] for row in self.eval_rows]
            was_training = model.training
            model.eval()
            try:
                result = eval_rollout_fn(prompts)
            finally:
                if was_training:
                    model.train()

            gold_answers = [row["gold_answer"] for row in self.eval_rows]
            tool_call_counts = result.get("tool_call_counts", [0] * len(prompts))
            reward_kwargs = {
                "completions": [t.final_response for t in result["trajectories"]],
                "prompts": prompts,
                "gold_answer": gold_answers,
                "tool_call_counts": tool_call_counts,
            }
            if "domain" in (self.eval_rows.column_names or []):
                reward_kwargs["domain"] = [row.get("domain", "finder") for row in self.eval_rows]
            if self.has_gold_chunks:
                reward_kwargs["retrieved_chunk_ids"] = result.get(
                    "retrieved_chunk_ids", [[] for _ in prompts]
                )
                reward_kwargs["gold_chunk_ids"] = [row["gold_chunk_ids"] for row in self.eval_rows]
            _ = self.reward_fn(**reward_kwargs)  # populates get_last_*_component_scores()

            from agenttune.rag.rewards.qa_metrics import (
                exact_match_score,
                extract_answer_tag,
                f1_score,
            )

            em_scores, f1_scores = [], []
            unique_ratios = []
            for traj, gold in zip(result["trajectories"], gold_answers, strict=False):
                pred = extract_answer_tag(traj.final_response)
                em_scores.append(exact_match_score(pred, str(gold)))
                f1_scores.append(f1_score(pred, str(gold)))
                queries = [
                    str(call.get("arguments", {}).get("query", call.get("arguments", "")))
                    for s in traj.steps
                    for call in extract_tool_calls(s)
                ]
                if queries:
                    normed = [q.strip().lower() for q in queries]
                    unique_ratios.append(len(set(normed)) / len(normed))

            import statistics

            n = len(prompts)
            eval_metrics = {
                "eval/em": statistics.mean(em_scores) if em_scores else 0.0,
                "eval/f1": statistics.mean(f1_scores) if f1_scores else 0.0,
                "eval/mean_searches": (
                    statistics.mean(tool_call_counts) if tool_call_counts else 0.0
                ),
                "eval/unique_query_ratio": statistics.mean(unique_ratios) if unique_ratios else 0.0,
                "eval/n_questions": n,
            }
            if self.has_gold_chunks:
                comp = get_last_scores()
                for name in ("golden_chunk_recall", "golden_chunk_recall_chunklevel"):
                    if comp.get(name):
                        eval_metrics[f"eval/{name}"] = statistics.mean(comp[name])

            if state.log_history:
                state.log_history[-1].update(eval_metrics)
            print(
                f"[eval step {step}] n={n} EM={eval_metrics['eval/em']:.3f} "
                f"F1={eval_metrics['eval/f1']:.3f} "
                f"searches={eval_metrics['eval/mean_searches']:.2f} "
                + (
                    f"chunk_recall={eval_metrics.get('eval/golden_chunk_recall', 0):.3f} "
                    if self.has_gold_chunks
                    else ""
                )
                + f"echo={eval_metrics['eval/unique_query_ratio']:.3f}"
            )

            # Keep the two BEST local checkpoints as copies under <output_dir>/best/
            # (by eval/F1). The HF Hub (HubPushCallback) retains every checkpoint;
            # locally the trainer keeps ONE resume checkpoint (save_total_limit=1)
            # and we additionally preserve the two best-by-eval. `best/` is nested
            # so HF's checkpoint rotation (globs checkpoint-*) never touches it.
            score = float(eval_metrics.get("eval/f1", 0.0))
            self._eval_scores.append((step, score))
            ckpt_dir = getattr(args, "output_dir", None)
            src = os.path.join(ckpt_dir, f"checkpoint-{step}") if ckpt_dir else None
            if ckpt_dir and src and os.path.isdir(src):
                import glob
                import shutil

                ckpts = sorted(
                    glob.glob(os.path.join(ckpt_dir, "checkpoint-*")),
                    key=lambda p: int(p.rsplit("-", 1)[1]),
                )
                # (score, checkpoint_path) for all ckpts with a known eval score
                scored = []
                for c in ckpts:
                    cstep = int(c.rsplit("-", 1)[1])
                    s = next((s for st, s in self._eval_scores if st == cstep), None)
                    if s is not None:
                        scored.append((s, c))
                if scored:
                    self._best = sorted(scored, key=lambda x: x[0], reverse=True)[:2]
                    best_dir = os.path.join(ckpt_dir, "best")
                    os.makedirs(best_dir, exist_ok=True)
                    for i, (s, c) in enumerate(self._best, 1):
                        dst = os.path.join(best_dir, f"best-{i}")
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(c, dst)
                    print(
                        f"[eval step {step}] best-2 (F1={score:.3f}): "
                        f"{[os.path.basename(c) for _, c in self._best]} -> {best_dir}"
                    )

    class HubPushCallback(TrainerCallback):
        """Push the LoRA adapter to the HF Hub every `push_steps` optimizer
        steps, into a `<run_name>/checkpoint-<step>` subfolder — multiple
        runs (different seeds, different arms) can share ONE repo_id without
        colliding. Requires HF_TOKEN in the environment; never accepts a
        token as a constructor arg (keeps it out of process argv / any
        serialised manifest)."""

        def __init__(self, repo_id: str, push_steps: int, run_name: str, private: bool = True):
            self.repo_id = repo_id
            self.push_steps = push_steps
            # Prefixed by run_name (e.g. "e1_finder_seed42") so multiple
            # seeds/runs pushed to the SAME repo_id don't collide on
            # "checkpoint-50" — each run gets its own subtree.
            self.run_name = run_name
            self.private = private
            self._api = None

        def on_log(self, args, state, control, model=None, processing_class=None, **kwargs):
            step = state.global_step
            if self.push_steps <= 0 or step == 0 or step % self.push_steps != 0:
                return
            if model is None:
                return
            try:
                from huggingface_hub import HfApi

                if self._api is None:
                    self._api = HfApi()  # reads HF_TOKEN from env
                    self._api.create_repo(self.repo_id, private=self.private, exist_ok=True)
                import tempfile

                with tempfile.TemporaryDirectory() as tmp:
                    model.save_pretrained(tmp)
                    if processing_class is not None and hasattr(
                        processing_class, "save_pretrained"
                    ):
                        processing_class.save_pretrained(tmp)
                    # PEFT writes the LOCAL load path (`/root/qwen3.5-9b`) into
                    # README.md frontmatter — both the `base_model:` field and the
                    # `base_model:adapter:/root/...` tag — and the Hub rejects a
                    # non-model-id value there. Rewrite every occurrence to the
                    # canonical HF id so the adapter is loadable via
                    # `from_pretrained(...)`.
                    readme = os.path.join(tmp, "README.md")
                    if os.path.exists(readme):
                        txt = open(readme).read()
                        txt = txt.replace("/root/qwen3.5-9b", "Qwen/Qwen3.5-9B")
                        open(readme, "w").write(txt)
                    path_in_repo = f"{self.run_name}/checkpoint-{step}"
                    self._api.upload_folder(
                        repo_id=self.repo_id,
                        folder_path=tmp,
                        path_in_repo=path_in_repo,
                        commit_message=f"{self.run_name}: checkpoint at step {step}",
                    )
                print(f"[HubPushCallback] pushed {path_in_repo} -> {self.repo_id}")
            except Exception as e:
                # Never let a Hub hiccup (rate limit, transient network) kill
                # a multi-hour training run — log and keep training.
                print(f"[HubPushCallback] push at step {step} failed (non-fatal): {e}")

    training_callbacks: list = [RewardSignalLogger()]
    if args.eval_steps > 0:
        training_callbacks.append(
            PeriodicEvalCallback(
                eval_dataset=eval_dataset,
                sample_size=args.eval_sample_size,
                eval_steps=args.eval_steps,
                context_length=args.eval_context_length,
                reward_fn=reward_fn,
                tools=tools,
                max_rollout_steps=args.max_rollout_steps,
                system_prompt=system_prompt,
                enable_thinking=enable_thinking,
                force_final_answer=args.force_final_answer,
                has_gold_chunks=gold_chunks_by_question is not None,
            )
        )
    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--push_to_hub requires --hub_model_id.")
        if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            raise ValueError(
                "--push_to_hub requires HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) "
                "in the environment — never pass it as a CLI argument."
            )
        training_callbacks.append(
            HubPushCallback(
                repo_id=args.hub_model_id,
                push_steps=args.hub_push_steps,
                run_name=os.path.basename(args.output_dir.rstrip("/")),
                private=args.hub_private,
            )
        )

    # wandb: HF Trainer's WandbCallback reads WANDB_API_KEY from the
    # environment on first log() call — never pass a key as a CLI argument.
    # WANDB_PROJECT / run name are set via env vars here (the standard W&B
    # env-var contract) rather than a wandb.init() call, so this stays a thin
    # passthrough — report_to=["wandb", ...] is what actually attaches the
    # callback; TRL/HF wire it automatically, same as "tensorboard".
    if "wandb" in args.report_to:
        if not os.environ.get("WANDB_API_KEY"):
            raise ValueError(
                "--report_to wandb requires WANDB_API_KEY in the environment "
                "— never pass it as a CLI argument."
            )
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ["WANDB_NAME"] = args.wandb_run_name or os.path.basename(
            args.output_dir.rstrip("/")
        )
    resolved_report_to = [r for r in args.report_to if r != "none"] or "none"

    trainer = create_agentic_trainer(
        algorithm="grpo",
        model=args.model,
        reward_funcs=reward_fn,
        rollout_func=rollout_fn,
        processing_class=processing_class,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=training_callbacks,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_completion_length=args.max_completion_length,
        # Checkpoint/resume scheme: a RESUME checkpoint every `save_steps`
        # (default 5) with save_total_limit=1 so only the newest survives
        # (unambiguous resume-from-latest); the two best-by-eval checkpoints are
        # preserved separately by PeriodicEvalCallback under <output_dir>/best/.
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        resume_from_checkpoint=args.resume,
        # See --device_map help text: pins the trainer's own model load to a
        # single GPU via GRPOConfig's standard `model_init_kwargs` passthrough
        # (a real TRL/transformers field, confirmed via introspection — not a
        # framework change). Fixes the same accelerate device_map="auto" ->
        # meta-device-offload -> NaN-logits failure mode as the rollout engine
        # side, contained entirely to this script.
        model_init_kwargs={"device_map": resolved_device_map},
        # LoRA on target_modules="all-linear" still needs forward activations
        # through every layer to compute the LoRA-adapter gradients — i.e.
        # activation memory comparable to full fine-tuning even though the
        # optimizer state is LoRA-only. On an 8GB GPU this OOM'd during
        # backward() with only 338MiB missing (7.42/7.67GB used) at
        # num_generations=2/max_completion_length=512/batch_size=1 — already
        # about as low as those knobs can go before hurting signal quality.
        # gradient_checkpointing is the standard fix (trades recompute for
        # activation memory) and is a real HF TrainingArguments/GRPOConfig
        # field — no framework change, just another kwarg from this script.
        gradient_checkpointing=True,
        # Liger chunked-fused GRPO loss: on a 9B × 150K-vocab model the standard
        # loss forward materializes (num_generations × seq_len × 150K) fp32 logits
        # per micro-batch — with 6 rollouts × ~10K-token conversations that is a
        # ~36GB tensor with gradients, which OOMs the 96GB box even after the
        # context-window truncation. Liger computes the loss in chunks and never
        # materializes the full logits tensor. Compatible with our LoRA config
        # (target_modules="all-linear" excludes lm_head — required). The reference
        # forward is unaffected (it still uses the standard logps path, which is
        # chunked at per_device_train_batch_size and bounded by the truncation).
        use_liger_kernel=True,
        learning_rate=args.learning_rate,
        seed=args.seed,
        # HF Trainer's default logging_steps (500) is larger than most of our
        # runs' max_steps — without this, trainer.state.log_history below
        # would come back empty or with a single point, useless for a
        # reward/loss curve. Force per-step logging instead.
        logging_steps=1,
        # Sprint 2: log to TensorBoard so loss/reward/kl/grad_norm curves are
        # viewable live on port 8080 (tensorboard --logdir output_dir --port 8080).
        # Also writes to the trainer's own log_history for training_log.json.
        # --report_to wandb (see resolved_report_to above) adds W&B in
        # parallel — HF's WandbCallback reads WANDB_API_KEY from the
        # environment on its own; nothing else to wire here.
        report_to=resolved_report_to,
        # Log per-component reward means each step (format/termination/search/
        # correct) by reading the stashed scores after each reward batch. This
        # is surfaced via a custom callback below.
        logging_nan_filter=False,
        **(
            {
                "use_vllm": True,
                "vllm_mode": args.vllm_mode,
                "vllm_max_model_length": args.vllm_max_model_length,
                "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
                "vllm_server_base_url": args.vllm_server_base_url,
                # Sprint 2: disable vLLM importance-sampling correction. The IS
                # ratio = exp(old_logps - sampling_logps) was coming out exactly 0
                # (sampling/importance_sampling_ratio/mean=0, max=0) because the
                # rollout's sampling logprobs (from vLLM's openenv) don't perfectly
                # align with the prompt_ids+completion_ids TRL recomputes old_logps
                # over — the mismatch makes ratios exceed the clip max (3.0) and
                # get masked to 0 → loss=0 → grad_norm=0 → no learning. Disabling
                # IS correction makes TRL use old_per_token_logps (recomputed from
                # the model) directly, the standard GRPO setup. This is the fix
                # for the grad_norm=0 blocker (see Sprint2_readme §5.15).
                "vllm_importance_sampling_correction": False,
                # NOTE: vllm_enable_sleep_mode was tested (to free the engine's
                # ~23GB during the reference-KL forward, which otherwise OOMs at
                # num_generations 6) but crashes on engine wake with
                # "CUDA error: invalid argument" in buffer.data.copy_ on this
                # model. Reverted. num_generations is set to 4 so the reference
                # forward (4 × ~8K-token convs × 150K vocab × fp32 ≈ 19GB) fits
                # in the ~22GB free with the engine awake.
            }
            if args.use_vllm
            else {}
        ),
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )

    result = trainer.train()

    # `TrlAgenticGrpo.get_training_stats()`'s own "training_history" field is
    # always [] (never populated anywhere in agentic_grpo.py) — the real
    # per-logging-step record (loss, reward, kl, grad_norm, learning_rate,
    # epoch) lives on the underlying HF GRPOTrainer's `trainer.trainer.state
    # .log_history`. Save it explicitly so training stability/reward trends
    # can actually be inspected after the run.
    log_history = []
    inner_trainer = getattr(trainer, "trainer", None)
    if inner_trainer is not None and getattr(inner_trainer, "state", None) is not None:
        log_history = inner_trainer.state.log_history
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    manifest = {
        "model": args.model,
        "dataset": args.dataset,
        "backend": args.backend,
        "ablation": args.ablation,
        "output_dir": args.output_dir,
        "trace_log_path": trace_path,
        "training_log_path": os.path.join(args.output_dir, "training_log.json"),
        "training_stats": result,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "effective_prompts_per_step": args.per_device_train_batch_size
        * args.gradient_accumulation_steps,
        "approx_epochs": (
            args.max_steps
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            / len(train_dataset)
        ),
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "num_log_points": len(log_history),
        "report_to": resolved_report_to,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id if args.push_to_hub else None,
        "hub_push_steps": args.hub_push_steps if args.push_to_hub else None,
        "eval_steps": args.eval_steps,
        "eval_sample_size": args.eval_sample_size if args.eval_steps > 0 else None,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w") as f:
        # `manifest["training_stats"]` embeds `result["config_kwargs"]` verbatim
        # (TrlAgenticGrpo.get_training_stats()), which now includes the
        # RewardSignalLogger callback instance passed to create_agentic_trainer
        # (see the callbacks= fix above) — not JSON-serializable, and crashed
        # this write on every run after that fix (model/adapter/training_log
        # all save fine beforehand). default=str degrades any such object to
        # its repr instead of losing the whole manifest to an exception this
        # late in the run.
        json.dump(manifest, f, indent=2, default=str)

    print(f"[train_grpo] Done. output_dir={args.output_dir} result={result}")


if __name__ == "__main__":
    main()
