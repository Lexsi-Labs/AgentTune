"""
REAL agentic-RAG GRPO — the documented one-liner, actually run end-to-end.
=========================================================================

`create_agentic_trainer`'s GRPO path advertises a one-liner::

    trainer = create_agentic_trainer(
        "grpo", model=..., train_dataset=qa, tools=[search_tool], reward_funcs=[reward_fn])
    trainer.train()

This script runs it for real: a live `Qwen2.5-3B-Instruct` policy that actually
calls the `search_corpus` tool during the GRPO rollout, retrieves real BM25
passages, and is scored by the real composite RAG reward (correctness + search
usage), with a real LoRA GRPO step and the adapter saved + reloaded.

NOTE (2026-07-10): this script originally used `agenttune.rag.environment.RAGEnvironment`,
which was deleted when the RAG package was overlaid with a fuller retrieval/tools/rewards
implementation. Ported to the
current `agenttune.rag.retrieval.sqlite_fts.SQLiteFTSBackend` +
`agenttune.rag.tools.search_corpus.SearchCorpusTool` + `agenttune.rag.rewards.phase1_rewards`,
using the rollout output's own `tool_call_counts` (from `create_rollout_fn`) instead of
`RAGEnvironment._retrieval_from_trajectory`, which no longer exists either.

Search queries: `SQLiteFTSBackend` AND-matches every term in a query, so a full
natural-language question usually fails to retrieve (it contains words like
"which"/"is" absent from the source text) — see `_BM25_QUERY_GUIDANCE` in
`agenttune.rag.data.hotpotqa`. The system prompt below asks the model for a short
keyword query instead, per that documented convention.

HONEST SCOPE — read this before quoting a number
-------------------------------------------------
The goal here is that the *documented pipeline executes end-to-end with a real,
non-vacuous, search-driven reward* — not a learning curve. On a small corpus with a
competent tool-caller and a deterministic BM25 retriever, the within-group reward
variance is ~0 (every sampled completion issues the same query, retrieves the same
chunk, and answers the same), so GRPO's advantage (reward − group-mean) ≈ 0 and no
meaningful weight movement is expected from a short run. We measure and report that
variance rather than cherry-picking a curve. This mirrors `ppo_real.py`: the example
proves the pipeline *runs* for real, not that it converges.

Needs a GPU + `Qwen/Qwen2.5-3B-Instruct` in the local HF cache + `trl` installed. Run:
    python examples/rag_training_real.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

import sys

sys.modules["vllm"] = None  # env vllm is ABI-broken vs torch; TRL imports it eagerly

import statistics
import tempfile

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
from agenttune.core.backend_factory import create_agentic_trainer
from agenttune.rag.retrieval.corpus_loader import CorpusDocument, build_index
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend
from agenttune.rag.rewards.phase1_rewards import rag_correctness_reward, search_usage_reward
from agenttune.rag.tools.search_corpus import SearchCorpusTool

MODEL = "Qwen/Qwen2.5-3B-Instruct"
OUT_DIR = "/tmp/agenttune-rag-grpo"
SYSTEM = (
    "You are a research assistant. Call the `search_corpus` tool with a SHORT keyword "
    "query (2-3 distinctive words, not a full question) to retrieve passages, then "
    "answer in one short phrase grounded in them."
)

# A distractor corpus: several adjacent facts per topic so BM25 *can* retrieve the
# wrong chunk — i.e. gold-chunk coverage is a real signal, not a freebie.
CORPUS = [
    "The Amazon River in South America is the largest river by discharge volume.",  # chunk_0
    "The Nile River in Africa is often cited as the longest river in the world.",  # chunk_1
    "The Yangtze is the longest river in Asia and the third-longest in the world.",  # chunk_2
    "The Danube flows through more countries than any other river, ten in total.",  # chunk_3
    "Jupiter is the largest planet in the Solar System.",  # chunk_4
    "Saturn is the second-largest planet and is famous for its ring system.",  # chunk_5
    "Mercury is the smallest and innermost planet in the Solar System.",  # chunk_6
    "Venus is the hottest planet due to its dense carbon-dioxide atmosphere.",  # chunk_7
    "Insulin is a hormone produced by the pancreas that regulates blood sugar.",  # chunk_8
    "Adrenaline is a hormone released by the adrenal glands during stress.",  # chunk_9
]
# (question, gold_answer, gold_chunk_ids, keyword_query)
QA = [
    ("Which river is the longest in the world?", "Nile", ["chunk_1"], "longest river world"),
    ("What is the largest planet in the Solar System?", "Jupiter", ["chunk_4"], "largest planet"),
    ("Which organ produces insulin?", "pancreas", ["chunk_8"], "insulin pancreas"),
    ("Which planet is the hottest?", "Venus", ["chunk_7"], "hottest planet"),
    ("Which river passes through the most countries?", "Danube", ["chunk_3"], "river countries"),
    ("Which planet is the smallest?", "Mercury", ["chunk_6"], "smallest planet"),
]


def rag_reward(completions=None, prompts=None, tool_call_counts=None, gold_answer=None, **kwargs):
    """correctness (token-F1 vs gold_answer) + search-usage, matching the weighting
    `get_training_reward()` uses (0.7 / 0.2, dropping the 0.1 format term since these
    completions aren't tagged with <answer> in this script's prompting)."""
    correctness = rag_correctness_reward(prompts, completions, gold_answer=gold_answer)
    usage = search_usage_reward(prompts, completions, tool_call_counts=tool_call_counts)
    return [0.7 * c + 0.3 * u for c, u in zip(correctness, usage, strict=False)]


def measure_base(search_tool, model, tok, group=4):
    """Base-model reward + within-group variance, and proof searches fire."""
    rollout = create_rollout_fn(
        rollout_backend="transformers",
        model=model,
        tokenizer=tok,
        tools=[search_tool],
        max_steps=4,
        system_prompt=SYSTEM,
    )
    print("\n── Base model: reward + within-group variance (before training) ──")
    all_std, searched_any = [], 0
    for q, gold_a, _gold_c, _kw in QA:
        out = rollout([q] * group, temperature=0.9, top_p=0.95)
        rewards = rag_reward(
            completions=out["responses"],
            prompts=[q] * group,
            tool_call_counts=out["tool_call_counts"],
            gold_answer=[gold_a] * group,
        )
        searched_any += sum(1 for c in out["tool_call_counts"] if c > 0)
        mean, std = statistics.mean(rewards), statistics.pstdev(rewards)
        all_std.append(std)
        print(f"  {q:<48} mean={mean:5.3f} std={std:5.3f} searches/gen={out['tool_call_counts']}")
    print(f"  → completions that issued ≥1 real search: {searched_any}/{len(QA) * group}")
    print(
        f"  → mean within-group std across questions: {statistics.mean(all_std):.4f} "
        f"(≈0 ⇒ GRPO advantage ≈0 ⇒ no curve expected; pipeline-executes is the claim)"
    )

    # Preflight: the LITERAL documented reward path must score non-zero.
    q, gold_a, gold_c, _kw = QA[0]
    out = rollout([q] * group, temperature=0.9, top_p=0.95)
    literal = rag_reward(
        completions=out["responses"],
        prompts=[q] * group,
        tool_call_counts=out["tool_call_counts"],
        gold_answer=[gold_a] * group,
    )
    print(f"  → rag_reward(...) on a real batch: {[round(x, 3) for x in literal]}")
    assert any(
        x > 0 for x in literal
    ), "the documented reward_funcs=[rag_reward] path scored all-zero — the bug this fixes"
    return searched_any


def main():
    tmpdir = tempfile.mkdtemp()
    docs = [
        CorpusDocument(doc_id=f"chunk_{i}", title=f"chunk_{i}", text=t)
        for i, t in enumerate(CORPUS)
    ]
    backend = SQLiteFTSBackend(os.path.join(tmpdir, "corpus.db"))
    build_index(backend, docs, chunk_size=400, overlap=0)
    search_tool = SearchCorpusTool(backend, top_k=2)

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    measure_base(search_tool, base, tok)
    del base
    torch.cuda.empty_cache()

    # Training dataset: use the pre-built keyword query as the prompt so the base
    # model doesn't have to reliably reduce a full question to keywords itself —
    # that capability is a separate (documented) concern, not what GRPO is testing here.
    ds = Dataset.from_list([{"prompt": kw, "gold_answer": a} for _q, a, _c, kw in QA])

    # A thin recorder around the reward. It delegates verbatim; it only records that
    # real searches fired *inside* train() — proof the tool loop ran during optimisation,
    # not just standalone above.
    proof = {"reward_calls": 0, "completions": 0, "with_search": 0, "nonzero_rewards": 0}

    def rag_reward_probe(
        completions=None, prompts=None, tool_call_counts=None, gold_answer=None, **kwargs
    ):
        scores = rag_reward(
            completions=completions,
            prompts=prompts,
            tool_call_counts=tool_call_counts,
            gold_answer=gold_answer,
        )
        proof["reward_calls"] += 1
        for c in tool_call_counts or []:
            proof["completions"] += 1
            if c > 0:
                proof["with_search"] += 1
        proof["nonzero_rewards"] += sum(1 for s in scores if s > 0)
        return scores

    rag_reward_probe.__name__ = "rag_reward"

    # ── The documented one-liner (reward wrapped in a thin recorder for proof; the
    #    bare rag_reward path is asserted non-zero in measure_base above) ──
    trainer = create_agentic_trainer(
        "grpo",
        model=MODEL,
        train_dataset=ds,
        tools=[search_tool],
        reward_funcs=[rag_reward_probe],
        system_prompt=SYSTEM,
        max_steps_per_turn=4,  # rollout tool-loop budget
        # GRPO knobs (kept small — this proves execution, not convergence)
        output_dir=OUT_DIR,
        num_generations=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=4,
        max_completion_length=320,
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
    print("\n── Running real GRPO (LoRA) via the documented create_agentic_trainer path ──")
    trainer.train()

    gt = trainer.trainer  # underlying TRL GRPOTrainer
    gt.save_model(OUT_DIR)

    print("\n── Proof the agentic rollout fired *during* training ──")
    print(f"  reward invocations                     : {proof['reward_calls']}")
    print(f"  completions scored                     : {proof['completions']}")
    print(f"  completions that issued a real search  : {proof['with_search']}")
    print(f"  completions with non-zero RAG reward   : {proof['nonzero_rewards']}")
    losses = [h["loss"] for h in gt.state.log_history if "loss" in h]
    print(
        f"  per-step train loss                    : {losses} "
        f"(0.0 ⇒ zero advantage from zero within-group variance — as scoped)"
    )

    # Adapter produced and reloadable
    adapter_ok = os.path.exists(os.path.join(OUT_DIR, "adapter_model.safetensors"))
    reload_ok = False
    if adapter_ok:
        base2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
        merged = PeftModel.from_pretrained(base2, OUT_DIR)
        reload_ok = isinstance(merged, PeftModel)

    print("\n── Summary ──")
    print("  documented one-liner executed end-to-end : True")
    print(f"  tools invoked in-rollout during train()  : {proof['with_search'] > 0}")
    print(f"  reward was real & non-vacuous (not 0.0)  : {proof['nonzero_rewards'] > 0}")
    print(f"  LoRA adapter saved                       : {adapter_ok}")
    print(f"  adapter reloadable as PeftModel          : {reload_ok}")
    print("  convergence: NOT claimed (near-zero within-group reward variance on this toy set)")

    assert proof["with_search"] > 0, "no real searches fired during training — rollout not wired"
    assert proof["nonzero_rewards"] > 0, "reward degenerated to all-zero (the bug this fixes)"
    assert adapter_ok and reload_ok, "adapter not produced/reloadable"
    print("\n✓ Agentic-RAG GRPO pipeline ran for real, end-to-end.")


if __name__ == "__main__":
    main()
