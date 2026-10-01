"""
REAL docs -> QA data-gen + difficulty curriculum, over a live LLM.
=================================================================

`agenttune.rag.datagen` advertises a P3 pipeline for making agentic-RAG training
work on a user's *own* corpus, where no question set exists::

    chunks   = [...]                                   # your documents
    qa       = generate_qa_from_corpus(chunks, generator)   # generator wraps an LLM
    labeled  = label_difficulty(qa, solver)                 # solver = the base model
    curric   = balance_by_difficulty(labeled)              # ~1:1 easy/hard (IKEA)

Until now both model-dependent callables (`generator`, `solver`) were only ever
exercised by **mocks** in `tests/rag/test_datagen.py`. This script runs the whole
generate -> label -> balance -> sort pipeline with a **real** `Qwen2.5-3B-Instruct`
on both ends, and feeds the result into the current real BM25 retriever + RAG
reward path that `rag_training_real.py` trains on.

NOTE (2026-07-10): this script originally used `agenttune.rag.environment.RAGEnvironment`
and `agenttune.rag.retriever.Chunk`. Both were deleted when the RAG package was
overlaid with a fuller retrieval/tools/rewards implementation. This version
is ported to
the *current* modules — `agenttune.rag.retrieval.sqlite_fts.SQLiteFTSBackend`,
`agenttune.rag.tools.search_corpus.SearchCorpusTool`, `agenttune.rag.rewards.phase1_rewards` —
with the same intent and success criteria as the original.

THE CORPUS IS DESIGNED TO STRADDLE THE KNOWLEDGE BOUNDARY
--------------------------------------------------------
`label_difficulty` probes the base model **with no document**: a question it can
answer unaided is *easy* (it already knows — deprioritise), one it cannot is *hard*
(a genuine gap — worth training the search behaviour on). If every fact were famous
(Paris, Jupiter) the base model would know them all -> everything *easy* -> `hard=0`
-> `balance_by_difficulty` returns `[]` and the curriculum is vacuous. So the corpus
deliberately mixes:
  * WELL-KNOWN facts the base model answers unaided        -> label *easy*
  * INVENTED facts (fictional entities/dates) it cannot know -> label *hard*
The generator *reads* each chunk to write a grounded Q&A either way (reading, not
recalling); the solver, probed blind, reveals the boundary. A non-degenerate split
(`n_easy > 0 and n_hard > 0`) is the explicit success criterion.

HONEST SCOPE — read before quoting
----------------------------------
This proves the *mechanism runs for real over an LLM*: real questions generated from
real chunks, grounded (verified against the real BM25 retriever), and a real
base-model difficulty probe that discriminates easy from hard. It does **not** claim
that auto-generated QA *trains well* — that remains an open bet (our own
step, a parity requirement). We report the grounding rate and the split; we do not
claim QA quality or a learning curve.

Needs a GPU + `Qwen/Qwen2.5-3B-Instruct` in the local HF cache. Run:
    python examples/rag_datagen_real.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

import sys

sys.modules["vllm"] = None  # env vllm is ABI-broken vs torch; keep it out of the import graph

import re
import tempfile

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
from agenttune.rag.datagen import (
    Chunk,
    balance_by_difficulty,
    generate_qa_from_corpus,
    label_difficulty,
    sort_by_difficulty,
)
from agenttune.rag.retrieval.corpus_loader import CorpusDocument, build_index
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend
from agenttune.rag.rewards.phase1_rewards import get_training_reward
from agenttune.rag.tools.search_corpus import SearchCorpusTool

MODEL = "Qwen/Qwen2.5-3B-Instruct"
# SQLiteFTSBackend AND-matches every term in a query (see `_BM25_QUERY_GUIDANCE` in
# agenttune.rag.data.hotpotqa) — a full natural-language question usually fails to
# retrieve because it contains words (who/what/where...) absent from the source
# text. Both the agent's system prompt and this script's own grounding check use a
# short keyword query instead, per that documented BM25 convention.
SYSTEM_RAG = (
    "You are a research assistant. Call the `search_corpus` tool with a SHORT "
    "keyword query (2-3 distinctive words, not a full question) to retrieve "
    "passages, then answer in one short phrase grounded in them."
)

_STOPWORDS = {
    "where",
    "what",
    "who",
    "when",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "does",
    "do",
    "did",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "that",
    "this",
    "you",
    "your",
}


def top_keyword(question: str) -> str:
    """Reduce a question to its single most distinctive keyword for a BM25 AND-query
    (prefers a capitalized/proper-noun word, falling back to the longest remaining
    word) — the same "short keyword query" convention SYSTEM_RAG asks the model for."""
    words = re.findall(r"[A-Za-z']+", question)
    kept = [w for w in words if w.lower() not in _STOPWORDS]
    if not kept:
        return question
    capitalized = [w for w in kept if w[0].isupper()]
    pool = capitalized or kept
    return max(pool, key=len)


# Well-known facts — the base model answers these unaided -> expected EASY.
KNOWN = [
    "The Eiffel Tower is located in the city of Paris.",
    "Insulin is the hormone produced by the pancreas.",
    "Jupiter is the largest planet in the Solar System.",
    "The chemical symbol for gold is Au.",
]
# Invented facts — no model can know these; only the chunk states them -> expected HARD.
INVENTED = [
    "The Kthonic Protocol was ratified in the year 3021 by the Vorlan Assembly.",
    "Zerelium is a synthetic metal whose melting point is 4820 degrees Brenn.",
    "The city of Quorindale is the capital of the fictional province of Aethmark.",
    "Doctor Selna Vex invented the tachyon looming press in the town of Ombervale.",
]
CORPUS = KNOWN + INVENTED  # chunk_0..chunk_3 known, chunk_4..chunk_7 invented


def chat(model, tok, system, user, max_new=48):
    """Greedy (deterministic) single-turn chat completion."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id
        )
    return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def make_generator(model, tok):
    """generator(chunk_text) -> [(question, answer)] — a real LLM reading each chunk."""
    sys_p = (
        "Read the passage and write ONE factual question that is answerable ONLY "
        "from it, plus its short answer (a few words). Reply EXACTLY as:\n"
        "Question: <q>\nAnswer: <a>"
    )

    def generator(text):
        raw = chat(model, tok, sys_p, f"Passage: {text}", max_new=64)
        q = re.search(r"Question:\s*(.+)", raw)
        a = re.search(r"Answer:\s*(.+)", raw)
        if not (q and a):
            return []
        return [(q.group(1).strip(), a.group(1).strip().split("\n")[0])]

    return generator


def make_solver(model, tok):
    """solver(question) -> answer — the base model probed with NO document (greedy)."""
    sys_p = "Answer the question in a few words. If you do not know, say 'unknown'."

    def solver(question):
        return chat(model, tok, sys_p, question, max_new=24)

    return solver


def main():
    tmpdir = tempfile.mkdtemp()
    docs = [
        CorpusDocument(doc_id=f"chunk_{i}", title=f"chunk_{i}", text=t)
        for i, t in enumerate(CORPUS)
    ]
    backend = SQLiteFTSBackend(os.path.join(tmpdir, "corpus.db"))
    build_index(backend, docs, chunk_size=400, overlap=0)
    search_tool = SearchCorpusTool(backend, top_k=3)
    reward_fn = get_training_reward()  # format(.1) + search_usage(.2) + correctness(.7)

    chunks = [Chunk(id=f"chunk_{i}", text=t) for i, t in enumerate(CORPUS)]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    # ── 1. Generate grounded QA from the corpus with a real LLM ──
    print("\n── Generating QA from corpus (real Qwen2.5-3B reads each chunk) ──")
    qa = generate_qa_from_corpus(chunks, make_generator(model, tok))
    for p in qa:
        print(f"  [{p.gold_chunk_ids[0]}] Q: {p.question}  |  A: {p.answer}")
    print(f"  → generated {len(qa)} grounded QA pairs from {len(chunks)} chunks")
    assert qa, "generator produced no parseable QA pairs"

    # ── 2. Grounding check against the REAL BM25 retriever ──
    grounded = 0
    for p in qa:
        results = backend.search(top_keyword(p.question), top_k=3)
        retrieved_docs = {r["metadata"].get("doc_id") for r in results}
        if p.gold_chunk_ids[0] in retrieved_docs:
            grounded += 1
    print(f"\n── Grounding: gold chunk retrieved in top-3 for {grounded}/{len(qa)} questions ──")

    # ── 3. Difficulty labelling by probing the base model (no document, greedy) ──
    print("\n── Difficulty: probe the base model with NO document ──")
    label_difficulty(
        qa, make_solver(model, tok), mode="f1", n_probes=1, pass_threshold=0.5, correct_at=0.4
    )
    for p in qa:
        klass = "known" if int(p.gold_chunk_ids[0].split("_")[1]) < len(KNOWN) else "invented"
        print(f"  [{klass:<8} {p.gold_chunk_ids[0]}] {p.difficulty:<4}  Q: {p.question}")
    n_easy = sum(p.difficulty == "easy" for p in qa)
    n_hard = sum(p.difficulty == "hard" for p in qa)
    print(f"  → easy={n_easy}  hard={n_hard}")

    # ── 4. Curriculum: 1:1 balance + easy->hard order ──
    balanced = balance_by_difficulty(qa, ratio=(1, 1))
    ordered = sort_by_difficulty(qa)
    print("\n── Curriculum ──")
    print(
        f"  balanced 1:1 subset : {len(balanced)} pairs "
        f"({sum(p.difficulty=='easy' for p in balanced)} easy / "
        f"{sum(p.difficulty=='hard' for p in balanced)} hard)"
    )
    print(f"  easy->hard order    : {[p.difficulty for p in ordered]}")

    # ── 5. The generated HARD pairs feed the real training path (no second train) ──
    hard = [p for p in qa if p.difficulty == "hard"]
    ds = Dataset.from_list(
        [
            {"prompt": p.question, "gold_answer": p.answer, "gold_chunk_ids": p.gold_chunk_ids}
            for p in hard
        ]
    )
    print("\n── Generated data feeds the documented reward path ──")
    print(f"  Dataset schema (identical to rag_training_real): {ds.column_names}")
    rollout = create_rollout_fn(
        rollout_backend="transformers",
        model=model,
        tokenizer=tok,
        tools=[search_tool],
        max_steps=4,
        system_prompt=SYSTEM_RAG,
    )
    probe = hard[0]
    out = rollout([probe.question] * 4, temperature=0.9, top_p=0.95)
    scores = reward_fn(
        completions=out["responses"],
        prompts=[probe.question] * 4,
        gold_answer=[probe.answer] * 4,
        tool_call_counts=out["tool_call_counts"],
    )
    searched = sum(1 for c in out["tool_call_counts"] if c > 0)
    print("  scored a generated HARD question through the real RAG reward on a real rollout:")
    print(f"    Q: {probe.question}")
    print(
        f"    get_training_reward()(...) = {[round(s, 3) for s in scores]}  "
        f"(searches fired: {searched}/4)"
    )

    # ── Summary + honest assertions ──
    print("\n── Summary ──")
    print(f"  real LLM generated grounded QA           : {len(qa)} pairs, {grounded} grounded")
    print(f"  difficulty probe discriminated           : easy={n_easy}, hard={n_hard}")
    print(f"  curriculum non-degenerate (1:1 possible) : {len(balanced) > 0}")
    print(f"  generated data scores through the reward : {any(s > 0 for s in scores)}")
    print("  scope: mechanism run for real; doc->QA training quality NOT claimed (open bet)")

    assert grounded > 0, "no generated question retrieves its gold chunk — generation ungrounded"
    assert n_easy > 0 and n_hard > 0, (
        f"degenerate difficulty split (easy={n_easy}, hard={n_hard}) — the probe did not "
        "discriminate; the boundary-straddling corpus should yield both classes"
    )
    assert len(balanced) > 0, "1:1 balance empty — curriculum vacuous"
    assert any(s > 0 for s in scores), "generated data scored all-zero through the reward"
    print("\n✓ Real doc->QA data-gen + difficulty curriculum ran end-to-end.")


if __name__ == "__main__":
    main()
