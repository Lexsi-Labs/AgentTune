"""
Stage 5 — Evaluation suite for the generated dataset.

Metrics are drawn from established frameworks, not invented:

1. **Answerability / self-consistency** (ARES-style, no gold needed): a solver
   LLM given the full gold path should answer correctly. Pass-rate measures
   whether the question is actually answerable from its supporting passages.
   Reuses the package's SQuAD F1 (qa_metrics.f1_score) as the scorer.

2. **Faithfulness / groundedness** (RAGAS faithfulness, SelfCheckGPT family):
   an LLM-judge rates whether the generated answer is supported by the gold
   passages (not hallucinated). Reuses the package's LLMJudge
   (build_groq_judge pattern).

3. **Retrieval recall@k**: of the gold chunks, how many does top-k retrieval
   surface for the question? High recall = the question is retrievable by a
   real RAG system. Reuses the package's SearchBackend.

4. **Multi-hop necessity** (Min et al. 2019 fix — our Stage 3 check 2):
   fraction of questions where masking an intermediate hop BREAKS the solver
   (= the hop is load-bearing). This is the dataset's "genuine multi-hop"
   rate — the key quality signal for an RL dataset that must reward real
   multi-step retrieval.

5. **Diversity**: lexical (unique question n-gram rate) + structural (question
   type distribution) + semantic (mean pairwise question-question cosine;
   lower = more diverse).

6. **Coverage**: fraction of corpus chunks that appear in at least one gold path.

7. **Difficulty distribution**: the 2D matrix cell histogram (GRADE).

Everything writes to a flat metrics dict + a per-sample scores table for CSV.
"""

from __future__ import annotations

from collections import Counter

from .schema import QASample


def _f1(pred: str, gold: str) -> float:
    try:
        from ..rewards.qa_metrics import f1_score

        return f1_score(pred, gold)
    except Exception:
        from ..datagen import token_f1

        return token_f1(pred, gold)


def answerability(
    samples: list[QASample], chunks_by_id, llm, *, f1_threshold=0.5, stage="eval", max_workers=64
) -> dict:
    """Solver-with-full-path pass rate. ~ARES answerability (no gold beyond the path).

    Reports BOTH scorers: the strict SQuAD F1 (comparable to prior runs /
    literature) and the legal-tolerant relaxed F1 (see qa_metrics.relaxed_f1 —
    handles "The ARC Group, Inc." vs "ARC Group", "Tel-Aviv" vs "Tel Aviv").
    The per-sample `answerable` gate is the relaxed score >= threshold: a
    question whose own gold path can't be solved is poison for training, and
    that's the score we gate on (strict F1 artifacts like suffix/punctuation
    mismatch must not drop otherwise-solvable questions).

    Solver calls run `max_workers`-way concurrent (all calls share the same
    purpose — a uniform batch). Per-sample ordering and records are
    preserved.
    """
    from .llm_client import parallel_chat

    passed, passed_relaxed, scores, scores_relaxed = 0, 0, [], []
    msgs = []
    for s in samples:
        ctx = "\n\n".join(chunks_by_id[c].text for c in s.gold_chunk_ids if c in chunks_by_id)
        msgs.append([{"role": "user", "content": _solver_prompt(ctx, s.question)}])
    # max_tokens=1024: reasoning models emit a thinking trace before the
    # JSON; 128 was eaten by the trace → spurious UNANSWERABLE → 0.0.
    results = parallel_chat(
        llm,
        msgs,
        max_workers=max_workers,
        stage=stage,
        purpose="eval_answerability",
        temperature=0.0,
        max_tokens=12288,
    )
    for s, (text, rec) in zip(samples, results, strict=False):
        s.llm_calls.append(rec)
        try:
            from .io_utils import extract_json

            ans = extract_json(text).get("answer", "").strip()
        except Exception:
            ans = "UNANSWERABLE"
        if ans == "UNANSWERABLE":
            sc = sc_relaxed = 0.0
        else:
            sc = _f1(ans, s.answer)
            sc_relaxed = _relaxed_f1(ans, s.answer)
        s.answerability_f1 = sc  # strict, per-sample (schema compat)
        s.answerability_f1_relaxed = sc_relaxed
        s.answerable = sc_relaxed >= f1_threshold
        scores.append(sc)
        scores_relaxed.append(sc_relaxed)
        if sc >= f1_threshold:
            passed += 1
        if sc_relaxed >= f1_threshold:
            passed_relaxed += 1
    n = max(1, len(samples))
    return {
        "answerability_pass_rate": passed / n,
        "answerability_mean_f1": sum(scores) / n,
        "answerability_pass_rate_relaxed": passed_relaxed / n,
        "answerability_mean_f1_relaxed": sum(scores_relaxed) / n,
        "answerable_rate": passed_relaxed / n,
    }


def _relaxed_f1(pred: str, gold: str) -> float:
    try:
        from ..rewards.qa_metrics import relaxed_f1_score

        return relaxed_f1_score(pred, gold)
    except Exception:
        return _f1(pred, gold)


def faithfulness(
    samples: list[QASample], chunks_by_id, judge_llm, *, stage="eval", max_workers=64
) -> dict:
    """LLM-judge groundedness of the answer in the gold passages (RAGAS/SelfCheckGPT family).

    Judge calls run `max_workers`-way concurrent (uniform purpose batch).
    """
    from .llm_client import parallel_chat

    scores = []
    msgs = []
    for s in samples:
        ctx = "\n\n".join(chunks_by_id[c].text for c in s.gold_chunk_ids if c in chunks_by_id)
        rubric = (
            "Is the following answer fully supported by (and correct given) the "
            "passages? 1.0 = fully grounded & correct, 0.0 = unsupported/wrong. "
            "Reply with a single float."
        )
        msgs.append(
            [
                {"role": "system", "content": rubric},
                {"role": "user", "content": f"Passages:\n{ctx}\n\nAnswer: {s.answer}"},
            ]
        )
    results = parallel_chat(
        judge_llm,
        msgs,
        max_workers=max_workers,
        stage=stage,
        purpose="eval_faithfulness",
        temperature=0.0,
        max_tokens=12288,
    )
    for s, (text, rec) in zip(samples, results, strict=False):
        s.llm_calls.append(rec)
        try:
            fs = float(text.strip().split()[0])
        except Exception:
            fs = 0.0
        s.faithfulness_score = fs  # persist per-sample (RAGAS groundedness)
        scores.append(fs)
    n = max(1, len(samples))
    return {
        "faithfulness_mean": sum(scores) / n,
        "faithfulness_ge_0.7": sum(1 for x in scores if x >= 0.7) / n,
    }


def retrieval_recall(samples: list[QASample], search_backend, *, top_k=5) -> dict:
    """Of gold chunks, fraction retrieved in top-k. High = retrievable by real RAG."""
    recalls = []
    for s in samples:
        results = search_backend.search(s.question, top_k=top_k)
        got = {r["metadata"].get("chunk_id") for r in results}
        gold = set(s.gold_chunk_ids)
        if gold:
            recalls.append(len(gold & got) / len(gold))
    n = max(1, len(samples))
    return {"retrieval_recall_at_5_mean": sum(recalls) / n if recalls else 0.0}


def retrieval_recall_dense(samples: list[QASample], chunks_by_id, embedder, *, top_k=5) -> dict:
    """Dense (embedding) recall@k — the fair counterpart to the BM25 recall above.

    The questions are deliberately paraphrased to avoid lexical leakage, so BM25
    recall is near-zero by construction and doesn't reflect whether a real
    dense-RAG system could retrieve the gold chunks. This embeds each question
    and ranks all corpus chunks by cosine, then measures how many gold chunks
    land in top-k. No LLM calls.
    """
    if not samples or embedder is None:
        return {"retrieval_recall_dense_at_5_mean": None}
    import numpy as np

    cids = [c.chunk_id for c in chunks_by_id.values() if c.embedding is not None]
    if not cids:
        return {"retrieval_recall_dense_at_5_mean": None}
    mat = np.array([chunks_by_id[c].embedding for c in cids])  # (N, d), normalized
    recalls = []
    q_vecs = embedder.embed_queries([s.question for s in samples])
    for s, qv in zip(samples, q_vecs, strict=False):
        q = np.array(qv)
        nq = np.linalg.norm(q) or 1.0
        sims = mat @ q / nq  # chunks normalized, q normalized
        k = min(top_k, len(sims) - 1)
        top = np.argpartition(-sims, k)[:top_k] if k >= 0 else np.arange(len(sims))
        got = {cids[i] for i in top}
        gold = set(s.gold_chunk_ids)
        if gold:
            recalls.append(len(gold & got) / len(gold))
    n = max(1, len(samples))
    return {"retrieval_recall_dense_at_5_mean": sum(recalls) / n if recalls else 0.0}


def multi_hop_necessity(samples: list[QASample]) -> dict:
    """Fraction of questions whose chain-dependency check PASSED (hop is load-bearing).

    This is the Stage-3 check-2 result aggregated — the dataset's genuine
    multi-hop rate (Min et al. 2019 fix).
    """
    n = max(1, len(samples))
    passed = sum(1 for s in samples if s.chain_dependency_pass)
    return {"multi_hop_necessity_rate": passed / n}


def diversity(samples: list[QASample], embedder=None) -> dict:
    """Lexical + structural + (optional) semantic diversity."""
    qs = [s.question for s in samples if s.question]
    # lexical: unique 4-gram ratio
    grams = set()
    total = 0
    for q in qs:
        toks = q.lower().split()
        for i in range(len(toks) - 3):
            grams.add(" ".join(toks[i : i + 4]))
            total += 1
    lexical = len(grams) / max(1, total)
    # structural: question type distribution entropy
    types = Counter(s.question_type for s in samples)
    import math

    ent = -sum((c / len(samples)) * math.log(c / len(samples)) for c in types.values() if c)
    # semantic: mean pairwise cosine (lower = more diverse)
    sem = None
    if embedder is not None and len(qs) >= 2:
        vecs = embedder.embed_queries(qs)
        import numpy as np

        V = np.array(vecs)
        sim = V @ V.T
        n = len(qs)
        sem = float((sim.sum() - n) / (n * (n - 1)))  # mean off-diagonal
    return {
        "diversity_lexical_4gram": lexical,
        "diversity_type_entropy": ent,
        "diversity_semantic_mean_cosine": sem,
    }


def coverage(samples: list[QASample], total_chunks: int) -> dict:
    """Fraction of corpus chunks appearing in at least one gold path."""
    used = set()
    for s in samples:
        used.update(s.gold_chunk_ids)
    return {"coverage_fraction": len(used) / max(1, total_chunks)}


def difficulty_distribution(samples: list[QASample]) -> dict:
    """The 2D matrix cell histogram (GRADE)."""
    cells = Counter(s.difficulty_cell or "unknown" for s in samples)
    return {
        "difficulty_cell_counts": dict(cells),
        "mean_retrieval_difficulty": (
            sum(s.retrieval_difficulty or 0 for s in samples) / max(1, len(samples))
        ),
    }


def cost_summary(samples: list[QASample]) -> dict:
    """Aggregate LLM token/cost/latency across all calls."""
    calls = [c for s in samples for c in s.llm_calls]
    return {
        "n_llm_calls": len(calls),
        "total_prompt_tokens": sum(c.prompt_tokens for c in calls),
        "total_completion_tokens": sum(c.completion_tokens for c in calls),
        "total_cost_usd": round(sum(c.cost_usd for c in calls), 6),
        "mean_latency_ms": round(sum(c.latency_ms for c in calls) / max(1, len(calls)), 2),
    }


def _solver_prompt(ctx, q):
    return (
        f"Answer the question using ONLY the passages. If not derivable, "
        f"say UNANSWERABLE.\n\nPassages:\n{ctx}\n\nQuestion: {q}\n\n"
        f'Respond ONLY with JSON: {{"answer": "<short answer or UNANSWERABLE>"}}'
    )
