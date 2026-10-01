"""
Stage 3 — Closed-loop verification (bounded to 1 retry).

Two checks per question, the differentiator of this pipeline. The check design
was reworked 2026-08-13 after a structural failure surfaced on the legal track:

  The OLD chain-dependency check masked an INTERMEDIATE hop and handed the
  solver first+last. But the generator's answer-first design places the answer
  in the LAST passage (near-verbatim) — so first+last always contained the
  answer, the solver always reproduced it, and every genuine 3+ hop question
  FAILED the check (measured ~35-40% rejection) while 2-hop paths skipped it
  entirely (len < 3 → trivial pass, never actually verified). The two checks
  contradicted each other: grounding demanded the answer tokens live in the
  last chunk, chain-dependency demanded the answer NOT be reconstructable from
  first+last. A verbatim-in-last answer cannot satisfy both.

  New design — each check tests exactly one property of a *definitional*
  multi-hop question (an earlier passage defines a term/rate/threshold/party;
  the last passage applies it opaquely or combines it with a figure):

  Check 1 — answerability (full chain): hand the solver ALL gold chunks. It
    must reproduce the answer (relaxed F1 >= 0.5, the same gate Stage 5 uses).
    This is the GROUNDING/verifiability gate: an answer the solver can't derive
    from its own gold chain is hallucinated. It subsumes the old token-overlap
    span check — a solver reproducing the exact answer from the gold context is
    strictly stronger — and it works for derived answers (a computed value that
    appears in no passage verbatim), which the span check could not handle.

  Check 2 — chain dependency (last passage alone): hand the solver ONLY the
    last gold chunk. It must NOT reproduce the answer (relaxed F1 < 0.7). This
    is the REAL multi-hop test, matching the generator's own hard requirement
    ("unanswerable from the last passage alone"): if the last passage alone
    suffices, the chain is decorative and a one-hop retriever would answer it.
    Applies to 2-hop paths too — no more free pass.

  Check 3 — retrieval necessity (one-shot leak): the top-1 retrieved passage
    must not alone carry the answer (answer tokens present in top-1 with >=
    0.6 coverage). If it does, a single-passage retriever solves the question
    and multi-hop retrieval is not necessary. Derived answers (value appears
    nowhere verbatim) and single-token answers are exempt by construction.

Only failures get ONE targeted regeneration (SAGE's feedback idea, Castform's
cost). Each sample records retrieval_necessity, chain_dependency,
revision_count, original_question, and discard_reason for full provenance.
"""

from __future__ import annotations

import logging
import re
import time

from .io_utils import extract_json
from .schema import QASample

logger = logging.getLogger(__name__)

_REVISE_PROMPT = """You are revising a multi-hop question that failed verification.

Original question: {question}
Original answer: {answer}
Problem: {problem}

The question must be genuinely multi-hop: an EARLIER passage defines a term,
rate, threshold, deadline, or party, and the LAST passage APPLIES that
definition opaquely (a defined term, "such rate", "as set forth in Section
4.1", "the foregoing") or combines it with a figure in the last passage. The
answer is the DEFINED VALUE (a short exact span from an earlier passage) or a
DERIVED value (a number/date the solver computes from the passages) — it must
NOT be recoverable from the LAST passage alone. Keep the answer short (<= 8
words) and exact, so a solver can reproduce it from the passages.

Passages (in order):
{path_text}

Respond ONLY with JSON:
{{"question": "<revised question>", "answer": "<answer, unchanged or refined>"}}

If the question cannot be salvaged, respond: {{"question": "", "answer": ""}}"""

_SOLVER_PROMPT = """Answer the question using ONLY the provided passages. If the answer is not derivable from the passages, respond with exactly "UNANSWERABLE".

Passages:
{context}

Question: {question}

Respond ONLY with JSON: {{"answer": "<short answer or UNANSWERABLE>"}}"""

# Reasoning models emit a thinking trace before the JSON answer. A small
# max_tokens (e.g. 128) is consumed entirely by the trace, so the JSON never
# appears → parse fails → "UNANSWERABLE" → spurious 0.0 answerability. Give the
# trace room to finish so the answer survives. (Non-reasoning models ignore the
# extra budget; cost impact is negligible since most calls finish early.)
_SOLVER_MAX_TOKENS = 8192

# Full-chain answerability gate — same threshold Stage 5 uses for `answerable`
# (relaxed F1, legal-tolerant normalizer). An accepted question must be
# solvable from its own gold chain.
_ANSWERABILITY_THRESHOLD = 0.5
# Last-passage-only gate — relaxed F1 must stay BELOW this when the chain is
# reduced to the last chunk alone, or the "chain" wasn't load-bearing.
_CHAIN_ACC_THRESHOLD = 0.7


def _f1(pred: str, gold: str) -> float:
    """Reuse the package's SQuAD F1 if available; fall back to datagen's."""
    try:
        from ..rewards.qa_metrics import f1_score

        return f1_score(pred, gold)
    except Exception:
        from ..datagen import token_f1

        return token_f1(pred, gold)


def _relaxed_f1(pred: str, gold: str) -> float:
    """Legal-tolerant token F1 (drops corporate suffixes, leading articles,
    folds hyphens) — the score Stage 3 gates on, same as Stage 5. Falls back
    to strict SQuAD F1 if the metrics module is unavailable."""
    try:
        from ..rewards.qa_metrics import relaxed_f1_score

        return relaxed_f1_score(pred, gold)
    except Exception:
        return _f1(pred, gold)


def _solver_answer(
    question: str, context: str, llm, *, stage="stage3", purpose="solver"
) -> tuple[str, object]:
    msg = [{"role": "user", "content": _SOLVER_PROMPT.format(context=context, question=question)}]
    text, rec = llm.chat(
        msg, stage=stage, purpose=purpose, temperature=0.0, max_tokens=_SOLVER_MAX_TOKENS
    )
    try:
        data = extract_json(text)
        return data.get("answer", "").strip(), rec
    except Exception:
        return "UNANSWERABLE", rec


# Answers that are a direct yes/no or a single-word judgment are exempt from
# the token-overlap grounding check (the answer is a decision, not an
# extractable span). Retained for the audit column `answer_grounded_span`;
# the hard grounding gate is now the full-chain answerability check.
_Y_N_WORDS = {
    "yes",
    "no",
    "n/a",
    "na",
    "none",
    "not specified",
    "unspecified",
    "unlimited",
    "not applicable",
}


def answer_span_grounded(answer: str, anchor_chunk_text: str, min_coverage: float = 0.6) -> bool:
    """Deterministic check: the answer's normalized tokens overlap the anchor
    chunk's tokens with token-coverage >= min_coverage.

    Lenient on purpose (a bit of paraphrase is fine, humans do it): the intent
    is to catch answers that are NOT grounded in the passage at all, not to ban
    paraphrasing. Yes/no and single-word judgments are exempt.

    NOTE (2026-08-13): this is now an AUDIT column, not a hard gate. The hard
    grounding gate is `check_answerability` — a solver reproducing the exact
    answer from the full gold chain. This span check cannot handle derived or
    cross-referenced answers (value defined in an earlier chunk, applied
    opaquely in the last), which are the whole point of the definitional
    multi-hop design; the solver check can.
    """
    a = answer.strip().lower()
    if not a:
        return False
    if a in _Y_N_WORDS:
        return True
    toks = re.sub(r"[^a-z0-9 ]", " ", a).split()
    if not toks:
        return False
    ctoks = set(re.sub(r"[^a-z0-9 ]", " ", anchor_chunk_text.lower()).split())
    if not ctoks:
        return False
    covered = sum(1 for t in toks if t in ctoks)
    return covered / len(toks) >= min_coverage


def check_retrieval_necessity(sample: QASample, search_backend, top_k=5) -> float:
    """Returns the rank (0-indexed) of the answer-anchor chunk in top-k, or -1.

    The rank is recorded for audit (the schema's `retrieval_necessity` column).
    The PASS decision is the one-shot-leak test in `verify_sample`: whether the
    single most-similar retrieved passage alone carries the answer.
    """
    results = search_backend.search(sample.question, top_k=top_k)
    result_ids = [r["metadata"].get("chunk_id") for r in results]
    anchor = sample.gold_chunk_ids[-1] if sample.gold_chunk_ids else None
    for i, rid in enumerate(result_ids):
        if rid == anchor:
            return float(i)
    return -1.0


_LEAK_PUNCT_RE = re.compile(r"[^\w\s]")
_LEAK_WS_RE = re.compile(r"\s+")


def _norm_tokens(text: str) -> list[str]:
    t = _LEAK_PUNCT_RE.sub(" ", (text or "").lower())
    return [w for w in _LEAK_WS_RE.sub(" ", t).strip().split() if w]


def _one_shot_leak(search_backend, question: str, answer: str, top_k=5, min_coverage=0.6) -> bool:
    """True if the top-1 retrieved passage alone carries the answer.

    Castform's retrieval-necessity intent, made compatible with within-document
    multi-hop: rejecting whenever the seed/anchor is at rank 0 over-rejected
    (any well-posed question retrieves its own contract's relevant passage —
    that's correct retrieval, and Check 2 already proves the anchor alone can't
    answer). What actually makes multi-hop retrieval unnecessary is a single
    passage that CONTAINS the answer. Derived answers (computed values that
    appear in no passage verbatim) and single-token answers (numbers, party
    names — too common to be a leak signal) are exempt by construction.
    """
    atoks = _norm_tokens(answer)
    if len(atoks) < 2:
        return False
    results = search_backend.search(question, top_k=top_k)
    if not results:
        return False
    ctoks = set(_norm_tokens(results[0]["content"]))
    if not ctoks:
        return False
    covered = sum(1 for t in atoks if t in ctoks)
    return covered / len(atoks) >= min_coverage


def check_answerability(
    sample: QASample, chunks_by_id, llm, *, stage="stage3"
) -> tuple[float, object]:
    """Check 1 — full-chain solver. Returns (relaxed F1, LLM record).

    The answer is verifiable (grounded) iff the solver, given ALL gold chunks,
    reproduces it. This is the grounding gate AND the answerability gate.
    """
    ctx = "\n\n".join(chunks_by_id[c].text for c in sample.gold_chunk_ids if c in chunks_by_id)
    ans, rec = _solver_answer(
        sample.question, ctx, llm, stage=stage, purpose="answerability_full_chain"
    )
    sample.llm_calls.append(rec)
    if ans == "UNANSWERABLE":
        return 0.0, rec
    return _relaxed_f1(ans, sample.answer), rec


def check_chain_dependency(
    sample: QASample, chunks_by_id, llm, *, stage="stage3"
) -> tuple[float, object]:
    """Check 2 — last-passage-only solver. Returns (relaxed F1, LLM record).

    `sample.chain_dependency` = the solver's score with the chain reduced to
    the LAST chunk alone. The question is genuinely multi-hop iff this is LOW
    (< `_CHAIN_ACC_THRESHOLD`): a reader who only sees the last passage must
    NOT be able to answer. Unlike the old mask-intermediate version this also
    applies to 2-hop paths (which used to pass the chain check untested).
    """
    if not sample.gold_chunk_ids:
        sample.chain_dependency = 0.0
        return 0.0, None
    anchor = chunks_by_id.get(sample.gold_chunk_ids[-1])
    if anchor is None:
        sample.chain_dependency = 0.0
        return 0.0, None
    ans, rec = _solver_answer(
        sample.question, anchor.text, llm, stage=stage, purpose="chain_dep_last_passage_only"
    )
    sample.llm_calls.append(rec)
    acc = 0.0 if ans == "UNANSWERABLE" else _relaxed_f1(ans, sample.answer)
    sample.chain_dependency = acc
    return acc, rec


def verify_sample(
    sample: QASample,
    chunks_by_id,
    search_backend,
    llm,
    *,
    retrieval_rank_threshold=0,
    chain_acc_threshold=_CHAIN_ACC_THRESHOLD,
    answerability_threshold=_ANSWERABILITY_THRESHOLD,
    stage="stage3",
) -> QASample:
    """Run all three checks, set pass/fail, retry once on failure.

    `retrieval_rank_threshold`: kept for call compatibility; the retrieval PASS
      decision now comes from the one-shot-leak test (`_one_shot_leak`), not
      the seed/anchor rank (see `check_retrieval_necessity` docstring).
    `chain_acc_threshold`: relaxed F1 the LAST-passage-only solver must stay
      under (0.7) — if it answers at/above this from the last chunk alone, the
      chain isn't load-bearing → fail.
    `answerability_threshold`: relaxed F1 the FULL-chain solver must reach
      (0.5) — if the gold chain can't produce the answer, it's unverifiable →
      fail.
    """
    # Short-circuit: an empty question or answer cannot be valid — reject
    # immediately (an empty question retrieves nothing and both solvers would
    # trivially "fail", letting it slip through as a non-multi-hop pass).
    if not sample.question or not sample.answer:
        sample.status = "rejected"
        sample.discard_reason = sample.discard_reason or "empty_qa"
        sample.retrieval_necessity_pass = False
        sample.chain_dependency_pass = False
        sample.answer_grounded_pass = False
        return sample

    # Audit column: the old deterministic token-overlap grounding check. NOT a
    # gate anymore (see `answer_span_grounded` docstring) — derived and
    # cross-referenced answers legitimately have no span in the last chunk.
    anchor = chunks_by_id.get(sample.gold_chunk_ids[-1]) if sample.gold_chunk_ids else None
    sample.answer_grounded_span = anchor is not None and answer_span_grounded(
        sample.answer, anchor.text
    )

    def _run_checks(s):
        # Check 3 — retrieval necessity (deterministic, no LLM cost)
        rank = check_retrieval_necessity(s, search_backend)
        s.retrieval_necessity = rank
        s.retrieval_necessity_pass = not _one_shot_leak(search_backend, s.question, s.answer)
        # Check 1 — answerability / grounding (full chain)
        full_acc, _ = check_answerability(s, chunks_by_id, llm, stage=stage)
        s.answer_grounded_pass = full_acc >= answerability_threshold
        s.answerability_f1_relaxed = full_acc
        s.answerable = s.answer_grounded_pass
        # Check 2 — chain dependency (last passage alone)
        last_acc, _ = check_chain_dependency(s, chunks_by_id, llm, stage=stage)
        s.chain_dependency_pass = last_acc < chain_acc_threshold
        return s.answer_grounded_pass and s.retrieval_necessity_pass and s.chain_dependency_pass

    passed = _run_checks(sample)

    if not passed and sample.revision_count == 0:
        # ONE targeted regeneration (SAGE feedback, bounded)
        sample.original_question = sample.question
        problem = []
        if not sample.answer_grounded_pass:
            problem.append(
                "a solver given the full passage chain could not reproduce "
                "the answer — the answer must be an exact short span from "
                "an earlier passage or a value a solver can compute from "
                "the passages"
            )
        if not sample.retrieval_necessity_pass:
            problem.append(
                "the single most-similar retrieved passage alone carries "
                "the answer — the question gives the answer away; "
                "paraphrase it so no one passage suffices"
            )
        if not sample.chain_dependency_pass:
            problem.append(
                "the question CAN be answered from the LAST passage alone — "
                "move the answer to an earlier passage and make the last "
                "passage apply it opaquely (defined term, 'such rate', 'as "
                "set forth in Section 4.1'), or make the answer a derived "
                "value that appears in no passage verbatim"
            )
        sample = _revise(sample, chunks_by_id, llm, problem="; ".join(problem), stage=stage)
        sample.revision_count = 1
        # Re-run all checks on the revised question (no second retry)
        passed = _run_checks(sample)

    if passed:
        sample.status = "accepted" if sample.revision_count == 0 else "revised_accepted"
    else:
        sample.status = "rejected"
        if not sample.discard_reason:
            reasons = []
            if not sample.answer_grounded_pass:
                reasons.append("answer_not_grounded")
            if not sample.retrieval_necessity_pass:
                reasons.append("retrieval_leak")
            if not sample.chain_dependency_pass:
                reasons.append("answerable_from_last_passage")
            sample.discard_reason = "+".join(reasons)
    return sample


def _revise(sample: QASample, chunks_by_id, llm, *, problem: str, stage="stage3") -> QASample:
    from .generate import _path_text

    msg = [
        {
            "role": "user",
            "content": _REVISE_PROMPT.format(
                question=sample.question,
                answer=sample.answer,
                problem=problem,
                path_text=_path_text(sample.path, chunks_by_id),
            ),
        }
    ]
    text, rec = llm.chat(
        msg, stage=stage, purpose="revise_question", temperature=0.3, max_tokens=12288
    )
    sample.llm_calls.append(rec)
    try:
        data = extract_json(text)
        new_q = data.get("question", "").strip()
        new_a = data.get("answer", "").strip()
        if new_q:
            sample.question = new_q
            if new_a:
                sample.answer = new_a
        else:
            sample.status = "rejected"
            sample.discard_reason = "revise_failed"
    except Exception:
        sample.status = "rejected"
        sample.discard_reason = "revise_parse_failure"
    return sample


class _ThreadSafeSearchBackend:
    """Wraps a SearchBackend with a lock around `search`.

    `verify_batch` runs samples on multiple threads; the SQLite backend
    holds a shared connection (opened with check_same_thread=False) and
    sqlite3 cursors are not safe to interleave from multiple threads. The
    lock serializes only the local, millisecond-fast retrieval call — LLM
    calls (the actual cost) stay fully parallel.
    """

    def __init__(self, backend):
        import threading

        self._backend = backend
        self._lock = threading.Lock()

    def search(self, *args, **kwargs):
        with self._lock:
            return self._backend.search(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._backend, name)


def verify_batch(
    samples: list[QASample],
    chunks_by_id,
    search_backend,
    llm,
    *,
    max_workers=64,
    checkpoint_path: str = None,
    **kw,
) -> list[QASample]:
    """Run Stage 3 verification over all samples, samples in parallel.

    Semantics identical to the sequential loop (each sample goes through the
    exact same `verify_sample` logic — checks, thresholds, one revision,
    discard reasons). Samples are independent: each thread mutates only its
    own sample; `chunks_by_id` is read-only; the search backend is wrapped
    thread-safe; the LLM client is safe for concurrent calls. This is the
    dominant wall-clock stage at scale (two solver calls per sample) — at
    DeepSeek's concurrency it is now ~O(1) per batch instead of O(n)
    sequential calls. Preserves input order.

    `checkpoint_path`: crash-safe incremental checkpoint (see checkpoint.py).
    Each fully-verified sample is appended as it finishes; a restart reuses
    completed samples and re-verifies only the missing ones. Keyed to (model,
    thresholds, input sample fingerprint) so a changed prompt/threshold/input
    invalidates it.
    """
    from concurrent.futures import ThreadPoolExecutor, wait

    from .checkpoint import CheckpointLog, stage_meta
    from .llm_client import _safe_max_workers

    if not samples:
        return samples

    done: dict[str, QASample] = {}
    ckpt = None
    if checkpoint_path:
        ckpt = CheckpointLog(
            checkpoint_path,
            meta=stage_meta(
                "stage3",
                model=getattr(llm, "model", "?"),
                thresholds={
                    "chain_acc": kw.get("chain_acc_threshold", _CHAIN_ACC_THRESHOLD),
                    "answerability": kw.get("answerability_threshold", _ANSWERABILITY_THRESHOLD),
                },
                samples=samples,
            ),
        )
        for rec in ckpt.load():
            if rec is not None and getattr(rec, "sample_id", None):
                done[rec.sample_id] = rec

    pending = [s for s in samples if s.sample_id not in done]
    if ckpt and done:
        logger.info(
            f"[stage3] resumed {len(done)}/{len(samples)} from checkpoint, "
            f"verifying {len(pending)}"
        )

    results: dict[str, QASample] = dict(done)
    if pending:
        safe_sb = _ThreadSafeSearchBackend(search_backend)
        workers = _safe_max_workers(max(1, max_workers))
        # Robust batch (2026-08-14): never let a worker exception OR a hung LLM
        # call freeze the whole stage. The old `with ThreadPoolExecutor` +
        # as_completed pattern blocked forever in shutdown(wait=True) when a
        # sibling future hung (observed: stage froze at 483/5457 with the main
        # thread in futex_wait). Here we poll with wait(timeout=), catch
        # per-future errors, and abandon futures that don't finish by a hard
        # deadline — the batch always terminates, and hung samples fall back to
        # rejected (the checkpoint lets a restart re-do only what never landed).
        # NOTE: pids.max on this box is 768; leftover threads from a previous
        # stage starve the next executor (observed pids.current 696 → executor
        # clamped to ~74 → effectively frozen). Keep `workers` well under the
        # pids headroom and let `_safe_max_workers` clamp.
        ex = ThreadPoolExecutor(max_workers=workers)
        futs = {ex.submit(verify_sample, s, chunks_by_id, safe_sb, llm, **kw): s for s in pending}
        remaining = set(futs)
        deadline = time.time() + _batch_budget(len(pending), workers)
        try:
            while remaining and time.time() < deadline:
                done_now, remaining = wait(remaining, timeout=60)
                for fut in done_now:
                    s = futs[fut]
                    try:
                        vs = fut.result()
                    except Exception as e:
                        s.status = "rejected"
                        s.discard_reason = s.discard_reason or "worker_error"
                        logger.error(f"[stage3] worker error on {s.sample_id}: {e!r}")
                        results[s.sample_id] = s
                        continue
                    results[vs.sample_id] = vs
                    if ckpt:
                        ckpt.append(vs)
            for fut in remaining:
                s = futs[fut]
                s.status = "rejected"
                s.discard_reason = s.discard_reason or "worker_error_timeout"
                results[s.sample_id] = s
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    return [results[s.sample_id] for s in samples]


def _batch_budget(n_pending: int, workers: int) -> int:
    """Wall-clock budget (seconds) for a whole batch, scaled to workload.

    A fixed 8-minute budget silently abandoned 3,035/5,271 futures in the
    2026-08-13 4K run — the dominant rejection cause there (57.6% of the
    corpus dropped as `worker_error_timeout`) even though verification itself
    was healthy at 78.6% acceptance among the futures that finished. Budget ≈
    2.5× expected time (~15 s/sample ÷ workers), floored at 15 min and capped
    at 6 h so a genuinely wedged batch still can't block the pipeline forever
    (the client's 60 s timeout + max_retries=0 already fail hung calls fast)."""
    expected_s = n_pending * 15 / max(1, workers)
    return max(15 * 60, min(6 * 60 * 60, int(2.5 * expected_s)))
