"""
Stage 2 — Answer-first QA generation (MHTS).

For each sampled path: fix the answer first (a concrete entity/fact drawn from
an earlier chunk, or a value derived by applying an earlier-defined term to the
last chunk), then generate a question that requires traversing exactly the
path's chunks to reach it. Answer-first avoids the question-first failure mode
of inventing a question then hallucinating supporting facts.

Definitional-dependency design (rework 2026-08-13): the answer is a DEFINED
VALUE (an exact span from an EARLIER chunk) or a DERIVED value (a computation
that appears in no chunk verbatim); the LAST chunk must never contain it. This
is what makes the question genuinely multi-hop under Stage 3's checks: the
last-chunk-alone solver must fail, the full-chain solver must succeed.

The generator is the injected LLM (Groq qwen3.6-27b in production; a fake for
tests). Structured output via JSON mode + the io_utils.extract_json parser.

Each generated QASample records the full LLM call metadata (tokens/cost/
latency) on its `llm_calls` list, for the per-call reproducibility dump.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from .io_utils import extract_json
from .schema import QASample, ReasoningPath

logger = logging.getLogger(__name__)

_ANSWER_FIRST_PROMPT = """You are generating a multi-hop question for a RAG training dataset.

You are given an ordered list of passages (a reasoning path). The answer to the
question is a SPECIFIC, concrete entity or short fact (a name, a date, a term)
derivable from the FINAL passage, but the question must REQUIRE the reader to
follow the chain through ALL passages to reach it.

Rules:
- Fix the answer first (it must appear in / follow from the last passage).
- The question must NOT name the answer, and must NOT quote passage text
  verbatim (paraphrase around named entities so a lexical match can't shortcut).
- The question must be self-contained/decontextualized (no dangling pronouns).
- Answer in ONE short phrase (<= ~6 words).

Passages (in traversal order):
{path_text}

Respond ONLY with JSON:
{{"answer": "<short concrete answer>", "question": "<the question>", "reasoning": "<one sentence: how the chain resolves the answer>"}}"""

# Legal-register variant (NLLP_SynthData.md Part A), validated on 40 real
# within-document paths across 14 diverse CUAD contracts — not tuned to a
# handful of examples. Three real, broad-sample findings drove the wording:
#
# 1. The generic prompt above, run as-is on contract text, was fine at
#    producing genuine multi-hop chains but read like trivia questions, not
#    something a contract reviewer would ask — hence the paralegal framing.
# 2. A recurring pattern in accepted questions was answers that restate a
#    descriptive label instead of directly answering the question's
#    grammatical form. The verification checks don't catch that — addressed
#    with the explicit direct-answer rule below.
# 3. STRUCTURAL rework (2026-08-13): the old design asked for a near-verbatim
#    answer in the LAST passage, but Stage 3's chain-dependency check masked
#    an intermediate hop while keeping first+last — so first+last always
#    contained the answer and every genuine 3+ hop question failed the check,
#    while 2-hop paths skipped it untested. The design is now
#    definitional-dependency: an earlier passage DEFINES a term/rate/party, the
#    last passage APPLIES it opaquely (or carries a figure), and the answer is
#    the defined value (from an earlier passage) or a derived value (nowhere
#    verbatim). The last passage must NOT contain the answer — that single
#    rule makes the question genuinely multi-hop under Stage 3's two solver
#    checks (full-chain must answer; last-passage-alone must fail).
_ANSWER_FIRST_PROMPT_LEGAL = """You are a contract reviewer writing a multi-hop training question about ONE
legal contract (not comparing it to any other document).

You are given passages from this single contract, in order. Build the question
on a REAL cross-reference inside the contract: an EARLIER passage defines a
term, rate, threshold, deadline, or party; the LAST passage APPLIES that
definition opaquely ("such rate", "the cure period", "as set forth in Section
4.1", "the foregoing") or combines it with a figure in the last passage.
Answering correctly requires joining the definition with the last passage -
exactly the cross-reference a real contract reviewer has to track.

Rules:
- Ask the way a paralegal reviewing this contract actually would (concrete, practical).
- The ANSWER is the DEFINED VALUE or a DERIVED value:
    * DEFINED VALUE: the number, date, rate, period, or party established in an
      EARLIER passage. It must be a SHORT (<= 8 words), EXACT, reproducible span -
      a solver must be able to find it word-for-word in the earlier passage
      ("30 days", "5%", "$1,000", a defined term, a party name). Do not paraphrase it.
    * DERIVED value (when the last passage applies a rate/threshold to a figure):
      the ANSWER is the RESULT (a number/date) that appears in NO passage verbatim -
      a solver computes it from the earlier rate/threshold and the last passage's figure.
- CRITICAL - the LAST passage must NOT contain the answer. The last passage only
  APPLIES the earlier definition with an opaque reference, or carries the figure
  that the earlier rate/threshold is applied to. A reader given ONLY the last
  passage must NOT be able to answer - they must go back to the definition.
- Do not name the answer in the question. Do not quote either passage verbatim in
  the QUESTION - paraphrase so no keyword match can shortcut. PARAPHRASE THE
  QUESTION, NOT THE ANSWER.
- Self-contained, no dangling pronouns.

Example (defined value):
  Passage 1: "Licensee shall pay Licensor a Royalty."
  Passage 2: "'Royalty' means 5% of Licensee's Net Sales."
  Passage 3: "The Royalty due each quarter is the Royalty Rate applied to Net
  Sales, payable within 30 days of quarter-end."
  Question: "How much of their Net Sales must the Licensee pay each quarter?"
  Answer: "5%"   (defined in Passage 2; Passage 3 alone cannot answer)

Example (derived value):
  Passage 1: "Company shall indemnify Executive for covered losses."
  Passage 2: "Each indemnity claim is subject to a $1,000 deductible."
  Passage 3: "Executive's covered loss for the Q3 claim is $5,000."
  Question: "How much must Company reimburse Executive for the Q3 claim?"
  Answer: "$4,000"   ($5,000 minus the $1,000 deductible; appears nowhere verbatim)

Passages (in traversal order):
{path_text}

Respond ONLY with JSON:
{{"answer": "<short exact span from an earlier passage, or computed value>", "question": "<the question>", "reasoning": "<one sentence: which earlier passage defines the term/rate and how the last passage applies it>"}}"""


def _path_text(path: ReasoningPath, chunks_by_id) -> str:
    lines = []
    for i, cid in enumerate(path.chunk_ids):
        ch = chunks_by_id.get(cid)
        txt = ch.text if ch else ""
        lines.append(f"[Passage {i+1}] {txt}")
    return "\n\n".join(lines)


def generate_qa(
    path: ReasoningPath,
    chunks_by_id,
    llm,
    *,
    stage="stage2",
    prompt_template: str = _ANSWER_FIRST_PROMPT,
) -> QASample:
    """Generate one answer-first QA pair from a path.

    `prompt_template`: defaults to the generic (HotpotQA/FinDER-validated)
    prompt. Pass `_ANSWER_FIRST_PROMPT_LEGAL` for the CUAD/legal-contract
    track (NLLP_SynthData.md Part A) — domain-appropriate wording, same
    generation function, no structural change.

    Returns a QASample with question/answer/gold_reasoning populated and the
    generation LLM call recorded. Verification fields are left for Stage 3.
    """
    msg = _gen_message(path, chunks_by_id, prompt_template)
    text, rec = llm.chat(
        msg, stage=stage, purpose="generate_answer_first", temperature=0.3, max_tokens=12288
    )
    return _parse_gen(text, rec, path)


def _gen_message(path: ReasoningPath, chunks_by_id, prompt_template: str = _ANSWER_FIRST_PROMPT):
    return [
        {
            "role": "user",
            "content": prompt_template.format(path_text=_path_text(path, chunks_by_id)),
        }
    ]


def _parse_gen(text, rec, path: ReasoningPath) -> QASample:
    sample = QASample(
        sample_id=f"qa_{path.path_id}",
        path=path,
        gold_chunk_ids=list(path.chunk_ids),
        hop_count=path.hop_count,
        question_type=path.question_type,
        specificity=path.specificity,
        llm_calls=[rec],
        created_at=datetime.now(UTC).isoformat(),
    )
    try:
        data = extract_json(text)
        sample.answer = data.get("answer", "").strip()
        sample.question = data.get("question", "").strip()
        sample.gold_reasoning = data.get("reasoning", "").strip()
    except Exception:
        sample.answer, sample.question, sample.gold_reasoning = "", "", ""
        sample.status = "rejected"
        sample.discard_reason = "generate_parse_failure"
    if not sample.question or not sample.answer:
        sample.status = "rejected"
        if not sample.discard_reason:
            sample.discard_reason = "empty_qa"
    return sample


def generate_batch(
    paths: list[ReasoningPath],
    chunks_by_id,
    llm,
    *,
    stage="stage2",
    prompt_template: str = _ANSWER_FIRST_PROMPT,
    max_workers=4,
    checkpoint_path: str = None,
) -> list[QASample]:
    """Generate QA for every path, `max_workers`-way concurrent. Returns samples in path order.

    `prompt_template`: see `generate_qa` — pass `_ANSWER_FIRST_PROMPT_LEGAL`
    for the CUAD/legal track.

    `checkpoint_path`: crash-safe incremental checkpoint (see checkpoint.py).
    Completed samples are appended as they finish; a restart reuses them and
    generates only the missing paths, so a crash mid-stage never re-spends the
    whole stage. The checkpoint is keyed to (model, prompt template) — changing
    either invalidates it, per the stale-artifact rule.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .checkpoint import CheckpointLog, stage_meta
    from .llm_client import _safe_max_workers

    done: dict[str, QASample] = {}
    ckpt = None
    if checkpoint_path:
        ckpt = CheckpointLog(
            checkpoint_path,
            meta=stage_meta(stage, model=getattr(llm, "model", "?"), prompt=prompt_template),
        )
        for rec in ckpt.load():
            if rec is not None and getattr(rec, "sample_id", None):
                done[rec.sample_id] = rec

    pending = [p for p in paths if f"qa_{p.path_id}" not in done]
    if ckpt and done:
        logger.info(
            f"[stage2] resumed {len(done)}/{len(paths)} from checkpoint, "
            f"generating {len(pending)}"
        )

    results: dict[str, QASample] = dict(done)
    if pending:
        workers = _safe_max_workers(max(1, max_workers))
        # Robust batch (2026-08-14): never let a worker exception or a hung LLM
        # call freeze the stage — poll with wait(timeout=), catch per-future
        # errors, abandon futures that miss a hard deadline (see verify_batch).
        import time as _time
        from concurrent.futures import wait as _wait

        ex = ThreadPoolExecutor(max_workers=workers)
        futs = {
            ex.submit(
                generate_qa, p, chunks_by_id, llm, stage=stage, prompt_template=prompt_template
            ): p
            for p in pending
        }
        remaining = set(futs)
        # Scale the batch deadline to workload (see _batch_budget in verify.py)
        # — a fixed budget abandoned 57% of a 5,271-sample batch in the
        # 2026-08-13 4K run (worker_error_timeout was the top rejection cause).
        _expected_s = len(pending) * 15 / max(1, workers)
        deadline = _time.time() + max(15 * 60, min(6 * 60 * 60, int(2.5 * _expected_s)))
        try:
            while remaining and _time.time() < deadline:
                done_now, remaining = _wait(remaining, timeout=60)
                for fut in done_now:
                    p = futs[fut]
                    try:
                        s = fut.result()
                    except Exception as e:
                        logger.error(f"[stage2] worker error on {p.path_id}: {e!r}")
                        s = QASample(
                            sample_id=f"qa_{p.path_id}",
                            path=p,
                            gold_chunk_ids=list(p.chunk_ids),
                            hop_count=p.hop_count,
                            status="rejected",
                            discard_reason="worker_error",
                        )
                        results[s.sample_id] = s
                        continue
                    results[s.sample_id] = s
                    if ckpt:
                        ckpt.append(s)
            for fut in remaining:
                p = futs[fut]
                s = QASample(
                    sample_id=f"qa_{p.path_id}",
                    path=p,
                    gold_chunk_ids=list(p.chunk_ids),
                    hop_count=p.hop_count,
                    status="rejected",
                    discard_reason="worker_error_timeout",
                )
                results[s.sample_id] = s
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    return [results[f"qa_{p.path_id}"] for p in paths]


# Hard wall-clock budget for a whole generation batch (safety net).
_BATCH_HARD_BUDGET = 60 * 60
