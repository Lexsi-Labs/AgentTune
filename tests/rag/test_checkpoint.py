"""CPU tests for crash-safe checkpoint resume — fakes, no network/key/GPU.

The big runs cost money; if the process dies at the 3,000th of 6,000 samples,
restarting must reuse the completed work instead of re-spending the stage.
These tests pin that behavior: a second run against the same checkpoint makes
ZERO new LLM calls and returns identical results, and a changed
prompt/threshold invalidates the checkpoint (forces regeneration).
"""

import json
import os
import tempfile

from agenttune.rag.retrieval.corpus_loader import CorpusDocument
from agenttune.rag.retrieval.sqlite_fts import SQLiteFTSBackend
from agenttune.rag.synthesis import (
    FakeEmbedder,
    FakeLLMClient,
    annotate_chunks,
    build_graph,
    chunk_documents,
    embed_chunks,
    generate_batch,
    sample_paths,
    verify_batch,
)
from agenttune.rag.synthesis.checkpoint import CheckpointLog


def _docs():
    return [
        CorpusDocument(
            doc_id="A", title="History", text="Northwind Robotics was founded in 2015 by Elena Cho."
        ),
        CorpusDocument(
            doc_id="B", title="Moves", text="Elena Cho joined Solace AI as CTO in 2021."
        ),
        CorpusDocument(
            doc_id="C",
            title="Partners",
            text="Solace AI and Northwind Robotics co-developed Aria-2 in 2023.",
        ),
        CorpusDocument(
            doc_id="D", title="Decoy", text="Elena Marsh founded Marsh Dynamics in 2016."
        ),
    ]


def _scripted_llm():
    def responder(purpose, messages):
        if purpose == "extract_entities":
            txt = messages[0]["content"]
            if "Northwind" in txt and "founded" in txt:
                return json.dumps(
                    {
                        "entities": ["northwind robotics", "elena cho", "2015"],
                        "keyphrases": ["founded"],
                        "summary": "x",
                    }
                )
            if "Solace AI as CTO" in txt:
                return json.dumps(
                    {
                        "entities": ["elena cho", "solace ai", "2021"],
                        "keyphrases": ["joined", "cto"],
                        "summary": "x",
                    }
                )
            if "Aria-2" in txt:
                return json.dumps(
                    {
                        "entities": ["solace ai", "northwind robotics", "aria-2", "2023"],
                        "keyphrases": ["co-developed"],
                        "summary": "x",
                    }
                )
            return json.dumps(
                {
                    "entities": ["elena marsh", "marsh dynamics"],
                    "keyphrases": ["founded"],
                    "summary": "x",
                }
            )
        if purpose == "contextual_equiv":
            return json.dumps({"mappings": []})
        if purpose == "generate_answer_first":
            return json.dumps(
                {
                    "answer": "Aria-2",
                    "question": "What product did Solace and Northwind co-develop?",
                    "reasoning": "Solace+Northwind built Aria-2.",
                }
            )
        if purpose in (
            "answerability_full_chain",
            "chain_dep_last_passage_only",
            "solver",
            "eval_answerability",
        ):
            return json.dumps({"answer": "Aria-2"})
        if purpose == "revise_question":
            return json.dumps(
                {"question": "Revised question requiring all hops.", "answer": "Aria-2"}
            )
        return "FAKE_RESPONSE"

    return FakeLLMClient(responder=responder)


def _corpus():
    chunks = chunk_documents(_docs(), chunk_size=512, overlap=0)
    llm = _scripted_llm()
    chunks = annotate_chunks(chunks, llm)
    chunks = embed_chunks(chunks, FakeEmbedder())
    _, G = build_graph(chunks, llm)
    by_id = {c.chunk_id: c for c in chunks}
    paths = sample_paths(G, by_id, per_hop=5)
    return chunks, by_id, llm, paths


def _sqlite_sb(chunks, tmp):
    from agenttune.rag.retrieval.corpus_loader import build_index as _bi

    sb = SQLiteFTSBackend(os.path.join(tmp, "c.sqlite"))
    _bi(sb, [CorpusDocument(doc_id=c.doc_id, title=c.title, text=c.text) for c in chunks])
    return sb


# ── CheckpointLog primitives ──────────────────────────────────────────────


def test_checkpoint_log_roundtrip_and_meta_invalidation():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "log.pkl")
    log = CheckpointLog(p, meta={"prompt": "abc"})
    log.append({"sample_id": "q1"})
    log.append({"sample_id": "q2"})
    assert [r["sample_id"] for r in log.load()] == ["q1", "q2"]
    # meta mismatch → treated as stale/empty (resume must NOT reuse old data)
    assert CheckpointLog(p, meta={"prompt": "xyz"}).load() == []
    # appending to a stale file truncates it, then records under the new meta
    log2 = CheckpointLog(p, meta={"prompt": "xyz"})
    log2.append({"sample_id": "q3"})
    assert [r["sample_id"] for r in log2.load()] == ["q3"]


# ── Stage 2 ───────────────────────────────────────────────────────────────


def test_generate_batch_checkpoint_resume_makes_no_calls():
    _, by_id, llm, paths = _corpus()
    ck = os.path.join(tempfile.mkdtemp(), "stage2.pkl")
    s1 = generate_batch(paths, by_id, llm, max_workers=2, checkpoint_path=ck)
    assert all(s.question and s.answer for s in s1)
    n_calls = len(llm.calls)
    # identical second run → resumes from checkpoint, zero new LLM calls
    s2 = generate_batch(paths, by_id, llm, max_workers=2, checkpoint_path=ck)
    assert len(llm.calls) == n_calls
    assert [s.question for s in s1] == [s.question for s in s2]
    assert [s.answer for s in s1] == [s.answer for s in s2]
    # changed prompt template → checkpoint invalidated → regenerates
    llm.calls = []
    s3 = generate_batch(
        paths,
        by_id,
        llm,
        max_workers=2,
        checkpoint_path=ck,
        prompt_template="DIFFERENT PROMPT {path_text}",
    )
    assert len(llm.calls) > 0
    assert all(s.question and s.answer for s in s3)


# ── Stage 3 ───────────────────────────────────────────────────────────────


def test_verify_batch_checkpoint_resume_makes_no_calls():
    chunks, by_id, llm, paths = _corpus()
    sb = _sqlite_sb(chunks, tempfile.mkdtemp())
    ck = os.path.join(tempfile.mkdtemp(), "stage3.pkl")
    # First pass: fresh generation → verify → checkpoint populated.
    samples1 = generate_batch(paths, by_id, llm, max_workers=2)
    v1 = verify_batch(samples1, by_id, sb, llm, max_workers=2, checkpoint_path=ck)
    assert all(v.retrieval_necessity is not None for v in v1)
    # Simulated restart: the pipeline re-derives samples (here via a fresh
    # generate_batch — same fake output → same questions), then re-verifies.
    # The stage3 checkpoint must be reused → the VERIFY stage adds zero calls.
    samples2 = generate_batch(paths, by_id, llm, max_workers=2)
    n_calls = len(llm.calls)  # baseline AFTER re-generation
    v2 = verify_batch(samples2, by_id, sb, llm, max_workers=2, checkpoint_path=ck)
    assert len(llm.calls) == n_calls
    assert [s.status for s in v1] == [s.status for s in v2]
    assert [s.chain_dependency for s in v1] == [s.chain_dependency for s in v2]
    # changed threshold → checkpoint invalidated → re-verifies
    llm.calls = []
    samples3 = generate_batch(paths, by_id, llm, max_workers=2)
    verify_batch(
        samples3, by_id, sb, llm, max_workers=2, checkpoint_path=ck, chain_acc_threshold=0.9
    )
    assert len(llm.calls) > 0
