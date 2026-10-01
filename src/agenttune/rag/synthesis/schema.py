"""
Core data model for the synthetic QA generation pipeline.

A single record (`QASample`) flows through every stage and accumulates metadata:
input chunk → graph path → generated Q/A → verification results → difficulty
labels → accept/reject decision. Every record is one row in the final tabular
dump (CSV/Parquet), so the whole pipeline is inspectable as one growing table.

Design mirrors `datagen.py`'s injectable-callable pattern: the model-dependent
parts (LLM, embeddings) are injected, so the pipeline logic is testable with
fakes on CPU with no network/key (see `tests/rag/test_synthesis.py`).

Nothing here touches agenttune's core — new code under one directory, consumed
through the existing `rag` package's public types (`CorpusDocument`, `SearchBackend`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A corpus chunk, grounded by id to its source document.

    Reuses the same id/text/metadata shape as `datagen.Chunk` and the chunks
    `corpus_loader.chunk_corpus` produces, so a real corpus plugs straight in.
    """

    chunk_id: str
    text: str
    doc_id: str = ""
    title: str = ""
    chunk_index: int = 0
    embedding: list[float] | None = None
    entities: list[str] = field(default_factory=list)
    keyphrases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """A typed edge between two chunks in the knowledge graph.

    Edge types (from GRADE + RAGAS, distilled):
      exact      — shared named entity (post exact-equivalence merge)
      contextual — LLM-resolved coreference/alias (GRADE contextual equivalence)
      abstract   — summary/theme cosine similarity (RAGAS MultiHopAbstract)
    """

    source: str
    target: str
    type: str
    weight: float = 1.0
    evidence: str = ""  # the shared entity / theme that justified the edge
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningPath:
    """A sampled multi-hop path over the chunk graph.

    `chunk_ids` is the ordered list of chunks a question must traverse.
    Tagged with hop count, question type (2Wiki typology), and specificity
    (RAGAS specific vs abstract split).
    """

    path_id: str
    chunk_ids: list[str]
    edges: list[GraphEdge] = field(default_factory=list)
    hop_count: int = 0
    question_type: str = "bridge"  # bridge | comparison | compositional | aggregation
    specificity: str = "specific"  # specific | abstract
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMCallRecord:
    """Metadata for a single LLM call — for reproducibility + cost accounting.

    One row per API request/response. Aggregated into `QASample.llm_calls`.
    """

    stage: str  # which pipeline stage made the call
    purpose: str  # e.g. "generate_answer", "verify_chain_dependency"
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    request_preview: str = ""  # truncated prompt, for debug
    response_preview: str = ""  # truncated response
    success: bool = True
    error: str = ""
    timestamp: str = ""


@dataclass
class QASample:
    """One question/answer pair with full provenance, the unit of the dataset.

    Starts populated at Stage 2 (generation); verification (Stage 3) and
    difficulty (Stage 4) append fields. `status` is the final accept/reject
    decision; `discard_reason` records why a question was dropped.
    """

    sample_id: str
    path: ReasoningPath | None = None
    question: str = ""
    answer: str = ""
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_reasoning: str = ""  # human/LLM-readable chain
    question_type: str = "bridge"  # surfaced from path for the dumps
    specificity: str = "specific"

    # Stage 3 — verification
    retrieval_necessity: float | None = None  # anchor-chunk rank in top-k (audit)
    retrieval_necessity_pass: bool | None = (
        None  # one-shot-leak test: top-1 doesn't carry the answer
    )
    chain_dependency: float | None = (
        None  # solver relaxed-F1 with ONLY the last chunk (multi-hop test)
    )
    chain_dependency_pass: bool | None = (
        None  # last-chunk-alone solver must NOT reproduce the answer
    )
    answer_grounded_pass: bool | None = (
        None  # full-chain solver reproduces the answer (relaxed F1 >= 0.5)
    )
    answer_grounded_span: bool | None = None  # audit: old token-overlap check, NOT a gate anymore
    revision_count: int = 0
    original_question: str = ""  # pre-revision version, if revised

    # Stage 4 — difficulty
    hop_count: int = 0
    retrieval_difficulty: float | None = None  # 1 - power-mean sim (GRADE)
    difficulty_cell: str = ""  # e.g. "3hop_hard"

    # Stage 5 — per-sample eval scores (ARES answerability F1 + RAGAS faithfulness)
    answerability_f1: float | None = None  # solver F1 vs gold answer (full path, standard SQuAD)
    answerability_f1_relaxed: float | None = None  # same, with legal-entity-tolerant scoring
    answerable: bool | None = None  # quality gate: relaxed F1 >= 0.5 with full path
    faithfulness_score: float | None = None  # LLM-judge groundedness 0..1

    # Final
    status: str = "pending"  # pending | accepted | rejected | revised
    discard_reason: str = ""

    # Provenance / cost
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    created_at: str = ""

    def to_row(self) -> dict[str, Any]:
        """Flatten to a dict for CSV/Parquet. List/dict fields are JSON-stringified."""
        import json

        d = asdict(self)
        # JSON-stringify the non-scalar fields so pandas can write them to CSV
        for k in ("gold_chunk_ids", "llm_calls", "entities"):
            if k in d:
                d[k] = json.dumps(d[k], ensure_ascii=False)
        if d.get("path") is not None:
            d["path_chunk_ids"] = json.dumps(
                d["path"].get("chunk_ids", []) if isinstance(d["path"], dict) else [],
                ensure_ascii=False,
            )
        else:
            d["path_chunk_ids"] = "[]"
        # collapse path into scalar columns
        d.pop("path", None)
        return d
