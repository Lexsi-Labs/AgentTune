"""Docs -> QA data generation + difficulty curriculum (RAG P3).

Make agentic-RAG training work on a user's *own* documents, where no ready-made
question set exists. Two steps, each with the model-dependent part behind an
injectable callable so the logic is testable without a GPU or network:

1. **Generate** grounded (question, answer) pairs from corpus chunks
   (``generate_qa_from_corpus`` with a ``generator`` callable — wrap an LLM).
2. **Curriculum** — probe the base model to split questions it already answers
   (*easy*) from genuine knowledge gaps (*hard*), then train on a balanced mix
   and/or an easy->hard order.

References: IKEA (2505.07596) knowledge-boundary easy/hard split trained ~1:1;
R1-Searcher / curriculum + contamination gates; FrugalRAG (2507.07634).

This module is self-contained: it carries its own lightweight ``Chunk`` dataclass
and SQuAD-style EM/F1 scorer so the QA-generation logic has no dependency on the
RAG package's training retrieval/reward stack (which uses a separate
``SearchBackend`` / ``qa_metrics`` design under ``retrieval/`` and ``rewards/``).
Only its two imports were inlined so the skeleton modules it used to import from
could be dropped without changing this logic.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

__all__ = [
    "Chunk",
    "QAPair",
    "answer_correctness_reward",
    "generate_qa_from_corpus",
    "label_difficulty",
    "balance_by_difficulty",
    "sort_by_difficulty",
]


# ─────────────────────────────────────────────────────────────────────────────
# Self-contained types + scorer (inlined from the dropped RAG skeleton so this
# module stands alone).
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    """A retrieved passage, grounded by ``id`` to its source document."""

    id: str
    text: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD-style)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    return _WS.sub(" ", text).strip()


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold (SQuAD-style)."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    """1.0 if normalized prediction equals normalized gold, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def answer_correctness_reward(prediction: str, gold_answer: str, mode: str = "f1") -> float:
    """Verifiable answer correctness. ``mode`` is ``"f1"`` or ``"em"``."""
    from agenttune.utils.score_logger import log_score

    if mode == "em":
        score = exact_match(prediction, gold_answer)
        log_score(
            "datagen.answer_correctness_reward",
            score,
            reasons=[f"exact_match({prediction!r}, {gold_answer!r}) -> {score}"],
        )
        return score
    if mode == "f1":
        score = token_f1(prediction, gold_answer)
        log_score(
            "datagen.answer_correctness_reward",
            score,
            reasons=[f"token_f1({prediction!r}, {gold_answer!r}) -> {score}"],
        )
        return score
    raise ValueError(f"Unknown mode {mode!r}; use 'f1' or 'em'.")


# generator(chunk_text) -> list of (question, answer) pairs
QAGenerator = Callable[[str], Sequence[tuple[str, str]]]
# solver(question) -> answer string
Solver = Callable[[str], str]


@dataclass
class QAPair:
    question: str
    answer: str
    gold_chunk_ids: list[str] = field(default_factory=list)
    difficulty: str | None = None  # "easy" | "hard" | None
    metadata: dict = field(default_factory=dict)


def generate_qa_from_corpus(chunks: Sequence[Chunk], generator: QAGenerator) -> list[QAPair]:
    """Generate grounded QA pairs from corpus chunks.

    ``generator`` is called per chunk and returns (question, answer) pairs; each
    pair is grounded to its source chunk id (``gold_chunk_ids``), which the
    reward layer uses for gold-chunk coverage.
    """
    pairs: list[QAPair] = []
    for chunk in chunks:
        for question, answer in generator(chunk.text):
            pairs.append(
                QAPair(
                    question=question,
                    answer=answer,
                    gold_chunk_ids=[chunk.id],
                    metadata={"source_text": chunk.text},
                )
            )
    return pairs


def label_difficulty(
    qa_pairs: Sequence[QAPair],
    solver: Solver,
    *,
    mode: str = "f1",
    n_probes: int = 1,
    pass_threshold: float = 0.5,
    correct_at: float = 0.5,
) -> list[QAPair]:
    """Label each pair *easy* or *hard* by probing a base ``solver``.

    Probe ``solver`` ``n_probes`` times per question; a probe counts as correct
    when its answer scores >= ``correct_at`` against the gold answer. If the pass
    rate >= ``pass_threshold`` the model already knows it (*easy*); otherwise it
    is a knowledge gap worth training on (*hard*). Mutates and returns the pairs.
    """
    for pair in qa_pairs:
        passes = 0
        for _ in range(n_probes):
            pred = solver(pair.question)
            if answer_correctness_reward(pred, pair.answer, mode=mode) >= correct_at:
                passes += 1
        pass_rate = passes / n_probes if n_probes else 0.0
        pair.difficulty = "easy" if pass_rate >= pass_threshold else "hard"
    return list(qa_pairs)


def balance_by_difficulty(
    qa_pairs: Sequence[QAPair], ratio: tuple[int, int] = (1, 1)
) -> list[QAPair]:
    """Return an easy/hard-balanced subset in the given ``ratio`` (easy:hard).

    IKEA trains on a ~1:1 mix so the model neither never-searches (all-easy) nor
    over-searches (all-hard). The result is capped by the scarcer class.
    """
    easy = [p for p in qa_pairs if p.difficulty == "easy"]
    hard = [p for p in qa_pairs if p.difficulty == "hard"]
    re, rh = ratio
    if re <= 0 or rh <= 0:
        raise ValueError("ratio components must be positive")
    # largest n with n*re <= len(easy) and n*rh <= len(hard)
    units = min(len(easy) // re, len(hard) // rh)
    return easy[: units * re] + hard[: units * rh]


def sort_by_difficulty(qa_pairs: Sequence[QAPair]) -> list[QAPair]:
    """Order easy -> hard (curriculum). Unlabeled pairs sort last."""
    order = {"easy": 0, "hard": 1}
    return sorted(qa_pairs, key=lambda p: order.get(p.difficulty, 2))
