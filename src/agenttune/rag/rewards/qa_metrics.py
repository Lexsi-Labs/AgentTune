"""
QA scoring — wraps HuggingFace's `evaluate.load("squad")` metric instead of
hand-rolling normalize_answer/EM/F1. `evaluate>=0.4.0` is already a declared
agenttune dependency; the SQuAD metric implements the same normalization
scheme HotpotQA's own official eval uses.
"""

import re
import threading

_SQUAD_METRIC = None
_SQUAD_METRIC_LOCK = threading.Lock()


def _get_squad_metric():
    """Lazily load the SQuAD metric, thread-safely (double-checked locking).

    `evaluate.load("squad")` downloads/initializes datasets artifacts into
    the HF cache — two concurrent initializations race on the same cache
    file (one thread mmaps an arrow file while another truncates/rewrites
    it) and crash the process with SIGBUS. The pipeline now verifies
    samples in parallel (verify_batch), so the first `f1_score` call can
    arrive from many threads at once — the lock serializes initialization;
    subsequent calls reuse the in-memory metric.
    """
    global _SQUAD_METRIC
    if _SQUAD_METRIC is None:
        with _SQUAD_METRIC_LOCK:
            if _SQUAD_METRIC is None:
                import evaluate

                _SQUAD_METRIC = evaluate.load("squad")
    return _SQUAD_METRIC


def _compute(pred: str, gold: str) -> dict:
    """SQuAD metric.compute(), serialized with the same lock as the load.

    `evaluate`'s compute() spawns a datasets ArrowWriter per call (temp
    artifact file, mmap'd); concurrent compute() calls from verify threads
    raced on the artifact path and SIGBUS'd (observed: Bus error in
    fsspec/_open during the full-suite run). Scoring is microseconds of CPU
    per call — serializing costs nothing; LLM calls (the real cost) are
    unaffected.
    """
    metric = _get_squad_metric()
    with _SQUAD_METRIC_LOCK:
        return metric.compute(
            predictions=[{"id": "0", "prediction_text": pred}],
            references=[{"id": "0", "answers": {"text": [gold], "answer_start": [0]}}],
        )


def exact_match_score(pred: str, gold: str) -> float:
    return _compute(pred, gold)["exact_match"] / 100.0


def f1_score(pred: str, gold: str) -> float:
    return _compute(pred, gold)["f1"] / 100.0


# ── Relaxed scoring for short legal answers ─────────────────────────────────
# SQuAD normalization is brittle for clause-entity answers: "The ARC Group,
# Inc." vs "ARC Group", "Tel-Aviv" vs "Tel Aviv", "30 days" vs "30 days after
# notice" all score 0.0 even when the solver answered correctly. For our
# answerability gate we compare on a legal-tolerant normalizer instead:
# lowercase, strip punctuation, drop leading articles and corporate-suffix
# tokens, fold hyphens to spaces. The strict SQuAD scores stay reported
# alongside (answerability_f1) — relaxed is the *gate* (answerable).
_CORPUS_SUFFIX_RE = re.compile(
    r"\b(inc|llc|ltd|corp|corporation|co|company|gmbh|s\.a\.?|plc)\b\.?$", re.IGNORECASE
)
_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _relaxed_tokens(text: str) -> list[str]:
    t = text.lower().strip()
    t = _PUNCT_RE.sub(" ", t)  # punctuation → spaces (folds hyphens too)
    t = _WS_RE.sub(" ", t).strip()
    t = _LEADING_ARTICLE_RE.sub("", t)  # drop leading "the"/"a"/"an"
    t = _CORPUS_SUFFIX_RE.sub("", t)  # drop trailing ", Inc." style suffixes
    t = _WS_RE.sub(" ", t).strip()
    return t.split()


def relaxed_f1_score(pred: str, gold: str) -> float:
    """Token F1 under the legal-tolerant normalizer.

    Covers the observed failure modes (party-name suffixes, city/state
    hyphenation, leading articles, role-phrase substitutions leave partial
    credit). Still 0.0 when nothing overlaps — it loosens normalization,
    not correctness.
    """
    p = _relaxed_tokens(pred)
    g = _relaxed_tokens(gold)
    if not p or not g:
        return 0.0
    overlap = len(set(p) & set(g))
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def extract_answer_tag(completion: str) -> str:
    """Pulls <answer>...</answer> out of a completion; falls back to the full text.
    Bespoke to our own prompt format, not something a library provides."""
    text = completion if isinstance(completion, str) else str(completion)
    match = _ANSWER_TAG_RE.search(text)
    return match.group(1).strip() if match else text.strip()
