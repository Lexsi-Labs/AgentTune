"""
agenttune_eval.py  —  AgentTune Unified Evaluator
==================================================
Supports three built-in use cases grounded in the real dataset schemas:

  email_search   corbt/enron_emails_sample_questions
                 answers: 1 word → 3 paragraphs (factual retrieval)
                 extras:  message_ids list, inbox_address, query_date

  finqa          rLLM/rLLM-FinQA-Dataset
                 answers: 500-3000 word filled templates with tables
                 extras:  explanation, question_type, table_name, company

  file_ingestion agenttune synthetic dataset
                 answers: aggregated numbers / short prose
                 extras:  sample_dir, quarter, year

  generic        any use case: coding, summarization, math, QA, etc.
                 uses universal_score as primary metric

Quick start
-----------
    from agenttune_eval import run_eval
    report = run_eval(
        model_path="./output/email_search_agent",
        use_case="email_search",
        dataset=val_dataset,
        tools=[search_inbox, read_email, list_senders],
    )
    print(report)
    report.save("./reports")
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# RESULT CONTAINERS
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class Sample:
    """Everything about one evaluated example."""

    idx: int
    question: str
    gold: str
    predicted: str
    tool_calls: list[str] = field(default_factory=list)
    n_tools: int = 0
    latency_ms: float = 0.0
    scores: dict = field(default_factory=dict)
    error: str | None = None

    # Per-row extras attached in run_eval loop
    # s._message_ids  — list[str]  for email_search
    # s._sample_dir   — str        for file_ingestion

    @property
    def passed(self) -> bool:
        """
        A sample 'passes' if either:
          - its primary answer metric >= 0.5 (checks token_f1,
            numerical_recall, exact_match_normalized, universal_score), OR
          - its mean score across all metrics >= 0.45
        """
        primary = max(
            self.scores.get("token_f1", 0.0),
            self.scores.get("numerical_recall", 0.0),
            self.scores.get("exact_match_normalized", 0.0),
            self.scores.get("universal_score", 0.0),  # ← added
        )
        mean = sum(self.scores.values()) / max(len(self.scores), 1)
        return primary >= 0.5 or mean >= 0.45


@dataclass
class Report:
    """Aggregated results across all samples."""

    use_case: str
    model: str
    n_samples: int
    n_passed: int
    n_errors: int
    means: dict
    stds: dict
    samples: list[Sample]
    duration_s: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def pass_rate(self) -> float:
        return self.n_passed / max(self.n_samples, 1)

    def __str__(self) -> str:
        w = 60
        sep = "─" * w
        lines = [
            f"┌{sep}┐",
            f"│  AgentTune Eval  ·  {self.use_case:<37}│",
            f"├{sep}┤",
            f"│  Model     : {Path(self.model).name:<45}│",
            f"│  Samples   : {self.n_samples:<45}│",
            f"│  Pass rate : {self.pass_rate*100:.1f}%{'':<43}│",
            f"│  Errors    : {self.n_errors:<45}│",
            f"│  Duration  : {self.duration_s:.1f}s{'':<44}│",
            f"├{sep}┤",
            f"│  {'Metric':<30}{'Mean':>9}  {'Std':>9}          │",
            f"│  {'──────':<30}{'────':>9}  {'───':>9}          │",
        ]
        for name, mean in self.means.items():
            std = self.stds.get(name, 0.0)
            lines.append(f"│  {name:<30}{mean:>9.3f}  {std:>9.3f}          │")
        lines.append(f"└{sep}┘")
        return "\n".join(lines)

    def save(self, directory: str = "./eval_reports") -> str:
        Path(directory).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(directory) / f"{self.use_case}_{ts}.json"
        data = {
            "use_case": self.use_case,
            "model": self.model,
            "timestamp": self.timestamp,
            "n_samples": self.n_samples,
            "pass_rate": self.pass_rate,
            "n_errors": self.n_errors,
            "duration_s": self.duration_s,
            "means": self.means,
            "stds": self.stds,
            "samples": [
                {
                    "idx": s.idx,
                    "question": s.question[:200],
                    "gold": s.gold[:500],
                    "predicted": s.predicted[:800],
                    "n_tools": s.n_tools,
                    "tool_calls": s.tool_calls,
                    "scores": s.scores,
                    "error": s.error,
                }
                for s in self.samples
            ],
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"  💾  Report saved → {path}")
        return str(path)

    def failures(self, top_n: int = 10) -> list[Sample]:
        """Return the top_n worst-scoring samples by primary metric."""

        def primary(s: Sample) -> float:
            return max(
                s.scores.get("token_f1", 0.0),
                s.scores.get("numerical_recall", 0.0),
                s.scores.get("exact_match_normalized", 0.0),
                s.scores.get("universal_score", 0.0),  # ← added
            )

        return sorted(self.samples, key=primary)[:top_n]


# ═════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════


def _norm(s: str) -> str:
    """Normalise for loose comparison: lowercase, strip $,% spaces."""
    return re.sub(r"[$,%\s]", "", s.lower().strip())


def _extract_answer(text: str) -> str:
    """Pull content from <answer>...</answer> if present, else return full text."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.I | re.S)
    return m.group(1).strip() if m else text.strip()


# ═════════════════════════════════════════════════════════════════════════════
# METRIC CLASS
# ═════════════════════════════════════════════════════════════════════════════


class _Metric:
    def __init__(self, name: str, fn: Callable[[Sample], float]):
        self.name = name
        self._fn = fn

    def __call__(self, s: Sample) -> float:
        try:
            return float(self._fn(s))
        except Exception:
            return 0.0

    def __repr__(self) -> str:
        return f"Metric({self.name!r})"


# ═════════════════════════════════════════════════════════════════════════════
# ── SHARED METRICS ─────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════


def answer_format() -> _Metric:
    """1.0 if response contains a non-trivial <answer>…</answer> block."""

    def _score(s: Sample) -> float:
        return 1.0 if re.search(r"<answer>.{3,}</answer>", s.predicted, re.I | re.S) else 0.0

    return _Metric("answer_format", _score)


def tool_use(tiers: dict | None = None) -> _Metric:
    """
    Tiered score based on number of tool calls made.
    Default: 0→0.0, 1→0.4, 2→0.7, 3+→1.0
    """
    _tiers = tiers or {0: 0.0, 1: 0.4, 2: 0.7}

    def _score(s: Sample) -> float:
        return _tiers.get(s.n_tools, 1.0)

    return _Metric("tool_use", _score)


def specific_tools(required: list[str], name: str = "specific_tools") -> _Metric:
    """Fraction of required tool names actually called."""

    def _score(s: Sample) -> float:
        if not required:
            return 1.0
        hits = sum(1 for t in required if any(t in c for c in s.tool_calls))
        return hits / len(required)

    return _Metric(name, _score)


def metric(name: str, fn: Callable[[Sample], float]) -> _Metric:
    """Create a one-off custom metric from any callable."""
    return _Metric(name, fn)


# ═════════════════════════════════════════════════════════════════════════════
# ── UNIVERSAL METRIC (works for ANY use case) ──────────────────────────────
#
# Why a universal metric:
#   Different use cases (coding, math, summarization, QA, email, finance)
#   have very different answer shapes. A single metric that handles all of
#   them robustly must:
#     1. Detect answer type automatically (numeric / code / prose)
#     2. Apply the most appropriate sub-scorer for that type
#     3. Fall back gracefully when type is ambiguous
#
#   universal_score is a WEIGHTED BLEND of three sub-scores:
#
#   a) token_f1_component     — always computed; handles all lengths.
#                               Primary for prose/QA/summarization.
#
#   b) numeric_component      — activated when gold contains significant
#                               numbers (math, finance, data).
#                               Uses relative-tolerance matching (2%).
#
#   c) code_component         — activated when gold looks like code
#                               (contains def/class/return/for/import etc.)
#                               Checks keyword overlap + structural similarity.
#
#   Final score = weighted average of active components, normalised to [0,1].
#
#   This means:
#     - Math answer "4736"        → numeric_component dominates
#     - Code answer "def foo():"  → code_component dominates
#     - Prose summary             → token_f1_component dominates
#     - Mixed (finqa table+prose) → all three contribute
#
#   The weights are calibrated so no single component can inflate the score:
#     token_f1   weight = 1.0  (always active)
#     numeric    weight = 1.5  (activated only when gold has ≥1 non-zero num)
#     code       weight = 1.2  (activated only when gold has ≥3 code keywords)
# ═════════════════════════════════════════════════════════════════════════════


def universal_score() -> _Metric:
    """
    Universal answer quality metric — works across ALL use cases:
    coding, math, summarization, QA, email retrieval, financial analysis, etc.

    Internally blends three sub-scores based on what the gold answer contains:
      - token_f1_component    : always active  (prose / factual / any)
      - numeric_component     : active when gold has numbers  (math / finance)
      - code_component        : active when gold looks like code

    Returns a single float in [0, 1].

    Examples:
      gold="4736"              pred="4736"               → ~1.0  (numeric)
      gold="def foo(): ..."    pred="def foo(): ..."     → ~1.0  (code)
      gold="The report shows…" pred="The report shows…" → ~1.0  (prose)
      gold="4736"              pred="4800"               → ~0.0  (numeric miss)
    """

    # ── stopwords for token_f1 ─────────────────────────────────────────────
    _STOP = {
        "a",
        "an",
        "the",
        "is",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "and",
        "or",
        "but",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "that",
        "this",
        "it",
        "its",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "not",
        "no",
        "as",
        "if",
        "so",
        "there",
        "their",
        "our",
    }

    # ── code keywords that signal a code answer ───────────────────────────
    _CODE_KW = {
        "def",
        "class",
        "return",
        "import",
        "for",
        "while",
        "if",
        "else",
        "elif",
        "try",
        "except",
        "with",
        "lambda",
        "yield",
        "assert",
        "raise",
        "pass",
        "break",
        "continue",
        "print",
        "function",
        "const",
        "let",
        "var",
        "=>",
        "async",
        "await",
        "struct",
        "fn",
        "#include",
        "public",
        "private",
        "void",
        "int",
        "float",
        "bool",
    }

    def _tokens(text: str) -> list[str]:
        raw = re.findall(r"[a-z0-9_]+", text.lower())
        return [t for t in raw if t not in _STOP and len(t) > 1]

    def _tf1(gold: str, pred: str) -> float:
        """Token-level F1 — prose/factual primary sub-score."""
        gt = _tokens(gold)
        pt = _tokens(pred)
        if not gt:
            return 1.0
        if not pt:
            return 0.0
        gc: dict = {}
        for t in gt:
            gc[t] = gc.get(t, 0) + 1
        pc: dict = {}
        for t in pt:
            pc[t] = pc.get(t, 0) + 1
        overlap = sum(min(gc.get(t, 0), pc.get(t, 0)) for t in pc)
        prec = overlap / len(pt)
        rec = overlap / len(gt)
        if prec + rec == 0:
            return 0.0
        return round(2 * prec * rec / (prec + rec), 4)

    def _parse_nums(text: str) -> list[float]:
        """Extract distinct non-zero numbers from text."""
        raw = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", "").replace("%", ""))
        seen: set = set()
        out: list[float] = []
        for r in raw:
            try:
                v = float(r.replace(",", ""))
                if abs(v) < 1e-9:
                    continue
                k = round(v, 4)
                if k not in seen:
                    seen.add(k)
                    out.append(v)
            except ValueError:
                pass
        return out

    def _numeric(gold: str, pred: str, tol: float = 0.02) -> float:
        """
        Numeric recall with 2% tolerance.
        Returns fraction of gold numbers found in pred.
        """
        gn = _parse_nums(gold)
        if not gn:
            return 1.0  # not a numeric answer — don't penalise
        pn = _parse_nums(pred)
        if not pn:
            return 0.0
        hits = 0
        for g in gn:
            denom = abs(g) if abs(g) > 1e-9 else 1.0
            if any(abs(g - p) / denom <= tol for p in pn):
                hits += 1
        return round(hits / len(gn), 4)

    def _code(gold: str, pred: str) -> float:
        """
        Code similarity sub-score.
        Checks:
          - keyword overlap (shared code keywords / gold keywords)
          - identifier overlap via token_f1 on code tokens
        Returns average of both.
        """
        gold_kws = {w for w in gold.lower().split() if w in _CODE_KW}
        pred_kws = {w for w in pred.lower().split() if w in _CODE_KW}
        if not gold_kws:
            return 1.0  # not a code answer — don't penalise
        kw_overlap = len(gold_kws & pred_kws) / len(gold_kws)
        id_score = _tf1(gold, pred)  # reuse token_f1 for identifier overlap
        return round((kw_overlap + id_score) / 2, 4)

    def _score(s: Sample) -> float:
        gold = s.gold
        pred = _extract_answer(s.predicted)

        # ── detect answer type ────────────────────────────────────────────
        gold_nums = _parse_nums(gold)
        gold_kws = {w for w in gold.lower().split() if w in _CODE_KW}
        is_numeric = len(gold_nums) >= 1
        is_code = len(gold_kws) >= 3

        # ── compute sub-scores ────────────────────────────────────────────
        tf1_score = _tf1(gold, pred)
        num_score = _numeric(gold, pred) if is_numeric else None
        code_score = _code(gold, pred) if is_code else None

        # ── weighted blend ────────────────────────────────────────────────
        total_weight = 1.0
        weighted_sum = tf1_score * 1.0

        if num_score is not None:
            weighted_sum += num_score * 1.5
            total_weight += 1.5

        if code_score is not None:
            weighted_sum += code_score * 1.2
            total_weight += 1.2

        return round(weighted_sum / total_weight, 4)

    return _Metric("universal_score", _score)


# ═════════════════════════════════════════════════════════════════════════════
# ── ENRON / EMAIL-SEARCH METRICS ─────────────────────────────────────────────
#
# Why these metrics for Enron:
#   The dataset (corbt/enron_emails_sample_questions) has answers ranging from
#   a single word ("251832", "Petrous LLC.") to multi-sentence summaries to
#   verbatim legal paragraphs.  A single metric can't cover all cases well.
#
#   token_f1           — works uniformly across all answer lengths. It measures
#                        the overlap of meaningful tokens (ignoring stopwords)
#                        between gold and prediction.  This is the PRIMARY metric.
#
#   exact_match_normalized — catches pure single-value answers where the model
#                        wraps the value in prose ("The number is 251832").
#                        After stripping $, commas, spaces, %, lowercasing.
#
#   substring_match    — catches verbatim legal-clause answers where the model
#                        copies the text almost exactly.
#
#   message_id_recall  — did the model cite the correct email(s)? The dataset
#                        provides ground-truth message_ids per question.
#                        This is the retrieval grounding check.
#
# What we deliberately do NOT use for Enron:
#   fuzzy_match    — gives 0.8 trivially for any answer that *contains* the gold
#                    as a substring, which floods every response with high scores.
#   numerical_accuracy — most Enron answers are not purely numeric.
# ═════════════════════════════════════════════════════════════════════════════


def exact_match_normalized() -> _Metric:
    """
    Hard exact match after normalisation (strip $,,%,whitespace, lowercase,
    trailing punctuation).
    Best for Enron single-value answers: IDs, phone numbers, prices, dates.

    Examples that pass:
      gold="251832"           pred="The confirmation number is 251832."  → 0.0
      gold="251832"           pred="<answer>251832</answer>"             → 1.0
      gold="$1,779,000"       pred="<answer>$1,779,000</answer>"         → 1.0
    Note: exact_match_normalized only scores 1.0 when the ENTIRE extracted
    answer normalises to the gold.  Use alongside token_f1 for prose answers.
    """

    def _n(t: str) -> str:
        return re.sub(r"[$,%\s]", "", t.lower().strip().rstrip(".,;:"))

    def _score(s: Sample) -> float:
        return 1.0 if _n(_extract_answer(s.predicted)) == _n(s.gold) else 0.0

    return _Metric("exact_match_normalized", _score)


def token_f1() -> _Metric:
    """
    Token-level F1 between gold and extracted prediction.

    Tokenises both strings into lowercase alphanumeric tokens, removes a
    fixed English stopword list, then computes:
      precision = overlap_count / pred_token_count
      recall    = overlap_count / gold_token_count
      F1        = harmonic mean

    This is the PRIMARY accuracy metric for Enron because it works uniformly
    across answer lengths (1 word → 3 paragraphs) and handles paraphrasing
    better than exact match.

    Examples:
      gold="8am Monday mornings"  pred="8 AM on Monday mornings" → ~0.75
      gold="251832"               pred="251832"                  → 1.0
      gold="Petrous LLC"          pred="It was Petrous LLC."     → 1.0
    """
    _STOP = {
        "a",
        "an",
        "the",
        "is",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "and",
        "or",
        "but",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "that",
        "this",
        "it",
        "its",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "not",
        "no",
        "as",
        "if",
        "so",
        "there",
        "their",
        "our",
    }

    def _tokens(text: str) -> list[str]:
        raw = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in raw if t not in _STOP and len(t) > 1]

    def _score(s: Sample) -> float:
        gold_toks = _tokens(s.gold)
        pred_toks = _tokens(_extract_answer(s.predicted))
        if not gold_toks:
            return 1.0
        if not pred_toks:
            return 0.0
        gold_counts: dict = {}
        for t in gold_toks:
            gold_counts[t] = gold_counts.get(t, 0) + 1
        pred_counts: dict = {}
        for t in pred_toks:
            pred_counts[t] = pred_counts.get(t, 0) + 1
        overlap = sum(min(gold_counts.get(t, 0), pred_counts.get(t, 0)) for t in pred_counts)
        precision = overlap / len(pred_toks)
        recall = overlap / len(gold_toks)
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)

    return _Metric("token_f1", _score)


def substring_match() -> _Metric:
    """
    1.0 if the normalised gold is a substring of the normalised prediction.
    Useful for Enron verbatim legal-clause answers.
    """

    def _score(s: Sample) -> float:
        g = _norm(s.gold)
        p = _norm(_extract_answer(s.predicted))
        return 1.0 if g and len(g) > 4 and g in p else 0.0

    return _Metric("substring_match", _score)


def message_id_recall() -> _Metric:
    """
    Fraction of ground-truth message_ids that appear anywhere in the prediction.
    Requires s._message_ids list to be attached (done automatically in run_eval).
    The Enron dataset provides 1-20 message_ids per question; we measure recall
    so multi-id questions are handled fairly.

    Score: hits / len(gold_ids)
    If no gold_ids present → 1.0 (not applicable, no penalty).
    """

    def _score(s: Sample) -> float:
        gold_ids: list = getattr(s, "_message_ids", [])
        if not gold_ids:
            return 1.0
        pred = s.predicted.lower()
        hits = sum(1 for mid in gold_ids if mid.lower() in pred)
        return round(hits / len(gold_ids), 4)

    return _Metric("message_id_recall", _score)


# ═════════════════════════════════════════════════════════════════════════════
# ── FINQA METRICS ────────────────────────────────────────────────────────────
#
# Why these metrics for FinQA:
#   The rLLM/rLLM-FinQA-Dataset answers are fully-filled structured templates:
#   markdown sections (PART 1 / PART 2 …), tables with financial figures,
#   bullet-point analysis, and KEY INSIGHTS.  Gold answers are 500-3000 words.
#
#   numerical_recall   — PRIMARY. FinQA is fundamentally about getting numbers
#                        right. Measures the fraction of DISTINCT numbers in the
#                        gold answer that appear in the prediction within 2%
#                        relative tolerance (handles rounding).
#
#   unfilled_slot_penalty — The user_query contains a template with ? / $?
#                        placeholders. A good response fills ALL of them.
#                        Penalises any remaining "?" in the prediction.
#
#   section_header_coverage — The gold answer has PART 1:, PART 2:, **Analysis**
#                        etc. headers. This measures structural completeness.
#
#   table_row_coverage — The gold has markdown tables. Measures what fraction
#                        of data-bearing table rows from gold have at least one
#                        matching number in the prediction.
#
#   length_ratio_score — A 50-word response to a 1500-word template is a
#                        failure even if it accidentally hits a few numbers.
#                        Penalises responses shorter than 40% of gold length.
#
# What we deliberately do NOT use for FinQA:
#   fuzzy_match  — any response > 4 words will contain some gold substring
#                  in a 1000-word gold, giving 0.8 trivially.
#   token_f1     — gold is too long; a short answer can score surprisingly
#                  high just by repeating key terms.
# ═════════════════════════════════════════════════════════════════════════════


def numerical_recall(tol: float = 0.02) -> _Metric:
    """
    Fraction of DISTINCT numbers in the gold answer that appear in the
    prediction within relative tolerance `tol` (default 2%).

    Why 'recall' not 'accuracy':
      FinQA gold answers contain 30-100+ numbers. We measure how many the
      model got right, not whether it hallucinated extra ones (precision).

    Deduplication: numbers are deduplicated in the gold before scoring so
    the same figure repeated in multiple table rows doesn't inflate the
    denominator.

    Tolerance: 0.02 handles rounding differences, e.g. gold=10.8 pred=10.79.
    Zero-valued numbers are skipped (they're trivially present in any response).
    """

    def _parse_distinct(text: str) -> list[float]:
        raw = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", "").replace("%", ""))
        seen: set = set()
        out: list[float] = []
        for r in raw:
            try:
                v = float(r.replace(",", ""))
                if abs(v) < 1e-9:  # skip zeros — trivially present
                    continue
                key = round(v, 4)
                if key not in seen:
                    seen.add(key)
                    out.append(v)
            except ValueError:
                pass
        return out

    def _score(s: Sample) -> float:
        gold_nums = _parse_distinct(s.gold)
        if not gold_nums:
            return 1.0
        pred_text = _extract_answer(s.predicted)
        pred_nums = _parse_distinct(pred_text)
        if not pred_nums:
            return 0.0
        hits = 0
        for g in gold_nums:
            denom = abs(g) if abs(g) > 1e-9 else 1.0
            for p in pred_nums:
                if abs(g - p) / denom <= tol:
                    hits += 1
                    break
        return round(hits / len(gold_nums), 4)

    return _Metric("numerical_recall", _score)


def unfilled_slot_penalty() -> _Metric:
    """
    Penalises predictions that still contain unfilled ? or $? placeholders.

    The FinQA user_query embeds a template like:
        | 2024 | $? | $? | $? |
    A correct response replaces every placeholder with a computed value.
    Any remaining ? in the prediction is a failure to complete the template.

    Score: max(0, 1 - n_unfilled_slots * 0.1)
    So: 0 slots left → 1.0, 10 slots left → 0.0
    """
    _SLOT = re.compile(r"\$?\?(?!\?)")  # $? or standalone ?  (not ??)

    def _score(s: Sample) -> float:
        unfilled = len(_SLOT.findall(s.predicted))
        return max(0.0, round(1.0 - unfilled * 0.1, 2))

    return _Metric("unfilled_slot_penalty", _score)


def section_header_coverage() -> _Metric:
    """
    Fraction of gold section headers present in the prediction.

    The FinQA gold answer has bold markdown headers:
        **PART 1: DOMESTIC VS INTERNATIONAL EFFECTIVE TAX RATES**
        **Analysis:**
        **KEY INSIGHTS:**
    This metric checks that the model's response preserves the structural
    skeleton — i.e., it attempted each section, not just part of the template.

    Matching: first 25 chars of header (lowercased) found anywhere in pred.
    """
    _HDR = re.compile(r"^\s*(?:\*\*|#+)\s*(.+?)(?:\*\*)?\s*$", re.M)

    def _score(s: Sample) -> float:
        gold_headers = [h.strip().lower() for h in _HDR.findall(s.gold)]
        if not gold_headers:
            return 1.0
        pred_lower = s.predicted.lower()
        hits = sum(1 for h in gold_headers if h[:25] in pred_lower)
        return round(hits / len(gold_headers), 4)

    return _Metric("section_header_coverage", _score)


def table_row_coverage() -> _Metric:
    """
    Fraction of data-bearing markdown table rows in the gold answer that have
    at least one numeric match in the prediction.

    FinQA answers are dense with tables like:
        | 2024 | $2,300 | $2,519 | $4,819 | 47.7% | 52.3% |
    A 'data-bearing row' is any line with ≥2 pipe characters AND ≥1 digit.
    For each such row, we check if ANY of its numbers appear in the prediction.

    This tests whether the model filled in table cells, not just prose.
    """

    def _row_nums(line: str) -> list[str]:
        cleaned = line.replace("$", "").replace("%", "").replace("(", "-").replace(")", "")
        return re.findall(r"-?\d[\d,.]*", cleaned)

    def _score(s: Sample) -> float:
        gold_rows = [
            l for l in s.gold.splitlines() if l.count("|") >= 2 and any(c.isdigit() for c in l)
        ]
        if not gold_rows:
            return 1.0
        pred = s.predicted
        hits = 0
        for row in gold_rows:
            nums = _row_nums(row)
            if not nums:
                hits += 1
                continue
            # A row 'passes' if any of its numbers appear verbatim in prediction
            if any(n in pred for n in nums if len(n) > 1):
                hits += 1
        return round(hits / len(gold_rows), 4)

    return _Metric("table_row_coverage", _score)


def length_ratio_score() -> _Metric:
    """
    Penalises responses that are far too short relative to the gold answer.

    FinQA gold answers are 500-3000 words. A 50-word stub is a failure even
    if it accidentally reproduces a few correct numbers.

    Scoring (word count ratio = pred_words / gold_words):
      ratio ≥ 0.40  → 1.0   (full credit, verbose is fine)
      ratio ∈ [0.2, 0.4) → linear ramp 0.0 → 1.0
      ratio < 0.20  → 0.0   (far too short)
    """

    def _score(s: Sample) -> float:
        gold_words = max(len(s.gold.split()), 1)
        pred_words = max(len(s.predicted.split()), 1)
        ratio = pred_words / gold_words
        if ratio >= 0.40:
            return 1.0
        if ratio >= 0.20:
            return round((ratio - 0.20) / 0.20, 4)
        return 0.0

    return _Metric("length_ratio_score", _score)


# ═════════════════════════════════════════════════════════════════════════════
# ── FILE-INGESTION METRICS ───────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════


def tool_chain_order(expected: list[str]) -> _Metric:
    """
    Checks that tools were called in the expected order (subsequence match).
    E.g. tool_chain_order(["list_dir", "read_file", "run_python"])
    Score = fraction of expected steps reached in order.
    """

    def _score(s: Sample) -> float:
        if not expected or not s.tool_calls:
            return 0.0
        idx = 0
        for call in s.tool_calls:
            if idx < len(expected) and expected[idx] in call:
                idx += 1
        return round(idx / len(expected), 4)

    return _Metric("tool_chain_order", _score)


def summary_written() -> _Metric:
    """1.0 if the agent called write_file at least once."""

    def _score(s: Sample) -> float:
        return 1.0 if any("write_file" in c for c in s.tool_calls) else 0.0

    return _Metric("summary_written", _score)


def numerical_accuracy_file() -> _Metric:
    """
    For file-ingestion answers that are typically a single aggregated number.
    Checks if the gold number (parsed) appears in the prediction within 1%.
    Falls back to token_f1 for non-numeric golds.
    """
    _tf1 = token_f1()

    def _parse_first(text: str) -> float | None:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", "").replace("%", ""))
        for n in nums:
            try:
                return float(n.replace(",", ""))
            except ValueError:
                pass
        return None

    def _score(s: Sample) -> float:
        g = _parse_first(s.gold)
        if g is None:
            return _tf1(s)
        p = _parse_first(_extract_answer(s.predicted))
        if p is None:
            return 0.0
        denom = abs(g) if abs(g) > 1e-9 else 1.0
        return 1.0 if abs(g - p) / denom <= 0.01 else 0.0

    return _Metric("numerical_accuracy_file", _score)


def enron_answer_similarity() -> _Metric:
    """
    PRIMARY accuracy metric for Enron short factual answers.
    Normalized exact match first, token_f1 fallback for paraphrased answers.
    """
    _tf1 = token_f1()

    def _n(t: str) -> str:
        return re.sub(r"[$,%\s\.,;:]", "", t.lower().strip())

    def _score(s: Sample) -> float:
        pred = _extract_answer(s.predicted)
        if _n(pred) == _n(s.gold):
            return 1.0
        return _tf1(s)

    return _Metric("enron_answer_similarity", _score)


def finqa_numeric_exact(tol: float = 0.01) -> _Metric:
    """
    PRIMARY accuracy metric for FinQA single-value answers like '0.968'.
    Parses first number from gold and pred, matches within 1% relative tolerance.
    Falls back to token_f1 if gold is not numeric.
    """
    _tf1 = token_f1()

    def _parse_first(text: str) -> float | None:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", "").replace("%", ""))
        for n in nums:
            try:
                return float(n.replace(",", ""))
            except ValueError:
                pass
        return None

    def _score(s: Sample) -> float:
        g = _parse_first(s.gold)
        if g is None:
            return _tf1(s)
        pred = _extract_answer(s.predicted)
        p = _parse_first(pred)
        if p is None:
            return 0.0
        denom = abs(g) if abs(g) > 1e-9 else 1.0
        return 1.0 if abs(g - p) / denom <= tol else 0.0

    return _Metric("finqa_numeric_exact", _score)


# ═════════════════════════════════════════════════════════════════════════════
# PRESETS
# ═════════════════════════════════════════════════════════════════════════════

PRESETS: dict[str, list[_Metric]] = {
    "email_search": [
        enron_answer_similarity(),  # PRIMARY
        token_f1(),
        exact_match_normalized(),
        substring_match(),
        message_id_recall(),
        answer_format(),
        tool_use(),
        specific_tools(["search_inbox", "read_email"], "email_tools"),
        specific_tools(["list_senders"], "sender_discovery"),
        universal_score(),
    ],
    "finqa": [
        finqa_numeric_exact(tol=0.01),  # PRIMARY
        numerical_recall(tol=0.02),
        unfilled_slot_penalty(),
        section_header_coverage(),
        table_row_coverage(),
        length_ratio_score(),
        answer_format(),
        tool_use(),
        specific_tools(["calculator"], "calculator_use"),
        specific_tools(["query_finqa_tables"], "sql_grounding"),
        universal_score(),
    ],
    "file_ingestion": [
        numerical_accuracy_file(),
        token_f1(),
        answer_format(),
        tool_use(),
        tool_chain_order(["list_dir", "read_file", "run_python"]),
        summary_written(),
        specific_tools(["list_dir", "read_file", "run_python"], "file_tools"),
        universal_score(),
    ],
    "generic": [
        universal_score(),
        token_f1(),
        exact_match_normalized(),
        answer_format(),
        tool_use(),
    ],
}


# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

import inspect


def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _build_rollout(
    model_path: str,
    tools: list = None,
    max_steps: int = 20,
    backend: str = "transformers",
    chat_template_kwargs: dict = None,
    system_prompt: str = None,
    **kwargs,
):
    from agenttune.agentic.rollout_engines.rollout_factory import (
        create_rollout_engine,
        create_rollout_fn,
    )

    resolved = backend if backend == "api" else "transformers"
    engine_kwargs = kwargs.pop("engine_kwargs", {})
    if chat_template_kwargs:
        engine_kwargs["chat_template_kwargs"] = chat_template_kwargs
    engine_kwargs = filter_kwargs(create_rollout_engine, engine_kwargs)
    engine = create_rollout_engine(
        backend=resolved,
        model_path=model_path,
        **engine_kwargs,
    )
    rollout_kwargs = filter_kwargs(create_rollout_fn, kwargs)
    if chat_template_kwargs:
        rollout_kwargs["chat_template_kwargs"] = chat_template_kwargs
    return create_rollout_fn(
        rollout_engine=engine,
        tools=tools,
        max_steps=max_steps,
        system_prompt=system_prompt,
        **rollout_kwargs,
    )


def _run_one(
    row: dict,
    rollout_fn,
) -> tuple[str, list[str], int, float, str | None]:
    """
    Run inference on one row.

    FIX: previously returned 6 values but caller unpacked 5.
    Now returns exactly 5 values (batch object dropped from return).

    Tool call extraction now reads from metadata['conversation'] which is
    the most reliable source in the actual batch output format, rather than
    traj.steps[].metadata.tool_calls which was unreliable.

    Returns: (predicted, tool_calls, n_tools, latency_ms, error_or_None)
    """
    prompt = row.get("prompt", [])
    t0 = time.time()
    try:
        batch = rollout_fn([prompt])

        # ── final response ─────────────────────────────────────────────────
        # Use responses[0] — the final assistant turn only, not the full
        # conversation with tool messages mixed in.
        predicted: str = (batch.get("responses") or [""])[0]

        # ── tool call count ────────────────────────────────────────────────
        n_tools: int = (batch.get("tool_call_counts") or [0])[0]

        # ── tool names ────────────────────────────────────────────────────
        # FIX: read from metadata['conversation'] — this is where the actual
        # tool_calls live in the real batch output (see sample batch doc).
        # The old path (traj.steps[].metadata.tool_calls) was not populated.
        tool_calls: list[str] = []
        trajs = batch.get("trajectories", [])
        if trajs:
            traj = trajs[0]
            # primary: read from metadata conversation (most reliable)
            conversation = (getattr(traj, "metadata", {}) or {}).get("conversation", [])
            for msg in conversation:
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls", []):
                        name = (tc.get("function") or {}).get("name", "")
                        if name:
                            tool_calls.append(name)
            # fallback: old path via steps (kept for backward compat)
            if not tool_calls:
                for step in getattr(traj, "steps", []):
                    for tc in (getattr(step, "metadata", {}) or {}).get("tool_calls", []):
                        name = (tc.get("function") or tc).get("name", "")
                        if name:
                            tool_calls.append(name)

        latency_ms = (time.time() - t0) * 1000
        return predicted, tool_calls, n_tools, latency_ms, None

    except Exception:
        latency_ms = (time.time() - t0) * 1000
        return "", [], 0, latency_ms, traceback.format_exc()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════


def run_eval(
    model_path: str,
    dataset,
    use_case: str = "generic",
    tools: list | None = None,
    metrics: list[_Metric] | None = None,
    backend: str = "auto",
    max_steps: int = 10,
    max_samples: int | None = None,
    completions: list[str] | None = None,
    verbose: bool = True,
    chat_template_kwargs: dict | None = None,
    system_prompt: str | None = None,  # ← add
) -> Report:
    rows = dataset.to_list() if hasattr(dataset, "to_list") else list(dataset)
    if max_samples:
        rows = rows[:max_samples]

    active_metrics = metrics or PRESETS.get(use_case, PRESETS["generic"])

    rollout_fn = None
    if completions is None:
        rollout_fn = _build_rollout(
            model_path,
            tools or [],
            max_steps,
            backend,
            chat_template_kwargs=chat_template_kwargs,
            system_prompt=system_prompt,
        )

    resolved_backend = backend if backend == "api" else "transformers"
    logger.info("\n  AgentTune Eval")
    logger.info(f"  Use case : {use_case}")
    logger.info(f"  Model    : {model_path}")
    logger.info(f"  Backend  : {resolved_backend}")
    logger.info(f"  Samples  : {len(rows)}")
    logger.info(f"  Metrics  : {[m.name for m in active_metrics]}\n")

    samples: list[Sample] = []
    t_start = time.time()

    for i, row in enumerate(tqdm(rows, desc=f"Evaluating [{use_case}]", unit="sample")):
        prompt = row.get("prompt", [])
        if isinstance(prompt, list):
            user_msgs = [m for m in prompt if isinstance(m, dict) and m.get("role") == "user"]
            question = user_msgs[-1]["content"] if user_msgs else str(prompt)
        else:
            question = str(prompt)

        # handle answer being a list (Enron dataset quirk)
        raw_answer = row.get("answer", "")
        gold = raw_answer[0] if isinstance(raw_answer, list) else str(raw_answer)

        if completions is not None:
            predicted = completions[i]
            tool_calls = []
            n_tools = 0
            latency = 0.0
            error = None
        else:
            predicted, tool_calls, n_tools, latency, error = _run_one(row, rollout_fn)

        s = Sample(
            idx=i,
            question=question[:300],
            gold=gold,
            predicted=predicted,
            tool_calls=tool_calls,
            n_tools=n_tools,
            latency_ms=latency,
            error=error,
        )

        # attach per-row extras
        raw_ids = row.get("message_ids", [])
        s._message_ids = (
            raw_ids[0]
            if (isinstance(raw_ids, list) and raw_ids and isinstance(raw_ids[0], list))
            else raw_ids
        )
        s._sample_dir = row.get("sample_dir", "")

        for m in active_metrics:
            s.scores[m.name] = m(s)
        samples.append(s)

    duration = time.time() - t_start

    all_names = sorted({n for s in samples for n in s.scores})
    means = {n: statistics.mean(s.scores.get(n, 0.0) for s in samples) for n in all_names}
    stds = {
        n: (statistics.stdev([s.scores.get(n, 0.0) for s in samples]) if len(samples) > 1 else 0.0)
        for n in all_names
    }

    report = Report(
        use_case=use_case,
        model=model_path,
        n_samples=len(samples),
        n_passed=sum(1 for s in samples if s.passed),
        n_errors=sum(1 for s in samples if s.error),
        means=means,
        stds=stds,
        samples=samples,
        duration_s=duration,
    )
    logger.info(f"\n{report}")
    return report
