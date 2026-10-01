"""
FinDER loading + corpus building + GRPO dataset formatting (E1 headline data).

Dataset: Linq-AI-Research/FinDER (arXiv 2504.15800) — 5,703 expert-annotated
query–evidence–answer triplets over SEC 10-K filings from 490 S&P 500
companies. Single HF split ("train"); schema:
  _id, text (query), reasoning (bool), category, references (list[str] — the
  gold evidence excerpts), answer (str, o1-standardised: leads with the direct
  answer, then shows the calculation), type (Subtract/Addition/.../None).

Split protocol (FINNLP_EXPERIMENTS.md v2 §1.1 — "copy Castform"): TICKER-LEVEL
train/val split — validation tickers never appear in training. FinDER rows
carry no ticker column, so ticker identity is recovered by matching each row's
gold references against the raw 10-K filings shipped in the dataset repo
(`10-k.zip`): every reference is a verbatim excerpt of exactly one filing.
Questions whose ticker can't be resolved go to TRAIN (never val), so the val
set is guaranteed uncontaminated.

Retrieval corpus (v1): the deduplicated union of ALL gold references (train +
val rows), matching the HotpotQA corpus pattern already used in this repo —
the val questions' evidence must be in the corpus for val to be answerable.
Gold-chunk ids per question are the corpus chunks of its references, computed
at index-build time (build_index_finder.py) into gold_chunks.json, keyed by
question _id.
"""

import hashlib
import json
import os
import re

from datasets import Dataset

from ..retrieval.corpus_loader import CorpusDocument

# ── System prompt ─────────────────────────────────────────────────────────────
# Finance-flavoured variant of hotpotqa.py's BM25 prompt. FinDER queries are
# short, acronym- and jargon-dense analyst queries ("fy23-fy24 net sales growth
# for azo"), and the corpus is 10-K excerpt text — so the guidance teaches:
# ticker/company + fiscal-year keyword queries, one unknown per search, and a
# concise numeric answer WITH units (the numeric-tolerance reward parses
# $/%/K/M/B/T, so the model should state units explicitly).
_FINDER_BASE_PROMPT = (
    "You are a financial research assistant answering analyst questions over a "
    "corpus of SEC 10-K filing excerpts. You do not know the answer from memory "
    "- you must use the search_corpus tool to find the relevant filing passages, "
    "and read_document if you need a passage's full context. "
    "Search as many times as needed (but no more than necessary) before answering. "
    "When you call a tool, your entire response must be ONLY the tool-call block "
    "- no other words before or after it. "
    "Most questions need more than one search. Break the question into the "
    "unknowns it contains (the company or ticker, the fiscal period, the line "
    "item), then resolve them one search at a time, in order. "
    "After each search, READ the returned passages before deciding whether to "
    "search again or answer - your next search query should be based on what "
    "you just found, not on the original question. "
    "Once you are confident you have the answer, give your final answer wrapped "
    "in <answer></answer> tags, and nothing else outside those tags. Keep the "
    "answer SHORT: the figure with its unit and period (e.g. "
    "<answer>$111.5 million, from FY2021 to FY2023</answer>) or one sentence "
    "for qualitative questions."
)

_FINDER_BM25_GUIDANCE = (
    "Each search query must be SHORT: 2-5 distinctive keywords - the ticker or "
    'company name, the line item, and the fiscal period (e.g. "AZO net sales '
    'FY2023" or "Cboe Data Access Solutions revenue"). Never paste the whole '
    "question - filler words dilute the BM25 match and return nothing. "
    "Financial synonyms help: revenue/net sales, operating income/operating "
    "earnings, fiscal year/FY. "
    "If a search returns no results, do NOT answer from memory - retry with a "
    "SHORTER query (drop the least distinctive keyword), expand an abbreviation "
    '(e.g. "CBOE" -> "Cboe"), or try a synonym for the line item. Only after '
    "2-3 failed searches with different terms should you answer with your best "
    "guess in <answer></answer> tags."
)

FINDER_SYSTEM_PROMPT = _FINDER_BASE_PROMPT + _FINDER_BM25_GUIDANCE


def get_finder_system_prompt(backend: str = "sqlite") -> str:
    """Backend-aware FinDER prompt. Only the BM25 variant exists for E1
    (sqlite FTS5 is the E1 backend per FINNLP_EXPERIMENTS v2 §4/E1); dense
    backends fall back to the same prompt."""
    return FINDER_SYSTEM_PROMPT


# ── Loading ───────────────────────────────────────────────────────────────────


def load_finder_rows() -> list[dict]:
    """Load all 5,703 FinDER rows as plain dicts (single HF 'train' split)."""
    from agenttune.data.loaders.hf_loader import HFLoader

    ds = HFLoader("Linq-AI-Research/FinDER", split="train").load()
    return [dict(row) for row in ds]


# ── Corpus building (dedup gold references) ───────────────────────────────────


def _normalize(text: str) -> str:
    """Whitespace/case-normalised form for dedup + filing containment checks."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def ref_key(reference_text: str) -> str:
    """Stable id for a gold reference passage (normalised-text hash)."""
    return hashlib.sha1(_normalize(reference_text).encode("utf-8")).hexdigest()[:16]


def build_corpus_from_finder(
    rows: list[dict],
    ref_to_ticker: dict[str, str] | None = None,
) -> tuple[list[CorpusDocument], dict[str, str]]:
    """Deduplicate all rows' gold references into CorpusDocuments.

    Returns (docs, ref_to_doc_id): ref_to_doc_id maps ref_key(reference) ->
    doc_id, so each question's gold references can be resolved to corpus doc
    ids (and thence, after chunking, to gold chunk ids).

    If ref_to_ticker is given (ref_key -> ticker, from the filings mapping),
    the ticker is PREPENDED to each doc's title. This is load-bearing, not
    cosmetic: reference excerpts are cut from the middle of filings and often
    never name the company, so an AND-semantics BM25 query containing the
    ticker (the query's most distinctive token) could never match its own gold
    chunk — measured 96% empty-result rate on keyword queries without this.
    A real filing corpus carries company metadata on every document; this
    replicates that.
    """
    docs: list[CorpusDocument] = []
    ref_to_doc_id: dict[str, str] = {}
    for row in rows:
        for ref in row["references"]:
            key = ref_key(ref)
            if key in ref_to_doc_id:
                continue
            doc_id = f"finder_{key}"
            ref_to_doc_id[key] = doc_id
            # Title: first non-empty line, truncated — usually a section or
            # company header; aids BM25 (title is a separate FTS column).
            first_line = next((ln.strip() for ln in ref.splitlines() if ln.strip()), "")
            ticker = (ref_to_ticker or {}).get(key)
            title = f"{ticker} | {first_line}" if ticker else first_line
            docs.append(CorpusDocument(doc_id=doc_id, title=title[:120], text=ref))
    return docs, ref_to_doc_id


# ── Ticker recovery from the raw 10-K filings (10-k.zip) ─────────────────────
# Every gold reference is a verbatim plain-text excerpt of exactly one filing.
# The dataset repo ships the raw filings as `10-k.zip` containing one
# `<TICKER>.html` per company (inline-XBRL HTML). Matching references to
# filings gives each question its ticker without any metadata.
#
# Matcher: window-hash alignment (stdlib only). Both sides are aggressively
# normalised (HTML tags stripped, entities unescaped, EVERYTHING except
# [a-z0-9] collapsed to single spaces — so "1,647" and "$111.5 million" become
# "1647" / "1115 million" identically on both sides, and their HTML->text
# conversion choices can't break the match). Each filing's normalised token
# stream is cut into W-word windows at stride W and hashed into a global
# index (window_hash -> tickers). A reference is probed at every phase offset
# o in [0, W): exactly one o aligns with the filing's window grid, so a single
# probe hit identifies the filing. Verified by a second window before
# accepting.

_WINDOW = 25  # words per window/probe


def _normalize(text: str) -> str:
    """Whitespace/case-normalised form for dedup (keeps punctuation)."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _normalize_for_match(text: str) -> list[str]:
    """Aggressive normalisation for filing containment: strip HTML, unescape
    entities, drop all punctuation, return the token list."""
    import html as _html

    t = _html.unescape(text or "")
    t = re.sub(r"<[^>]+>", " ", t)  # strip HTML tags
    t = re.sub(r"[^a-z0-9]+", " ", t.lower())
    return t.split()


def load_filings(filings_dir: str) -> dict[str, list[str]]:
    """Load extracted 10-K filings as {ticker: normalised_token_list}.

    Layout: one `<TICKER>.html` per filing (FinDER's 10-k.zip). HTML is
    stripped via _normalize_for_match. Multi-file tickers have their token
    lists concatenated.
    """
    filings: dict[str, list[str]] = {}
    for root, _dirs, files in os.walk(filings_dir):
        for fn in sorted(files):
            if fn.startswith(".") or fn.lower().endswith((".zip", ".json", ".csv", ".md")):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            tokens = _normalize_for_match(text)
            if not tokens:
                continue
            ticker = re.split(r"[-_]", os.path.splitext(fn)[0])[0].upper()
            filings.setdefault(ticker, []).extend(tokens)
    return filings


def map_rows_to_tickers(rows: list[dict], filings: dict[str, list[str]]) -> dict[str, str | None]:
    """Map question _id -> ticker via window-hash alignment of gold references
    against filings. Returns None for unresolved rows (they go to train,
    never val, so val stays contamination-free)."""
    # Global window index: window_hash -> set of tickers containing it.
    window_index: dict[int, set] = {}
    for ticker, tokens in filings.items():
        for i in range(0, max(len(tokens) - _WINDOW + 1, 1), _WINDOW):
            window_index.setdefault(hash(" ".join(tokens[i : i + _WINDOW])), set()).add(ticker)

    def _ref_tickers(ref_text: str) -> set:
        tokens = _normalize_for_match(ref_text)
        hits: set = set()
        if len(tokens) < _WINDOW:
            return hits
        # Probe every phase offset; the aligned phase hits the filing's grid.
        for o in range(_WINDOW):
            h = hash(" ".join(tokens[o : o + _WINDOW]))
            hits |= window_index.get(h, set())
        return hits

    row_ticker: dict[str, str | None] = {}
    for row in rows:
        ref_hits = [_ref_tickers(ref) for ref in row["references"]]
        votes: dict[str, int] = {}
        for hits in ref_hits:
            for t in hits:
                votes[t] = votes.get(t, 0) + 1
        # Accept the winner only if EVERY reference votes for it (references
        # all come from one filing; a dissenting vote means a false-positive
        # window collision — safer to leave the row unresolved -> train).
        winner = None
        if votes and ref_hits:
            best = max(votes, key=votes.get)
            if all(best in hits for hits in ref_hits):
                winner = best
        row_ticker[row["_id"]] = winner
    return row_ticker


# ── Ticker-level split ────────────────────────────────────────────────────────


def finder_ticker_split(
    rows: list[dict],
    row_ticker: dict[str, str | None],
    val_frac: float = 0.30,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Split rows into train/val by TICKER (val tickers disjoint from train).

    Rows with unresolved tickers go to train only. Val is sized to ~val_frac
    of rows by accumulating whole tickers (shuffled deterministically).
    """
    import random

    by_ticker: dict[str, list[dict]] = {}
    unresolved: list[dict] = []
    for row in rows:
        t = row_ticker.get(row["_id"])
        if t is None:
            unresolved.append(row)
        else:
            by_ticker.setdefault(t, []).append(row)

    tickers = sorted(by_ticker)
    random.Random(seed).shuffle(tickers)
    target = int(len(rows) * val_frac)
    val_rows: list[dict] = []
    val_tickers = set()
    for t in tickers:
        if len(val_rows) >= target:
            break
        val_tickers.add(t)
        val_rows.extend(by_ticker[t])
    train_rows = [r for t, rs in by_ticker.items() if t not in val_tickers for r in rs]
    train_rows.extend(unresolved)
    return train_rows, val_rows


# ── GRPO dataset formatting ───────────────────────────────────────────────────


def _format_prompt(question: str, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def _doc_of(chunk_id: str) -> str:
    """Strip the `::<idx>` suffix to get the reference/doc id."""
    return chunk_id.rsplit("::", 1)[0] if "::" in chunk_id else chunk_id


def to_grpo_dataset_finder(
    rows: list[dict],
    gold_chunk_map: dict[str, list[str]],
    system_prompt: str = FINDER_SYSTEM_PROMPT,
    domain: str = "finder",
) -> Dataset:
    """Flat HF Dataset for create_agentic_trainer: `prompt` (chat-formatted
    query) + `gold_answer` + `question_id` + `gold_chunk_ids` (list[str] —
    TRL forwards dataset columns to reward fns as kwargs, same path as
    `gold_answer`; consumed by golden_chunk_recall_reward). Deliberately no
    `context` column — the model must retrieve the evidence itself.

    Also emits `domain` (gates the per-domain correctness branch in
    finder_rewards.numeric_correctness_reward) and `optimal_search_count`
    (the per-question frugality target = distinct gold reference docs; one
    search per reference is the honest minimum for reference-level evidence).
    """
    records = [
        {
            "prompt": _format_prompt(row["text"], system_prompt),
            "gold_answer": row["answer"],
            "question_id": row["_id"],
            "gold_chunk_ids": gold_chunk_map.get(row["_id"], []),
            "domain": domain,
            "optimal_search_count": max(
                len({_doc_of(c) for c in gold_chunk_map.get(row["_id"], [])}), 1
            ),
        }
        for row in rows
    ]
    return Dataset.from_list(records)


def load_gold_chunk_map(index_dir: str) -> dict[str, list[str]]:
    """Read gold_chunks.json written by build_index_finder.py ->
    {question_id: [chunk_id, ...]}."""
    path = os.path.join(index_dir, "gold_chunks.json")
    with open(path) as f:
        data = json.load(f)
    return {qid: entry["gold_chunk_ids"] for qid, entry in data.items()}


def load_finder_split_rows(
    index_dir: str,
    rows: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Replay the ticker split recorded in splits.json (written by
    build_index_finder.py) so training uses exactly the same split the index
    was built with. `rows` defaults to a fresh load_finder_rows() call."""
    with open(os.path.join(index_dir, "splits.json")) as f:
        splits = json.load(f)
    if rows is None:
        rows = load_finder_rows()
    by_id = {r["_id"]: r for r in rows}
    train_rows = [by_id[qid] for qid in splits["train_ids"] if qid in by_id]
    val_rows = [by_id[qid] for qid in splits["val_ids"] if qid in by_id]
    return train_rows, val_rows
