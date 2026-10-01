"""
HotpotQA loading + corpus building + GRPO dataset formatting.

Uses agenttune's existing HFLoader (see retrieval/corpus_loader.py's
load_corpus_from_hf for the same pattern) rather than a bespoke HF-hub
loader — HotpotQA is a standard HF dataset, no custom BaseLoader subclass
needed. Confirmed schema (hotpotqa/hotpot_qa, config "distractor"/"fullwiki"):
  id, question, answer, type, level,
  supporting_facts: {title: list[str], sent_id: list[int]},
  context: {title: list[str], sentences: list[list[str]]}
"""

from datasets import Dataset

from agenttune.data.loaders.hf_loader import HFLoader

from ..retrieval.corpus_loader import CorpusDocument

# Shared base: the agentic-RAG task framing + tool-call format constraint
# (model's entire response when calling a tool must be ONLY the tool-call block).
_BASE_PROMPT = (
    "You are a research assistant. You do not know the answer to the user's "
    "question from memory alone - you must use the search_corpus tool to find "
    "relevant passages, and read_document if you need a passage's full context. "
    "Search as many times as needed (but no more than necessary) before answering. "
    "When you call a tool, your entire response must be ONLY the tool-call block "
    "- no other words before or after it. "
    "Most questions need more than one search. Break the question into the "
    "unknowns it contains, then resolve them one search at a time, in order. "
    "After each search, READ the returned passages before deciding whether to "
    "search again or answer - your next search query should be based on what "
    "you just found, not on the original question. "
    "Once you are confident you have the answer, give your final answer wrapped "
    "in <answer></answer> tags, and nothing else outside those tags."
)

# BM25/FTS5 is lexical and ranks on token overlap, so long natural-language
# queries dilute the match with filler tokens and return nothing. The model
# must emit short keyword queries (2-5 distinctive terms). This mirrors
# R1-Searcher's prompt ("only list keywords", "a query must involve only a
# single triple") paired with a sparse retriever.
_BM25_QUERY_GUIDANCE = (
    "Each search query must be SHORT: 2-5 distinctive keywords or proper nouns "
    "(names, places, dates, domain terms). Never paste the whole question or a "
    "full sentence - extra filler words (what, the, year, is) dilute the match "
    "and return nothing. "
    'Example - question: "What year was the university that Alan Turing '
    'attended founded?" Two unknowns: which university, and when founded. '
    'Resolve in order: (1) search "Alan Turing university attended"; '
    "(2) READ the result; (3) if it says King's College Cambridge, search "
    '"King\'s College Cambridge founded"; (4) READ that; (5) answer in '
    "<answer></answer> tags. "
    # Retry-on-empty (BM25). A lexical retriever returns "No results found"
    # when the query has too many tokens diluting the match, or when the exact
    # terms arent indexed. Rather than give up and answer from (wrong) memory,
    # the model retries: drop the least distinctive term and search again with
    # a shorter query, or try a different proper-noun from the question.
    # Deterministic degradation toward the retrievers sweet spot (2-3 strong
    # terms) - more reliable than asking a small model to rephrase fluently.
    "If a search returns no results, do NOT answer from memory - retry with a "
    "SHORTER query (drop the least distinctive keyword and search the remaining "
    "2-3 terms), or try a different proper noun from the question. Only after "
    "2-3 failed searches with different terms should you answer with your best "
    "guess in <answer></answer> tags."
)

# Dense retrieval (Chroma embeddings) matches on semantic similarity, so
# natural-language queries work well and are preferred - the model can phrase
# searches as it would phrase them to a person. This matches R1-Searcher's
# BGE-large dense retriever setup, where the multi-hop chain uses fluent
# queries. Each query is still one hop at a time (single intent), not a
# keyword dump.
_DENSE_QUERY_GUIDANCE = (
    "Phrase each search query as a short natural-language question or phrase "
    "- the retriever matches on meaning, not exact words, so write the query "
    "the way you would describe the missing fact to a person. Keep each query "
    "to ONE unknown at a time (one hop), not a keyword dump. "
    'Example - question: "What year was the university that Alan Turing '
    'attended founded?" Two unknowns: which university, and when founded. '
    'Resolve in order: (1) search "Which university did Alan Turing attend?"; '
    "(2) READ the result; (3) if it says King's College Cambridge, search "
    '"When was King\'s College Cambridge founded?"; (4) READ that; (5) answer '
    "in <answer></answer> tags."
)

DEFAULT_SYSTEM_PROMPT = _BASE_PROMPT + _BM25_QUERY_GUIDANCE
DEFAULT_SYSTEM_PROMPT_DENSE = _BASE_PROMPT + _DENSE_QUERY_GUIDANCE


# ── M1 (MEM1-style rewritten state) prompt ────────────────────────────────────
# Adapted from MEM1's make_prefix() (qa_search_train_merge_multi.py:26). The
# model must emit a <state>...</state> block each turn carrying its running
# memory: {running_summary, open_questions, evidence_notes}. The M1 rewrite
# hook (rag/memory/m1_rewrite.py) extracts this block and wipes the rest of
# the conversation, so the model only ever sees prompt + its latest state +
# the latest compressed tool output. RL trains it to write a state that
# preserves answer-relevant info (reward = outcome EM/F1).
#
# This prompt is layered ON TOP of the backend-specific query guidance, so the
# model still emits short keyword queries (BM25) or natural-language ones
# (dense) — the state block is additional structure the model produces before
# each tool call.
_M1_STATE_INSTRUCTION = (
    "MEMORY PROTOCOL (critical): Before EVERY action, you MUST emit a "
    "<state>...</state> block summarizing what you know so far, THEN immediately "
    "take an action in the SAME response. Your response is ALWAYS two parts: "
    "(1) a <state> block, then (2) EITHER a tool-call block OR <answer> tags. "
    "NEVER emit a <state> block alone and stop — a state block with no following "
    "action is an incomplete response.\n"
    "This <state> block is your ONLY memory — after each search, your prior "
    "history is wiped and only your <state> block plus the latest search result "
    "survive. Write the state as three labeled lines:\n"
    "  running_summary: <one-sentence recap of facts established so far>\n"
    "  open_questions: <the remaining unknowns still to resolve>\n"
    "  evidence_notes: <key facts with their chunk_id, e.g. 'King's College "
    "Cambridge founded 1441 [chunk_id=...]'>\n"
    "Keep the state SHORT (under 80 words). If this is your first action, "
    "running_summary and evidence_notes are empty and open_questions restates "
    "the question's unknowns.\n"
    "DECISION RULE (after writing <state>): if open_questions is non-empty, "
    "emit a search_corpus tool call for the FIRST open question. If "
    "open_questions is empty (all unknowns resolved from evidence), emit your "
    "final answer in <answer></answer> tags. Do NOT search again once you have "
    "enough evidence — answer immediately."
)

DEFAULT_SYSTEM_PROMPT_M1 = _BASE_PROMPT + _BM25_QUERY_GUIDANCE + "\n\n" + _M1_STATE_INSTRUCTION
DEFAULT_SYSTEM_PROMPT_M1_DENSE = (
    _BASE_PROMPT + _DENSE_QUERY_GUIDANCE + "\n\n" + _M1_STATE_INSTRUCTION
)


# ── M2 (trained memory decisions) prompt ──────────────────────────────────────
# M2 extends M1: instead of always compressing, the model chooses a memory
# operation (keep/drop/compress) and an action (search-again/answer-now) each
# turn, and states that choice as an explicit token the memory/m2_decisions.py
# hook parses (parse_decision_token) and the decision_reward scores. Layered
# on top of the M1 state instruction — the model still writes the <state>
# block (memory_op_post_step_hook falls back to M1's compress when no
# decision token is found, so this prompt must keep the model emitting state
# even before it learns to emit decisions).
_M2_DECISION_INSTRUCTION = (
    "MEMORY DECISIONS (extends the memory protocol above): after your <state> "
    "block, also choose and emit ONE decision token before your action: "
    "<decision:MEMORY_OP:ACTION_OP>. MEMORY_OP is one of:\n"
    "  keep     - the current context is short enough, no rewrite needed\n"
    "  drop     - wipe everything except the original question (evidence so "
    "far wasn't useful)\n"
    "  compress - rewrite to your <state> block (the default; use this most "
    "of the time)\n"
    "ACTION_OP is one of:\n"
    "  search-again - you still have open questions, emit a tool call next\n"
    "  answer-now   - you have enough evidence, emit <answer></answer> next\n"
    "Your full response is now three parts, in order: (1) <state> block, "
    "(2) exactly one <decision:...:...> token, (3) EITHER a tool-call block "
    "OR your final answer, matching ACTION_OP. Example: "
    "<decision:compress:answer-now> must be immediately followed by an "
    "<answer> tag, not a tool call."
)

DEFAULT_SYSTEM_PROMPT_M2 = DEFAULT_SYSTEM_PROMPT_M1 + "\n\n" + _M2_DECISION_INSTRUCTION
DEFAULT_SYSTEM_PROMPT_M2_DENSE = DEFAULT_SYSTEM_PROMPT_M1_DENSE + "\n\n" + _M2_DECISION_INSTRUCTION


def get_system_prompt(backend: str, m1: bool = False, m2: bool = False) -> str:
    """Return the system prompt tuned for the retrieval backend.

    BM25/lexical backends (sqlite FTS5) need short keyword queries - long
    natural-language queries dilute the token-overlap match. Dense backends
    (chroma embeddings) match on semantic similarity, so natural-language
    queries work better. Both teach multi-hop decomposition (search -> read
    -> search again with a query informed by the prior result), which HotpotQA
    questions require. See README.md for the research basis
    (R1-Searcher, Search-R1).

    If ``m1=True``, the MEM1-style memory-protocol instruction is appended
    (see DEFAULT_SYSTEM_PROMPT_M1) so the model emits the <state> block the
    M1 rewrite hook extracts. If ``m2=True`` (implies m1=True), the decision
    token instruction is appended on top so the model also emits
    <decision:MEMORY_OP:ACTION_OP>, which memory_op_post_step_hook parses.
    """
    b = (backend or "").lower()
    dense = b in ("chroma", "dense", "vector")
    if m2:
        return DEFAULT_SYSTEM_PROMPT_M2_DENSE if dense else DEFAULT_SYSTEM_PROMPT_M2
    if m1:
        return DEFAULT_SYSTEM_PROMPT_M1_DENSE if dense else DEFAULT_SYSTEM_PROMPT_M1
    return DEFAULT_SYSTEM_PROMPT_DENSE if dense else DEFAULT_SYSTEM_PROMPT


def load_hotpotqa_splits(
    config: str = "distractor",
    train_size: int = 2000,
    eval_size: int = 200,
    seed: int = 0,
) -> tuple[Dataset, Dataset]:
    """Deterministic HotpotQA train/eval slices via the existing HFLoader."""
    train_full = HFLoader("hotpotqa/hotpot_qa", config_name=config, split="train").load()
    eval_full = HFLoader("hotpotqa/hotpot_qa", config_name=config, split="validation").load()

    train_full = train_full.shuffle(seed=seed)
    eval_full = eval_full.shuffle(seed=seed)

    train_split = train_full.select(range(min(train_size, len(train_full))))
    eval_split = eval_full.select(range(min(eval_size, len(eval_full))))
    return train_split, eval_split


def build_corpus_from_hotpotqa(
    dataset: Dataset, max_docs: int | None = None
) -> list[CorpusDocument]:
    """Flattens each example's gold+distractor context into deduplicated
    CorpusDocuments — this is the fixed corpus search_corpus searches over."""
    seen: dict = {}
    for row in dataset:
        titles = row["context"]["title"]
        sentence_lists = row["context"]["sentences"]
        for title, sentences in zip(titles, sentence_lists, strict=False):
            if title in seen:
                continue
            seen[title] = CorpusDocument(doc_id=title, title=title, text="".join(sentences))
            if max_docs is not None and len(seen) >= max_docs:
                return list(seen.values())
    return list(seen.values())


def _format_prompt(question: str, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def to_grpo_dataset(hf_split: Dataset, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> Dataset:
    """
    Produces the flat HF Dataset create_agentic_trainer needs: a `prompt`
    column (chat-formatted question) + `gold_answer` + `question_id`.
    Deliberately no `context` column — the model must retrieve it itself.
    """
    records = [
        {
            "prompt": _format_prompt(row["question"], system_prompt),
            "gold_answer": row["answer"],
            "question_id": row["id"],
        }
        for row in hf_split
    ]
    return Dataset.from_list(records)
