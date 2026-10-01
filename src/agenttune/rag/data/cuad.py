"""
CUAD loading + corpus building for the legal within-document multi-hop track
(NLLP_SynthData.md Part A). Mirrors `finder.py`'s shape: load raw records,
build CorpusDocuments, provide a train/test split, hand both to the
synthesis pipeline and to a native-task eval harness.

Source: CUAD v1 — 510 real contracts from SEC EDGAR. Two interchangeable
loading paths:

  1. `load_cuad_contracts(txt_dir)` — the `full_contract_txt/*.txt` files
     (plain text, what the original HotpotQA-style corpus path used).
  2. `load_cuad_squad_json(path)` — the official CUAD-QA SQuAD-format JSON
     (`CUADv1.json` / `train_separate_questions.json` / `test.json` from the
     Atticus-Project/cuad GitHub `data.zip`). Each doc is one paragraph whose
     `context` IS the full contract text, and each question id encodes the
     clause category (`<title>__<Category>[_<n>]`). Verified (2026-08-13)
     against the official release: JSON context == .txt content
     (whitespace-normalized) for all sampled contracts, 13,823/13,823 answer
     spans resolve exactly (context[start:start+len(text)] == text), and the
     official train/test split is perfectly contract-disjoint.

Split: CUAD-QA ships the OFFICIAL train/test split (22,450 / 4,182 rows,
partitioned by contract). Verified contract-disjoint on 2026-08-13: 408 train
/ 102 test contract titles, ZERO overlap, union = all 510 contracts. Use
`official_split_contracts(train_json, test_json)` when the JSONs are
available (it hard-asserts disjointness); `split_contracts` remains as a
deterministic hash-based fallback for when only the txt corpus is available.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

from datasets import Dataset

from ..retrieval.corpus_loader import CorpusDocument

# Official CUAD-QA release asset (Atticus-Project/cuad, default branch `main`).
# Contains CUADv1.json (all 510 contracts + questions), train_separate_questions.json,
# and test.json (the official contract-partitioned train/test split).
OFFICIAL_ZIP_URL = "https://raw.githubusercontent.com/The-Atticus-Project/cuad/main/data.zip"

# Expected official split sizes — verified against the real release, asserted
# by `official_split_contracts` so a changed/updated dataset fails loudly.
OFFICIAL_TRAIN_CONTRACTS = 408
OFFICIAL_TEST_CONTRACTS = 102


def load_cuad_contracts(txt_dir: str) -> list[dict]:
    """Load every `.txt` file under `txt_dir` (recursively — CUAD ships
    `Part_I/`, `Part_II/`, ... subfolders) as a contract record.

    Returns [{"contract_id": <filename w/o ext>, "text": <full text>,
    "source_path": <path>}, ...], sorted by contract_id for determinism.
    """
    records = []
    for root, _dirs, files in os.walk(txt_dir):
        for fn in sorted(files):
            if not fn.lower().endswith(".txt"):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if not text.strip():
                continue
            records.append(
                {
                    "contract_id": os.path.splitext(fn)[0],
                    "text": text,
                    "source_path": path,
                }
            )
    records.sort(key=lambda r: r["contract_id"])
    return records


def build_corpus_from_cuad(contracts: list[dict]) -> list[CorpusDocument]:
    """One CorpusDocument per contract. `doc_id` = contract_id, so the
    same-document guard in graph.build_graph (same_doc_only=True) has a
    stable, human-readable key to restrict edges to."""
    return [
        CorpusDocument(doc_id=c["contract_id"], title=c["contract_id"], text=c["text"])
        for c in contracts
    ]


def split_contracts(
    contracts: list[dict], val_fraction: float = 0.30, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """Deterministic contract-level train/test split by hashing contract_id.

    FALLBACK, not the official CUAD-QA split — see module docstring. Uses a
    stable hash (not `random`, which would depend on call order) so the same
    split is reproduced across runs/machines given the same contract set.
    """

    def _bucket(contract_id: str) -> float:
        h = hashlib.sha1(f"{seed}:{contract_id}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    test = [c for c in contracts if _bucket(c["contract_id"]) < val_fraction]
    train = [c for c in contracts if _bucket(c["contract_id"]) >= val_fraction]
    return train, test


def get_cuad_system_prompt() -> str:
    """System prompt for the CUAD retrieval agent — legal-register variant
    of hotpotqa.py's BM25 prompt, adapted for within-document contract
    review questions."""
    return (
        "You are a contract review assistant answering questions about legal "
        "agreements. You do not know the answer from memory — you must use the "
        "search_corpus tool to find the relevant contract passages, and "
        "read_document if you need a passage's full context. Most questions "
        "require combining a definition or obligation from one part of a "
        "contract with a clause elsewhere in the SAME contract — search for "
        "each piece separately, in order, before answering. "
        "When you call a tool, your entire response must be ONLY the tool-call "
        "block — no other words before or after it. "
        "Once you are confident you have the answer, give your final answer "
        "wrapped in <answer></answer> tags, and nothing else outside those "
        "tags. Keep the answer SHORT — the specific fact, term, or figure "
        "asked for, not a restatement of the question."
    )


# ── Official CUAD-QA JSON path ─────────────────────────────────────────────


def category_from_qa_id(qa_id: str) -> str:
    """Extract the clause category from a CUAD-QA question id.

    Id format (verified against the official release): `<title>__<Category>`
    in CUADv1.json, `<title>__<Category>_<n>` in train_separate_questions.json
    (multi-answer questions split into one row per answer). The double
    underscore never appears in titles (they use single `_`); the trailing
    `_<digits>` of the separated form is stripped.
    """
    parts = qa_id.split("__", 1)
    if len(parts) != 2:
        return ""
    return re.sub(r"_\d+$", "", parts[1])


def load_cuad_squad_json(path: str) -> list[dict]:
    """Load a CUAD-QA SQuAD-format JSON into contract records.

    Works for `CUADv1.json`, `train_separate_questions.json`, or `test.json`.
    Verified against the official release (2026-08-13): each document has
    exactly one paragraph whose `context` is the full contract text, and every
    answer span resolves exactly against it.

    Returns records of the same shape as `load_cuad_contracts` plus a `qas`
    list: {"contract_id", "text", "qas": [{"category", "question", "answers":
    [{"text", "answer_start"}]}]}. `build_corpus_from_cuad` accepts both
    record shapes.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for doc in data.get("data", []):
        title = doc.get("title", "")
        paragraphs = doc.get("paragraphs", [])
        if len(paragraphs) != 1:
            # The official release is 1 paragraph per doc (verified); anything
            # else would change span coordinates — fail loudly, don't guess.
            raise ValueError(
                f"CUAD JSON doc {title!r} has {len(paragraphs)} paragraphs; "
                f"expected exactly 1 (answer_start is doc-relative)"
            )
        ctx = paragraphs[0].get("context", "")
        qas = []
        for qa in paragraphs[0].get("qas", []):
            answers = [
                {"text": a["text"], "answer_start": a["answer_start"]}
                for a in qa.get("answers", [])
            ]
            qas.append(
                {
                    "category": category_from_qa_id(qa.get("id", "")),
                    "question": qa.get("question", ""),
                    "answers": answers,
                }
            )
        records.append({"contract_id": title, "text": ctx, "qas": qas})
    records.sort(key=lambda r: r["contract_id"])
    return records


def official_split_contracts(train_json: str, test_json: str) -> tuple[list[str], list[str]]:
    """Contract ids from the OFFICIAL CUAD-QA train/test split.

    Verified (2026-08-13): 408 train / 102 test contracts, ZERO title overlap,
    union = all 510 contracts. This replaces `split_contracts` whenever the
    official JSONs are available — it hard-asserts disjointness and the
    expected counts, so an updated or malformed release fails loudly instead
    of silently training on leaked test contracts.

    Returns (train_contract_ids, test_contract_ids), both sorted.
    """
    train_ids = sorted({r["contract_id"] for r in load_cuad_squad_json(train_json)})
    test_ids = sorted({r["contract_id"] for r in load_cuad_squad_json(test_json)})
    overlap = set(train_ids) & set(test_ids)
    assert not overlap, (
        f"official CUAD-QA split is NOT contract-disjoint: "
        f"{len(overlap)} overlapping title(s): {sorted(overlap)[:5]}"
    )
    if len(train_ids) != OFFICIAL_TRAIN_CONTRACTS or len(test_ids) != OFFICIAL_TEST_CONTRACTS:
        raise ValueError(
            f"official split sizes changed: train={len(train_ids)} (expected "
            f"{OFFICIAL_TRAIN_CONTRACTS}), test={len(test_ids)} (expected "
            f"{OFFICIAL_TEST_CONTRACTS}) — the release was updated; re-verify "
            f"before trusting this split."
        )
    return train_ids, test_ids


# ── Category tagging (NLLP_SynthData.md §A2, metadata only) ────────────────


def labeled_spans_from_records(records: list[dict]) -> list[dict]:
    """Flatten CUAD-QA answer spans into {contract_id, category, start, end, text}.

    `start`/`end` are character offsets into the contract text (the same
    coordinate system `chunk_text_with_offsets` produces), so they align with
    chunk metadata["start"]/["end"] exactly. Multiple answers of the same
    category in one contract stay separate entries (a category can annotate
    several disjoint clauses).
    """
    spans = []
    for r in records:
        for qa in r.get("qas", []):
            for a in qa.get("answers", []):
                s, t = a["answer_start"], a["text"]
                spans.append(
                    {
                        "contract_id": r["contract_id"],
                        "category": qa["category"],
                        "start": s,
                        "end": s + len(t),
                        "text": t,
                    }
                )
    return spans


def tag_chunks_with_categories(chunks, spans) -> list:
    """Tag chunks with the CUAD categories whose labeled spans overlap them.

    Per NLLP_SynthData.md §A2: the 41 category labels are repurposed as
    metadata, not a new edge type. A chunk overlapping ANY labeled span of a
    category gets that category (a chunk can carry several). Chunks must have
    metadata["start"]/["end"] (chunk_text_with_offsets).

    Tags are stored at chunk.metadata["cuad_categories"] as a sorted list of
    unique category names. Deterministic, no LLM. Mutates and returns `chunks`.
    """
    by_doc: dict[str, list[dict]] = {}
    for sp in spans:
        by_doc.setdefault(sp["contract_id"], []).append(sp)
    for c in chunks:
        start, end = c.metadata.get("start"), c.metadata.get("end")
        if start is None or end is None:
            c.metadata["cuad_categories"] = []
            continue
        cats = set()
        for sp in by_doc.get(c.doc_id, []):
            # any nonzero overlap between [start,end) and the labeled span
            if sp["start"] < end and sp["end"] > start:
                cats.add(sp["category"])
        c.metadata["cuad_categories"] = sorted(cats)
    return chunks


def gold_chunk_categories(chunks_by_id, gold_chunk_ids) -> list[str]:
    """The union of category tags across a question's gold chunks — the
    human-legible description of what a question bridges (e.g. a question
    that "bridges the Confidentiality clause and the Survival clause")."""
    cats = set()
    for cid in gold_chunk_ids:
        c = chunks_by_id.get(cid)
        if c is None:
            continue
        cats.update(c.metadata.get("cuad_categories", []))
    return sorted(cats)


# ── Synthetic CUAD GRPO dataset (the mixed-training half) ──────────────────
# The synthesis pipeline emits `dataset_grpo.jsonl` (and the JSON-array form,
# e.g. smoke_train_10.json). Verified 2026-08-14: `prompt` is already the
# chat-formatted [system, user] pair with the legal system prompt baked in;
# `gold_path` is a JSON-ENCODED STRING (json.loads it) listing the ordered
# reasoning-path chunk ids (`<contract_id>::<chunk_idx>`); `gold_passages`
# carries each chunk's text for alignment verification.


def load_cuad_grpo_rows(path: str) -> list[dict]:
    """Load CUAD synthetic GRPO rows from a JSON array (.json) or JSONL (.jsonl).

    Tolerates a {"data": [...]} / {"rows": [...]} wrapper dict. `gold_path`
    stays as stored (a JSON string) — `to_grpo_dataset_cuad` parses it."""
    if str(path).endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", data.get("rows", []))
    return list(data)


def to_grpo_dataset_cuad(rows: list[dict], domain: str = "cuad") -> Dataset:
    """Flat HF Dataset for create_agentic_trainer from CUAD synthetic rows.

    Each row's `prompt` is used as-is (the synthesis already baked in the
    legal-register CUAD system prompt). `gold_chunk_ids` = parsed `gold_path`.
    `domain` gates the per-domain correctness branch in
    finder_rewards.numeric_correctness_reward (relaxed legal F1 for CUAD).
    `optimal_search_count` = hop_count, the per-question frugality target.
    Columns are intentionally identical to to_grpo_dataset_finder's so the
    mixed interleave builds a homogeneous dataset (no per-domain columns)."""
    records = []
    for r in rows:
        if r.get("answerable") is False:
            continue
        gold_path = r.get("gold_path", "[]")
        if isinstance(gold_path, str):
            gold_path = json.loads(gold_path)
        gold_path = list(gold_path or [])
        records.append(
            {
                "prompt": r["prompt"],
                "gold_answer": r["gold_answer"],
                "question_id": r["question_id"],
                "gold_chunk_ids": gold_path,
                "domain": domain,
                "optimal_search_count": max(int(r.get("hop_count", 1) or 1), 1),
            }
        )
    return Dataset.from_list(records)
