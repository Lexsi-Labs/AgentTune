import pytest

from agenttune.rag.data.hotpotqa import (
    DEFAULT_SYSTEM_PROMPT,
    build_corpus_from_hotpotqa,
    load_hotpotqa_splits,
    to_grpo_dataset,
)

# load_hotpotqa_splits pulls the full HotpotQA dataset from the Hub (multi-GB,
# ~80s locally) — a download suite, not a fast-gate test.
pytestmark = pytest.mark.slow


def test_load_hotpotqa_splits_smoke():
    train, eval_ = load_hotpotqa_splits(train_size=5, eval_size=3, seed=0)
    assert len(train) == 5
    assert len(eval_) == 3
    for row in train:
        assert {"id", "question", "answer", "context"} <= set(row.keys())


def test_load_hotpotqa_splits_deterministic():
    a_train, a_eval = load_hotpotqa_splits(train_size=3, eval_size=2, seed=42)
    b_train, b_eval = load_hotpotqa_splits(train_size=3, eval_size=2, seed=42)
    assert [r["id"] for r in a_train] == [r["id"] for r in b_train]
    assert [r["id"] for r in a_eval] == [r["id"] for r in b_eval]


def test_build_corpus_from_hotpotqa_deduplicates_titles():
    train, _ = load_hotpotqa_splits(train_size=10, eval_size=1, seed=0)
    docs = build_corpus_from_hotpotqa(train)
    titles = [d.title for d in docs]
    assert len(titles) == len(set(titles))
    assert all(d.text.strip() for d in docs)


def test_build_corpus_from_hotpotqa_respects_max_docs():
    train, _ = load_hotpotqa_splits(train_size=10, eval_size=1, seed=0)
    docs = build_corpus_from_hotpotqa(train, max_docs=3)
    assert len(docs) <= 3


def test_to_grpo_dataset_schema():
    _, eval_ = load_hotpotqa_splits(train_size=1, eval_size=4, seed=0)
    ds = to_grpo_dataset(eval_)
    assert len(ds) == 4
    row = ds[0]
    assert set(row.keys()) == {"prompt", "gold_answer", "question_id"}
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][0]["content"] == DEFAULT_SYSTEM_PROMPT
    assert row["prompt"][1]["role"] == "user"
    assert row["gold_answer"]
