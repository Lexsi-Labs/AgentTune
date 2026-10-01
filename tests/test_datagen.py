"""Tests for docs->QA data generation + difficulty curriculum (P3)."""

from agenttune.rag.datagen import (
    Chunk,
    QAPair,
    balance_by_difficulty,
    generate_qa_from_corpus,
    label_difficulty,
    sort_by_difficulty,
)


def _chunks():
    return [
        Chunk(id="c1", text="The Eiffel Tower is in Paris."),
        Chunk(id="c2", text="Mount Everest is the tallest mountain."),
    ]


def test_generate_qa_grounds_to_source_chunk():
    def fake_generator(text):
        # one grounded (question, answer) per chunk
        if "Eiffel" in text:
            return [("Where is the Eiffel Tower?", "Paris")]
        return [("What is the tallest mountain?", "Mount Everest")]

    qa = generate_qa_from_corpus(_chunks(), fake_generator)
    assert len(qa) == 2
    assert all(isinstance(p, QAPair) for p in qa)
    eiffel = next(p for p in qa if "Eiffel" in p.question)
    assert eiffel.answer == "Paris"
    assert eiffel.gold_chunk_ids == ["c1"]  # grounded to the source chunk


def test_generate_qa_multiple_per_chunk():
    def gen(text):
        return [("q1", "a1"), ("q2", "a2")]

    qa = generate_qa_from_corpus([Chunk(id="c1", text="x")], gen)
    assert len(qa) == 2
    assert {p.gold_chunk_ids[0] for p in qa} == {"c1"}


def test_label_difficulty_easy_vs_hard():
    qa = [
        QAPair(question="easy?", answer="yes", gold_chunk_ids=["c1"]),
        QAPair(question="hard?", answer="complicated", gold_chunk_ids=["c2"]),
    ]

    # solver knows the easy one, flubs the hard one
    def solver(question):
        return "yes" if question == "easy?" else "no idea"

    labeled = label_difficulty(qa, solver, mode="em")
    by_q = {p.question: p.difficulty for p in labeled}
    assert by_q["easy?"] == "easy"  # model already answers -> easy (drop/deprioritize)
    assert by_q["hard?"] == "hard"  # knowledge gap -> hard (worth training on)


def test_label_difficulty_uses_pass_threshold_over_probes():
    qa = [QAPair(question="q", answer="right", gold_chunk_ids=["c1"])]
    calls = {"n": 0}

    def flaky_solver(question):
        calls["n"] += 1
        # correct on 1 of 4 probes -> 25% < 50% threshold -> hard
        return "right" if calls["n"] == 1 else "wrong"

    labeled = label_difficulty(qa, flaky_solver, mode="em", n_probes=4, pass_threshold=0.5)
    assert labeled[0].difficulty == "hard"
    assert calls["n"] == 4


def test_balance_by_difficulty_one_to_one():
    qa = [QAPair(f"e{i}", "a", ["c"], difficulty="easy") for i in range(5)] + [
        QAPair(f"h{i}", "a", ["c"], difficulty="hard") for i in range(2)
    ]
    balanced = balance_by_difficulty(qa, ratio=(1, 1))
    n_easy = sum(p.difficulty == "easy" for p in balanced)
    n_hard = sum(p.difficulty == "hard" for p in balanced)
    assert n_easy == n_hard == 2  # limited by the scarcer class


def test_sort_by_difficulty_easy_first():
    qa = [
        QAPair("h", "a", ["c"], difficulty="hard"),
        QAPair("e", "a", ["c"], difficulty="easy"),
    ]
    ordered = sort_by_difficulty(qa)
    assert [p.difficulty for p in ordered] == ["easy", "hard"]
