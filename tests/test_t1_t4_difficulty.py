"""
Unit tests for T1 (solve-difficulty prober) + T4 (requires-search filter).

Pure-Python tests that don't need a GPU — they mock the generation and test
the scoring/filtering logic. The actual model probing runs on GPU.
"""

from unittest.mock import patch

from agenttune.rag.synthesis.requires_search_filter import (
    filter_to_requires_search,
    label_requires_search,
)
from agenttune.rag.synthesis.solve_difficulty import filter_contaminated


class TestFilterContaminated:
    """T1 contamination gate: drop questions the model already solves."""

    def test_drops_all_correct(self):
        """Questions with pass_rate=1.0 (all k samples correct) are dropped."""
        questions = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "Capital of France?", "answer": "Paris"},
            {"question": "Who wrote Hamlet?", "answer": "Shakespeare"},
        ]
        probe_results = [
            {
                "question": "What is 2+2?",
                "gold": "4",
                "pass_rate": 1.0,
                "solve_difficulty": 0.0,
                "already_solved": True,
                "samples": [],
            },
            {
                "question": "Capital of France?",
                "gold": "Paris",
                "pass_rate": 1.0,
                "solve_difficulty": 0.0,
                "already_solved": True,
                "samples": [],
            },
            {
                "question": "Who wrote Hamlet?",
                "gold": "Shakespeare",
                "pass_rate": 0.0,
                "solve_difficulty": 1.0,
                "already_solved": False,
                "samples": [],
            },
        ]
        kept_q, kept_r = filter_contaminated(probe_results, questions, threshold=1.0)
        assert len(kept_q) == 1
        assert kept_q[0]["question"] == "Who wrote Hamlet?"

    def test_keeps_partial(self):
        """Questions with 0 < pass_rate < 1 are kept (model sometimes fails)."""
        questions = [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]
        probe_results = [
            {
                "question": "Q1",
                "gold": "A1",
                "pass_rate": 0.5,
                "solve_difficulty": 0.5,
                "already_solved": False,
                "samples": [],
            },
            {
                "question": "Q2",
                "gold": "A2",
                "pass_rate": 0.0,
                "solve_difficulty": 1.0,
                "already_solved": False,
                "samples": [],
            },
        ]
        kept_q, kept_r = filter_contaminated(probe_results, questions, threshold=1.0)
        assert len(kept_q) == 2

    def test_custom_threshold(self):
        """Threshold 0.5 drops anything with pass_rate >= 0.5."""
        questions = [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]
        probe_results = [
            {
                "question": "Q1",
                "gold": "A1",
                "pass_rate": 0.5,
                "solve_difficulty": 0.5,
                "already_solved": False,
                "samples": [],
            },
            {
                "question": "Q2",
                "gold": "A2",
                "pass_rate": 0.25,
                "solve_difficulty": 0.75,
                "already_solved": False,
                "samples": [],
            },
        ]
        kept_q, _ = filter_contaminated(probe_results, questions, threshold=0.5)
        assert len(kept_q) == 1
        assert kept_q[0]["question"] == "Q2"

    def test_empty(self):
        kept_q, kept_r = filter_contaminated([], [], threshold=1.0)
        assert kept_q == [] and kept_r == []


class TestLabelRequiresSearch:
    """T4 requires-search filter: label questions that need retrieval."""

    def test_labels_correctly(self):
        """Questions with high pass_rate → requires_search=False."""
        questions = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        mock_probe = [
            {
                "question": "Q1",
                "gold": "A1",
                "pass_rate": 0.8,
                "solve_difficulty": 0.2,
                "already_solved": False,
                "samples": [],
            },
            {
                "question": "Q2",
                "gold": "A2",
                "pass_rate": 0.2,
                "solve_difficulty": 0.8,
                "already_solved": False,
                "samples": [],
            },
        ]
        with patch(
            "agenttune.rag.synthesis.solve_difficulty.probe_solve_difficulty",
            return_value=mock_probe,
        ):
            labeled = label_requires_search(questions, "fake_model", k=4, use_vllm=False)

        assert labeled[0]["requires_search"] is False  # pass_rate 0.8 >= 0.5
        assert labeled[1]["requires_search"] is True  # pass_rate 0.2 < 0.5

    def test_custom_threshold(self):
        """With threshold=0.9, only pass_rate>=0.9 is requires_search=False."""
        questions = [{"question": "Q1", "answer": "A1"}]
        mock_probe = [
            {
                "question": "Q1",
                "gold": "A1",
                "pass_rate": 0.75,
                "solve_difficulty": 0.25,
                "already_solved": False,
                "samples": [],
            },
        ]
        with patch(
            "agenttune.rag.synthesis.solve_difficulty.probe_solve_difficulty",
            return_value=mock_probe,
        ):
            labeled = label_requires_search(
                questions, "fake_model", k=4, pass_threshold=0.9, use_vllm=False
            )
        assert labeled[0]["requires_search"] is True  # 0.75 < 0.9

    def test_preserves_question_and_gold(self):
        questions = [{"question": "What year?", "answer": "1441"}]
        mock_probe = [
            {
                "question": "What year?",
                "gold": "1441",
                "pass_rate": 0.0,
                "solve_difficulty": 1.0,
                "already_solved": False,
                "samples": [],
            },
        ]
        with patch(
            "agenttune.rag.synthesis.solve_difficulty.probe_solve_difficulty",
            return_value=mock_probe,
        ):
            labeled = label_requires_search(questions, "fake_model", k=4, use_vllm=False)
        assert labeled[0]["question"] == "What year?"
        assert labeled[0]["gold"] == "1441"


class TestFilterToRequiresSearch:
    """T4 filter: keep only requires_search=True questions."""

    def test_filters_correctly(self):
        questions = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
            {"question": "Q3", "answer": "A3"},
        ]
        labeled = [
            {"question": "Q1", "requires_search": True},
            {"question": "Q2", "requires_search": False},
            {"question": "Q3", "requires_search": True},
        ]
        kept = filter_to_requires_search(labeled, questions)
        assert len(kept) == 2
        assert kept[0]["question"] == "Q1"
        assert kept[1]["question"] == "Q3"

    def test_all_need_search(self):
        questions = [{"question": "Q1", "answer": "A1"}]
        labeled = [{"question": "Q1", "requires_search": True}]
        kept = filter_to_requires_search(labeled, questions)
        assert len(kept) == 1

    def test_none_need_search(self):
        questions = [{"question": "Q1", "answer": "A1"}]
        labeled = [{"question": "Q1", "requires_search": False}]
        kept = filter_to_requires_search(labeled, questions)
        assert len(kept) == 0

    def test_empty(self):
        assert filter_to_requires_search([], []) == []
