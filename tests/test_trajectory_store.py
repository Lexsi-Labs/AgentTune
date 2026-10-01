"""
Unit tests for the trajectory store and multi-format document loaders.

Pure-Python tests — no GPU, no model, no network. Uses temp files for SQLite
and sample documents.
"""

import json
import os
import tempfile

import pytest

from agenttune.rag.retrieval.corpus_loader import CorpusDocument
from agenttune.rag.retrieval.document_loaders import (
    get_supported_formats,
    load_documents,
    load_file,
)
from agenttune.rag.storage.trajectory_store import (
    TrajectoryRecord,
    TrajectoryStore,
)


class TestTrajectoryStore:
    """Trajectory database — store and retrieve agent trajectories."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_trajectories.db")
        self.store = TrajectoryStore(self.db_path)

    def test_register_run(self):
        run_id = self.store.register_run(
            "run1", "training", "Qwen3.5-4B", reward_stack="T3", config={"lr": 1e-6}
        )
        assert run_id == "run1"

    def test_store_and_retrieve(self):
        """Store a trajectory and retrieve it by question."""
        run_id = self.store.register_run("run1", "eval", "Qwen3.5-4B")
        record = TrajectoryRecord(
            run_id=run_id,
            question="What year was Cambridge founded?",
            gold_answer="1209",
            final_answer="<answer>1209</answer>",
            reward=0.84,
            n_tool_calls=2,
            has_answer_tag=True,
            condition="m1",
            model="Qwen3.5-4B",
            reward_components={"correctness": 1.0, "necessity": 0.67},
            steps=[
                {
                    "step_number": 1,
                    "thought": "I need to search",
                    "action_name": "search_corpus",
                    "observation": "Cambridge founded 1209",
                    "is_tool_step": True,
                }
            ],
        )
        traj_id = self.store.store(record)
        assert traj_id == record.trajectory_id

        # Retrieve by question
        results = self.store.get_by_question("What year was Cambridge founded?")
        assert len(results) == 1
        assert results[0]["gold_answer"] == "1209"
        assert results[0]["reward"] == 0.84

    def test_store_with_tool_calls(self):
        """Tool calls are stored and retrievable."""
        run_id = self.store.register_run("run1", "training", "test-model")
        record = TrajectoryRecord(
            run_id=run_id,
            question="Test Q",
            gold_answer="A",
            final_answer="<answer>A</answer>",
            reward=0.5,
            n_tool_calls=1,
            has_answer_tag=True,
            condition="plain",
            steps=[
                {
                    "step_number": 1,
                    "thought": "searching",
                    "action_name": "search_corpus",
                    "action_args": {"query": "test"},
                    "observation": "result",
                    "is_tool_step": True,
                    "tool_calls": [
                        {"tool_name": "search_corpus", "query": "test", "result": "result"}
                    ],
                }
            ],
        )
        self.store.store(record)
        tc = self.store.get_tool_calls(record.trajectory_id)
        assert len(tc) == 1
        assert tc[0]["tool_name"] == "search_corpus"
        assert tc[0]["query"] == "test"

    def test_get_full_trajectory(self):
        """Full trajectory includes steps and tool calls."""
        run_id = self.store.register_run("run1", "eval", "test")
        record = TrajectoryRecord(
            run_id=run_id,
            question="Full Q",
            gold_answer="gold",
            final_answer="answer",
            reward=0.7,
            n_tool_calls=2,
            has_answer_tag=True,
            condition="m1",
            reward_components={"format": 0.1, "correctness": 0.7},
            steps=[
                {
                    "step_number": 1,
                    "thought": "s1",
                    "action_name": "search_corpus",
                    "observation": "r1",
                    "is_tool_step": True,
                    "tool_calls": [{"tool_name": "search_corpus", "query": "q1", "result": "r1"}],
                },
                {
                    "step_number": 2,
                    "thought": "s2",
                    "action_name": "search_corpus",
                    "observation": "r2",
                    "is_tool_step": True,
                    "tool_calls": [{"tool_name": "search_corpus", "query": "q2", "result": "r2"}],
                },
            ],
        )
        self.store.store(record)
        full = self.store.get_full_trajectory(record.trajectory_id)
        assert full is not None
        assert full["question"] == "Full Q"
        assert len(full["steps"]) == 2
        assert full["reward_components"]["correctness"] == 0.7

    def test_get_by_reward_range(self):
        """Query trajectories by reward range."""
        run_id = self.store.register_run("run1", "training", "test")
        for i, r in enumerate([0.1, 0.5, 0.8, 0.3]):
            self.store.store(
                TrajectoryRecord(
                    run_id=run_id,
                    question=f"Q{i}",
                    gold_answer="a",
                    final_answer="a",
                    reward=r,
                    condition="plain",
                )
            )
        high = self.store.get_by_reward_range(0.5, 1.0)
        assert len(high) == 2  # 0.5 and 0.8
        assert all(r["reward"] >= 0.5 for r in high)

    def test_stats(self):
        run_id = self.store.register_run("run1", "training", "test")
        for i in range(5):
            self.store.store(
                TrajectoryRecord(
                    run_id=run_id,
                    question=f"Q{i}",
                    gold_answer="a",
                    final_answer="a",
                    reward=0.5,
                    n_tool_calls=3,
                    condition="plain",
                )
            )
        stats = self.store.stats()
        assert stats["total_trajectories"] == 5
        assert stats["total_runs"] == 1
        assert stats["avg_reward"] == 0.5

    def test_export_jsonl(self):
        run_id = self.store.register_run("run1", "eval", "test")
        self.store.store(
            TrajectoryRecord(
                run_id=run_id,
                question="Export Q",
                gold_answer="a",
                final_answer="a",
                reward=0.9,
                condition="m1",
            )
        )
        out_path = os.path.join(self.tmpdir, "export.jsonl")
        self.store.export_jsonl(out_path, run_id=run_id)
        with open(out_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["question"] == "Export Q"


class TestDocumentLoaders:
    """Multi-format document ingestion — load PDF, DOCX, PPTX, etc."""

    def test_supported_formats(self):
        formats = get_supported_formats()
        assert ".pdf" in formats
        assert ".docx" in formats
        assert ".pptx" in formats
        assert ".md" in formats
        assert ".txt" in formats
        assert ".csv" in formats
        assert ".json" in formats

    def test_load_text(self):
        """Plain text loading."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.")
            fpath = f.name
        docs = load_file(fpath)
        assert len(docs) == 1
        assert "test document" in docs[0].text
        assert docs[0].metadata["format"] == "text"
        os.unlink(fpath)

    def test_load_markdown(self):
        """Markdown loading."""
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# Title\n\nSome **markdown** content.")
            fpath = f.name
        docs = load_file(fpath)
        assert len(docs) == 1
        assert "markdown" in docs[0].text.lower()
        os.unlink(fpath)

    def test_load_csv(self):
        """CSV loading — one doc per row."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("title,content\nDoc1,Content one\nDoc2,Content two\n")
            fpath = f.name
        docs = load_file(fpath)
        assert len(docs) == 2
        assert "Content one" in docs[0].text
        assert "Content two" in docs[1].text
        os.unlink(fpath)

    def test_load_json_list(self):
        """JSON list loading — one doc per object."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(
                [{"name": "Doc1", "text": "Content one"}, {"name": "Doc2", "text": "Content two"}],
                f,
            )
            fpath = f.name
        docs = load_file(fpath)
        assert len(docs) == 2
        os.unlink(fpath)

    def test_load_json_dict(self):
        """JSON dict loading — one doc."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"name": "Doc", "content": "Single doc"}, f)
            fpath = f.name
        docs = load_file(fpath)
        assert len(docs) == 1
        os.unlink(fpath)

    def test_load_unsupported(self):
        """Unsupported format raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            fpath = f.name
        with pytest.raises(ValueError, match="Unsupported"):
            load_file(fpath)
        os.unlink(fpath)

    def test_load_documents_from_dir(self):
        """Load all files from a directory."""
        tmpdir = tempfile.mkdtemp()
        # Create multiple files
        with open(os.path.join(tmpdir, "doc1.txt"), "w") as f:
            f.write("Document one content")
        with open(os.path.join(tmpdir, "doc2.md"), "w") as f:
            f.write("# Document two")
        with open(os.path.join(tmpdir, "doc3.csv"), "w") as f:
            f.write("col1,col2\nval1,val2\n")
        docs = load_documents(tmpdir, recursive=True)
        assert len(docs) >= 3  # 1 txt + 1 md + 1 csv row

    def test_load_documents_filtered_extensions(self):
        """Only load specified extensions."""
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "doc1.txt"), "w") as f:
            f.write("text doc")
        with open(os.path.join(tmpdir, "doc2.md"), "w") as f:
            f.write("# markdown doc")
        docs = load_documents(tmpdir, extensions=[".txt"])
        assert len(docs) == 1
        assert docs[0].metadata["format"] == "text"

    def test_corpus_document_compat(self):
        """Loaded docs are CorpusDocument objects compatible with build_index."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("Test content for indexing")
            fpath = f.name
        docs = load_file(fpath)
        assert isinstance(docs[0], CorpusDocument)
        assert docs[0].doc_id
        assert docs[0].title
        assert docs[0].text
        os.unlink(fpath)
