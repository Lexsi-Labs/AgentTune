"""
Extended audit tests: JSONL writing, reading, DPO extraction, edge cases.
"""

import json
from pathlib import Path

from conftest import make_state

from agenttune.decide.audit import AuditReader, AuditWriter

# ---------------------------------------------------------------------------
# AuditWriter — write tests
# ---------------------------------------------------------------------------


class TestAuditWriterExtended:
    def test_writes_stage_entry_as_jsonl(self, tmp_dir):
        path = str(tmp_dir / "audit.jsonl")
        writer = AuditWriter(path)
        state = make_state(pipeline_id="pipe-x")
        stage = {"id": "extract", "type": "llm_call", "model": "claude-haiku-4-5"}
        result = {"output": {"name": "Alice"}, "latency_ms": 200, "cost_usd": 0.001}

        writer.log_stage(state, stage, result)

        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["stage_id"] == "extract"
        assert entry["pipeline_id"] == "pipe-x"

    def test_appends_multiple_stages(self, tmp_dir):
        path = str(tmp_dir / "audit.jsonl")
        writer = AuditWriter(path)
        state = make_state()

        for stage_id in ("extract", "check", "decide"):
            stage = {"id": stage_id, "type": "llm_call"}
            writer.log_stage(state, stage, {"output": {}, "latency_ms": 100})

        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["stage_id"] == "extract"
        assert json.loads(lines[2])["stage_id"] == "decide"

    def test_write_final_state(self, tmp_dir):
        path = str(tmp_dir / "audit.jsonl")
        writer = AuditWriter(path)
        state = make_state(
            verdict="APPROVE",
            is_complete=True,
            step_count=3,
            elapsed_seconds=1.5,
        )
        writer.write(state)

        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["verdict"] == "APPROVE"
        assert entry["is_complete"] is True

    def test_handles_missing_output_fields(self, tmp_dir):
        path = str(tmp_dir / "audit.jsonl")
        writer = AuditWriter(path)
        state = make_state()
        stage = {"id": "partial", "type": "rules"}
        result = {}  # minimal result, no output/latency keys

        writer.log_stage(state, stage, result)
        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == 1

    def test_records_error_field(self, tmp_dir):
        path = str(tmp_dir / "audit.jsonl")
        writer = AuditWriter(path)
        state = make_state()
        stage = {"id": "broken", "type": "llm_call"}
        result = {"error": "API timeout", "output": None}

        writer.log_stage(state, stage, result)
        entry = json.loads(Path(path).read_text().strip())
        assert entry["error"] == "API timeout"

    def test_records_iteration_count(self, tmp_dir):
        path = str(tmp_dir / "audit.jsonl")
        writer = AuditWriter(path)
        state = make_state(stage_iterations={"extract": 2})
        stage = {"id": "extract", "type": "llm_call"}
        writer.log_stage(state, stage, {"output": {}})

        entry = json.loads(Path(path).read_text().strip())
        assert entry["iteration"] == 2
        assert entry["is_retry"] is True

    def test_large_dataset_write(self, tmp_dir):
        """Write 500 entries and verify all are valid JSONL."""
        path = str(tmp_dir / "large_audit.jsonl")
        writer = AuditWriter(path)
        state = make_state()

        for i in range(500):
            stage = {"id": f"stage_{i}", "type": "llm_call"}
            writer.log_stage(state, stage, {"output": {"index": i}, "latency_ms": 100})

        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) == 500
        for line in lines:
            entry = json.loads(line)
            assert "stage_id" in entry

    def test_concurrent_write_no_corruption(self, tmp_dir):
        """Writing from multiple AuditWriter instances doesn't corrupt the file."""
        import threading

        path = str(tmp_dir / "concurrent.jsonl")
        state = make_state()
        errors = []

        def write_10(thread_id):
            w = AuditWriter(path)
            for i in range(10):
                try:
                    stage = {"id": f"stage_{thread_id}_{i}", "type": "llm_call"}
                    w.log_stage(state, stage, {"output": {}})
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=write_10, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        lines = [json.loads(l) for l in Path(path).read_text().strip().split("\n")]
        assert len(lines) == 50


# ---------------------------------------------------------------------------
# AuditReader — read tests
# ---------------------------------------------------------------------------


class TestAuditReaderExtended:
    def test_reads_all_entries(self, audit_file, sample_audit_entries):
        reader = AuditReader(audit_file)
        entries = reader.read_all()
        assert len(entries) == len(sample_audit_entries)

    def test_filter_by_stage_id(self, audit_file):
        reader = AuditReader(audit_file)
        entries = reader.filter_by_stage("extract")
        assert all(e["stage_id"] == "extract" for e in entries)
        assert len(entries) >= 1

    def test_extract_dpo_pairs(self, audit_file):
        reader = AuditReader(audit_file)
        pairs = reader.extract_dpo_pairs("decision_judge")
        assert len(pairs) >= 1
        pair = pairs[0]
        assert "input" in pair or "prompt" in pair
        assert "rejected_output" in pair or "rejected" in pair
        assert "chosen_output" in pair or "chosen" in pair

    def test_no_dpo_pairs_for_clean_stage(self, audit_file):
        reader = AuditReader(audit_file)
        pairs = reader.extract_dpo_pairs("extract")  # no human feedback entries
        assert len(pairs) == 0

    def test_handles_malformed_lines(self, tmp_dir):
        path = tmp_dir / "bad.jsonl"
        path.write_text('{"stage_id": "x"}\nNOT_JSON\n{"stage_id": "y"}\n')
        reader = AuditReader(str(path))
        entries = reader.read_all()
        assert len(entries) == 2  # malformed line skipped

    def test_handles_empty_file(self, tmp_dir):
        path = tmp_dir / "empty.jsonl"
        path.write_text("")
        reader = AuditReader(str(path))
        assert reader.read_all() == []

    def test_filter_by_pipeline_id(self, audit_file):
        reader = AuditReader(audit_file)
        entries = reader.filter_by_stage("extract")
        assert len(entries) >= 1

    def test_extract_trajectories_returns_dataset(self, audit_file):
        from agenttune.decide.training_bridge import DecideToTrainerBridge

        bridge = DecideToTrainerBridge(audit_file)
        ds = bridge.extract_trajectories("decision_judge")
        assert hasattr(ds, "__len__")

    def test_extract_bco_labels(self, audit_file):
        from agenttune.decide.training_bridge import DecideToTrainerBridge

        bridge = DecideToTrainerBridge(audit_file)
        labels = bridge.extract_bco_labels("approve")
        assert isinstance(labels, list)

    def test_read_all_returns_list_of_dicts(self, audit_file):
        reader = AuditReader(audit_file)
        entries = reader.read_all()
        assert isinstance(entries, list)
        assert all(isinstance(e, dict) for e in entries)

    def test_audit_reader_file_not_found(self, tmp_dir):
        reader = AuditReader(str(tmp_dir / "missing.jsonl"))
        assert reader.read_all() == []

    def test_multiple_pipelines_in_audit(self, tmp_dir):
        """Verify entries from two pipelines are both returned."""
        path = tmp_dir / "multi.jsonl"
        entries = [
            {"pipeline_id": "A", "stage_id": "extract", "stage_type": "llm_call"},
            {"pipeline_id": "B", "stage_id": "extract", "stage_type": "llm_call"},
        ]
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        reader = AuditReader(str(path))
        all_entries = reader.read_all()
        pipeline_ids = {e["pipeline_id"] for e in all_entries}
        assert "A" in pipeline_ids and "B" in pipeline_ids
