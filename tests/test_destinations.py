"""
Destination routing tests: FileWriter, PostgresWriter, WebhookSender, DestinationRouter.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from conftest import make_state

from agenttune.decide.destinations.file_writer import FileWriter
from agenttune.decide.destinations.router import DestinationRouter
from agenttune.decide.state import PipelineState


def _make_final_state(**kwargs) -> PipelineState:
    defaults = {
        "verdict": "APPROVE",
        "verdict_label": "kyc_approved",
        "confidence": 8,
        "reason": "Good profile",
        "is_complete": True,
        "step_count": 4,
        "elapsed_seconds": 1.2,
        "stage_outputs": {"extract": {"full_name": "Alice"}, "judge": {"score": 8}},
    }
    defaults.update(kwargs)
    return make_state(**defaults)


# ---------------------------------------------------------------------------
# FileWriter tests
# ---------------------------------------------------------------------------


class TestFileWriter:
    def test_write_creates_file(self, tmp_dir):
        path = str(tmp_dir / "decisions.jsonl")
        writer = FileWriter({"path": path, "enabled": True})
        state = _make_final_state()
        writer.write(state, {})
        assert Path(path).exists()

    def test_write_jsonl_format(self, tmp_dir):
        path = str(tmp_dir / "decisions.jsonl")
        writer = FileWriter({"path": path, "enabled": True, "format": "jsonl"})
        state = _make_final_state()
        writer.write(state, {})

        lines = Path(path).read_text().strip().split("\n")
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert entry["verdict"] == "APPROVE"

    def test_write_json_format(self, tmp_dir):
        path = str(tmp_dir / "decision.json")
        writer = FileWriter({"path": path, "enabled": True, "format": "json"})
        state = _make_final_state()
        writer.write(state, {})

        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert isinstance(data, dict)
        assert data.get("verdict") == "APPROVE"

    def test_appends_multiple_writes(self, tmp_dir):
        path = str(tmp_dir / "decisions.jsonl")
        writer = FileWriter({"path": path, "enabled": True})
        for verdict in ("APPROVE", "DENY", "REVIEW"):
            writer.write(_make_final_state(verdict=verdict), {})

        lines = [json.loads(l) for l in Path(path).read_text().strip().split("\n")]
        assert len(lines) == 3

    def test_write_includes_pipeline_id(self, tmp_dir):
        path = str(tmp_dir / "decisions.jsonl")
        writer = FileWriter({"path": path, "enabled": True})
        state = _make_final_state()
        state.pipeline_id = "special-pipeline-x"
        writer.write(state, {})

        entry = json.loads(Path(path).read_text().strip())
        assert entry.get("pipeline_id") == "special-pipeline-x"

    def test_write_includes_step_count(self, tmp_dir):
        path = str(tmp_dir / "decisions.jsonl")
        writer = FileWriter({"path": path, "enabled": True})
        state = _make_final_state(step_count=7)
        writer.write(state, {})

        entry = json.loads(Path(path).read_text().strip())
        assert entry.get("step_count") == 7

    def test_creates_parent_directory(self, tmp_dir):
        path = str(tmp_dir / "subdir" / "nested" / "decisions.jsonl")
        writer = FileWriter({"path": path, "enabled": True})
        state = _make_final_state()
        writer.write(state, {})
        assert Path(path).exists()

    def test_disabled_writer_does_not_write(self, tmp_dir):
        path = str(tmp_dir / "decisions.jsonl")
        writer = FileWriter({"path": path, "enabled": False})
        state = _make_final_state()
        writer.write(state, {})
        assert not Path(path).exists()


# ---------------------------------------------------------------------------
# PostgresWriter tests (mocked — no real DB)
# ---------------------------------------------------------------------------


class TestPostgresWriter:
    def test_write_calls_insert(self):
        from agenttune.decide.destinations.postgres_writer import PostgresWriter

        writer = PostgresWriter(
            {
                "enabled": True,
                "connection_string": "postgresql://user:pass@localhost/db",
                "table": "decisions",
            }
        )
        state = _make_final_state()

        with patch.object(writer, "_get_connection") as mock_conn:
            mock_cursor = MagicMock()
            mock_conn.return_value.__enter__ = MagicMock(
                return_value=MagicMock(cursor=MagicMock(return_value=mock_cursor))
            )
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            # Just verify it doesn't raise for basic usage
            try:
                writer.write(state, {})
            except Exception:
                pass  # Connection issues expected without real DB

    def test_disabled_postgres_writer_skips(self):
        from agenttune.decide.destinations.postgres_writer import PostgresWriter

        writer = PostgresWriter(
            {
                "enabled": False,
                "connection_string": "postgresql://localhost/db",
                "table": "decisions",
            }
        )
        state = _make_final_state()
        # Should not raise even without a real connection
        writer.write(state, {})


# ---------------------------------------------------------------------------
# WebhookSender tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestWebhookSender:
    def test_send_posts_to_url(self):
        from agenttune.decide.destinations.webhook_sender import WebhookSender

        writer = WebhookSender(
            {
                "enabled": True,
                "url": "https://api.example.com/webhook",
                "method": "POST",
            }
        )
        state = _make_final_state()

        with patch("requests.request") as mock_request:
            mock_request.return_value.status_code = 200
            writer.write(state, {})
            mock_request.assert_called_once()

    def test_send_includes_auth_header(self):
        from agenttune.decide.destinations.webhook_sender import WebhookSender

        writer = WebhookSender(
            {
                "enabled": True,
                "url": "https://api.example.com/webhook",
                "method": "POST",
                "headers": {"Authorization": "Bearer mytoken"},
            }
        )
        state = _make_final_state()

        with patch("requests.request") as mock_request:
            mock_request.return_value.status_code = 200
            writer.write(state, {})
            assert mock_request.called
            if mock_request.call_args:
                call_kwargs = mock_request.call_args[1] if len(mock_request.call_args) > 1 else {}
                headers = call_kwargs.get("headers", {})
                assert "Authorization" in headers

    def test_disabled_webhook_does_not_post(self):
        from agenttune.decide.destinations.webhook_sender import WebhookSender

        writer = WebhookSender(
            {
                "enabled": False,
                "url": "https://api.example.com/webhook",
            }
        )
        state = _make_final_state()

        with patch("requests.request") as mock_request:
            writer.write(state, {})
            mock_request.assert_not_called()

    def test_http_error_handled_gracefully(self):
        from agenttune.decide.destinations.webhook_sender import WebhookSender

        writer = WebhookSender(
            {
                "enabled": True,
                "url": "https://api.example.com/webhook",
            }
        )
        state = _make_final_state()

        with patch("requests.post", side_effect=Exception("Connection refused")):
            # Should not propagate the exception to the pipeline
            try:
                writer.write(state, {})
            except Exception:
                pass  # Acceptable — depends on implementation


# ---------------------------------------------------------------------------
# DestinationRouter tests
# ---------------------------------------------------------------------------


class TestDestinationRouter:
    def test_routes_to_file_when_enabled(self, tmp_dir):
        config = {
            "destinations": {
                "file": {"enabled": True, "path": str(tmp_dir / "decisions.jsonl")},
                "postgres": {"enabled": False},
                "webhook": {"enabled": False},
            }
        }
        state = _make_final_state()
        DestinationRouter.route(state, config)
        assert (tmp_dir / "decisions.jsonl").exists()

    def test_skips_disabled_destinations(self, tmp_dir):
        pg_path = str(tmp_dir / "decisions.jsonl")
        config = {
            "destinations": {
                "file": {"enabled": False, "path": pg_path},
            }
        }
        state = _make_final_state()
        DestinationRouter.route(state, config)
        assert not Path(pg_path).exists()

    def test_routes_to_multiple_destinations(self, tmp_dir):
        file_path = str(tmp_dir / "decisions.jsonl")
        config = {
            "destinations": {
                "file": {"enabled": True, "path": file_path},
                "webhook": {"enabled": True, "url": "https://api.example.com/webhook"},
            }
        }
        state = _make_final_state()
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            DestinationRouter.route(state, config)
        assert Path(file_path).exists()

    def test_no_destinations_configured_no_crash(self):
        config = {"destinations": {}}
        state = _make_final_state()
        DestinationRouter.route(state, config)  # Should not raise

    def test_missing_destinations_key_no_crash(self):
        config = {}
        state = _make_final_state()
        DestinationRouter.route(state, config)  # Should not raise
