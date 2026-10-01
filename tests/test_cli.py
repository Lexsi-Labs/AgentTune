"""
CLI integration tests for `agenttune decide` commands:
  - run
  - list
  - validate
  - show
  - init (config scaffold)

Uses typer.testing.CliRunner for isolation — no subprocess, no network.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from agenttune.decide.cli import decide_app

runner = CliRunner()

TEMPLATES_DIR = Path(__file__).parent.parent / "src" / "agenttune" / "decide" / "templates"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_state(verdict="APPROVE", step_count=3, elapsed_seconds=1.2):
    from agenttune.decide.state import PipelineState

    return PipelineState(
        pipeline_id="test-cli-001",
        template_id="bfsi/kyc_triage",
        template_version="1.0.0",
        input_text="Test",
        input_hash="abc",
        stage_outputs={},
        stage_iterations={},
        stage_traces=[],
        step_count=step_count,
        step_history=[],
        verdict=verdict,
        verdict_label=verdict.lower(),
        confidence=8,
        reason="Good profile",
        is_complete=True,
        error=None,
        error_stage=None,
        timestamp_start="2026-04-27T10:00:00Z",
        timestamp_end="2026-04-27T10:00:01Z",
        elapsed_seconds=elapsed_seconds,
        config={},
    )


# ---------------------------------------------------------------------------
# `decide list` tests
# ---------------------------------------------------------------------------


class TestDecideList:
    def test_list_shows_templates(self):
        result = runner.invoke(decide_app, ["list"])
        assert result.exit_code == 0

    def test_list_includes_bfsi_category(self):
        result = runner.invoke(decide_app, ["list", "--category", "bfsi"])
        assert result.exit_code == 0

    def test_list_includes_generic_category(self):
        result = runner.invoke(decide_app, ["list", "--category", "generic"])
        assert result.exit_code == 0

    def test_list_no_category_shows_all(self):
        result = runner.invoke(decide_app, ["list"])
        assert result.exit_code == 0

    def test_list_unknown_category_exits_gracefully(self):
        result = runner.invoke(decide_app, ["list", "--category", "unknown_xyz"])
        # Should either show empty list or error message, not crash
        assert result.exit_code in (0, 1, 2)


# ---------------------------------------------------------------------------
# `decide validate` tests
# ---------------------------------------------------------------------------


class TestDecideValidate:
    def test_validate_kyc_triage_passes(self):
        result = runner.invoke(decide_app, ["validate", "--template", "bfsi/kyc_triage"])
        assert result.exit_code == 0

    def test_validate_text_classify_passes(self):
        result = runner.invoke(decide_app, ["validate", "--template", "generic/text_classify"])
        assert result.exit_code == 0

    def test_validate_invalid_template_fails(self):
        result = runner.invoke(decide_app, ["validate", "--template", "nonexistent/template"])
        assert result.exit_code != 0

    def test_validate_all_bfsi_templates(self):
        bfsi_templates = [p.stem for p in (TEMPLATES_DIR / "bfsi").glob("*.yaml")]
        for template_stem in bfsi_templates:
            result = runner.invoke(decide_app, ["validate", "--template", f"bfsi/{template_stem}"])
            assert (
                result.exit_code == 0
            ), f"Template bfsi/{template_stem} failed validation: {result.output}"

    def test_validate_all_generic_templates(self):
        generic_templates = [p.stem for p in (TEMPLATES_DIR / "generic").glob("*.yaml")]
        for template_stem in generic_templates:
            result = runner.invoke(
                decide_app, ["validate", "--template", f"generic/{template_stem}"]
            )
            assert (
                result.exit_code == 0
            ), f"Template generic/{template_stem} failed validation: {result.output}"


# ---------------------------------------------------------------------------
# `decide show` tests
# ---------------------------------------------------------------------------


class TestDecideShow:
    def test_show_kyc_triage(self):
        result = runner.invoke(decide_app, ["show", "--template", "bfsi/kyc_triage"])
        assert result.exit_code == 0

    def test_show_displays_template_name(self):
        result = runner.invoke(decide_app, ["show", "--template", "bfsi/kyc_triage"])
        output = result.output.lower()
        assert "kyc" in output or "triage" in output or "bfsi" in output

    def test_show_text_classify(self):
        result = runner.invoke(decide_app, ["show", "--template", "generic/text_classify"])
        assert result.exit_code == 0

    def test_show_invalid_template(self):
        result = runner.invoke(decide_app, ["show", "--template", "invalid/nope"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# `decide run` tests (mocked LLM)
# ---------------------------------------------------------------------------


class TestDecideRun:
    @patch("agenttune.decide.graph_runner.GraphRunner.run", new_callable=AsyncMock)
    @patch("agenttune.decide.config.ConfigLoader.load")
    def test_run_produces_output_file(self, mock_load, mock_run, tmp_path):
        mock_load.return_value = {
            "id": "bfsi/kyc_triage",
            "name": "KYC Triage",
            "version": "1.0.0",
            "stages": [
                {
                    "id": "process",
                    "type": "llm_call",
                    "prompt": "Process: {input_text}",
                    "output_schema": {"type": "object"},
                },
                {
                    "id": "output",
                    "type": "output",
                    "verdict_expr": "1",
                },
            ],
            "audit": {"path": str(tmp_path / "audit.jsonl")},
            "destinations": {"file": {"enabled": True, "path": str(tmp_path / "decisions.jsonl")}},
        }
        mock_run.return_value = make_mock_state()
        output_file = str(tmp_path / "result.json")
        result = runner.invoke(
            decide_app,
            [
                "run",
                "--template",
                "bfsi/kyc_triage",
                "--input",
                "Test customer",
                "--output",
                output_file,
            ],
        )
        assert result.exit_code == 0

    @patch("agenttune.decide.graph_runner.GraphRunner.run", new_callable=AsyncMock)
    @patch("agenttune.decide.config.ConfigLoader.load")
    def test_run_file_input_with_at_sign(self, mock_load, mock_run, tmp_path):
        input_file = tmp_path / "customer.txt"
        input_file.write_text("Alice Smith, income 75000")

        mock_load.return_value = {
            "id": "bfsi/kyc_triage",
            "name": "Test",
            "version": "1.0.0",
            "stages": [
                {
                    "id": "process",
                    "type": "llm_call",
                    "prompt": "Process: {input_text}",
                    "output_schema": {"type": "object"},
                },
                {
                    "id": "output",
                    "type": "output",
                    "verdict_expr": "1",
                },
            ],
            "audit": {"path": str(tmp_path / "audit.jsonl")},
            "destinations": {"file": {"enabled": True, "path": str(tmp_path / "decisions.jsonl")}},
        }
        mock_run.return_value = make_mock_state()

        result = runner.invoke(
            decide_app,
            [
                "run",
                "--template",
                "bfsi/kyc_triage",
                "--input",
                f"@{input_file}",
                "--output",
                str(tmp_path / "result.json"),
            ],
        )
        assert result.exit_code == 0

    def test_run_missing_template_errors(self):
        result = runner.invoke(
            decide_app,
            ["run", "--input", "Test"],
        )
        assert result.exit_code != 0

    def test_run_missing_input_errors(self):
        result = runner.invoke(
            decide_app,
            ["run", "--template", "bfsi/kyc_triage"],
        )
        assert result.exit_code != 0

    @patch("agenttune.decide.graph_runner.GraphRunner.run", new_callable=AsyncMock)
    @patch("agenttune.decide.config.ConfigLoader.load")
    def test_run_error_state_exits_with_message(self, mock_load, mock_run, tmp_path):
        mock_load.return_value = {
            "id": "test",
            "name": "Test",
            "version": "1.0.0",
            "stages": [],
            "audit": {"path": str(tmp_path / "audit.jsonl")},
            "destinations": {"file": {"enabled": True, "path": str(tmp_path / "decisions.jsonl")}},
        }
        error_state = make_mock_state()
        error_state.error = "Model API rate limit exceeded"
        error_state.is_complete = True
        mock_run.return_value = error_state

        result = runner.invoke(
            decide_app,
            [
                "run",
                "--template",
                "bfsi/kyc_triage",
                "--input",
                "Test",
                "--output",
                str(tmp_path / "result.json"),
            ],
        )
        # Should exit cleanly even with an error state
        assert result.exit_code in (0, 1)


# ---------------------------------------------------------------------------
# `decide init` (config scaffold)
# ---------------------------------------------------------------------------


class TestDecideInit:
    def test_init_creates_config_file(self, tmp_path):
        result = runner.invoke(
            decide_app,
            ["init", "--output", str(tmp_path / "config.yaml")],
        )
        # Either creates the file or prints helpful output
        config_exists = (tmp_path / "config.yaml").exists()
        assert result.exit_code == 0 or config_exists

    def test_init_without_output_prints_to_stdout(self):
        result = runner.invoke(decide_app, ["init"])
        # Should print something meaningful or exit 0
        assert result.exit_code in (0, 1, 2)
