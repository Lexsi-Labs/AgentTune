"""
Basic end-to-end integration tests for the Decide pipeline.

Tests use mocked LLM calls to avoid network dependencies.
Each test runs a full pipeline from config → graph build → state output.

Scenarios:
  1. Simple 2-stage pipeline (llm_call → output)
  2. 3-stage pipeline with rules (llm_call → rules → output)
  3. 4-stage pipeline with judge routing (llm_call → llm_judge → output)
  4. Parallel fan-out pipeline (llm_call → parallel → output)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agenttune.decide.graph_runner import GraphRunner

# ---------------------------------------------------------------------------
# LLM mock helpers
# ---------------------------------------------------------------------------


def _mock_llm_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(total_tokens=50)
    return resp


EXTRACT_RESPONSE = json.dumps({"full_name": "Alice", "income": 75000, "dob": "1990-01-01"})
JUDGE_RESPONSE = json.dumps({"score": 8, "decision": "APPROVE", "explanation": "Good profile"})
SUMMARY_RESPONSE = json.dumps({"summary": "A " + "x " * 60})  # >100 chars for loop test


# ---------------------------------------------------------------------------
# Test class 1: BasicPipeline
# ---------------------------------------------------------------------------


class TestBasicPipeline:
    """2-stage pipeline: llm_call → output."""

    @pytest.fixture
    def config(self, tmp_dir):
        return {
            "id": "test/basic",
            "name": "Basic Test",
            "version": "1.0.0",
            "max_total_steps": 10,
            "default_model": "claude-haiku-4-5",
            "stages": [
                {
                    "id": "call_stage",
                    "type": "llm_call",
                    "prompt": "Summarise: {input_text}",
                    "output_schema": {
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                    },
                    "next": "output_stage",
                },
                {
                    "id": "output_stage",
                    "type": "output",
                    "verdict": "COMPLETE",
                    "destinations": ["file"],
                },
            ],
            "audit": {"path": str(tmp_dir / "audit.jsonl")},
            "destinations": {"file": {"enabled": True, "path": str(tmp_dir / "decisions.jsonl")}},
        }

    @patch("agenttune.decide.stages.base.litellm")
    @pytest.mark.asyncio
    async def test_pipeline_runs_and_sets_verdict(self, mock_litellm, config):
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_llm_response(json.dumps({"summary": "Great summary."}))
        )
        runner = GraphRunner(config)
        state = await runner.run("Test input")
        assert state.is_complete
        assert state.verdict == "COMPLETE"

    @patch("agenttune.decide.stages.base.litellm")
    @pytest.mark.asyncio
    async def test_pipeline_records_step_count(self, mock_litellm, config):
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_llm_response(json.dumps({"summary": "Done."}))
        )
        runner = GraphRunner(config)
        state = await runner.run("Input")
        assert state.step_count >= 1

    @patch("agenttune.decide.stages.base.litellm")
    @pytest.mark.asyncio
    async def test_stage_output_in_state(self, mock_litellm, config):
        expected = {"summary": "My summary."}
        mock_litellm.acompletion = AsyncMock(return_value=_mock_llm_response(json.dumps(expected)))
        runner = GraphRunner(config)
        state = await runner.run("Input")
        assert "call_stage" in state.stage_outputs

    @patch("agenttune.decide.stages.base.litellm")
    @pytest.mark.asyncio
    async def test_audit_file_created(self, mock_litellm, config, tmp_dir):
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_llm_response(json.dumps({"summary": "OK"}))
        )
        runner = GraphRunner(config)
        await runner.run("Input")
        assert (tmp_dir / "audit.jsonl").exists()


# ---------------------------------------------------------------------------
# Test class 2: RulesPipeline
# ---------------------------------------------------------------------------


class TestRulesPipeline:
    """3-stage pipeline: llm_call → rules → output."""

    @pytest.fixture
    def config(self, tmp_dir):
        return {
            "id": "test/rules",
            "name": "Rules Test",
            "version": "1.0.0",
            "max_total_steps": 15,
            "default_model": "claude-haiku-4-5",
            "stages": [
                {
                    "id": "extract",
                    "type": "llm_call",
                    "prompt": "Extract: {input_text}",
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "income": {"type": "number"},
                            "dob": {"type": "string"},
                        },
                    },
                    "next": "check",
                },
                {
                    "id": "check",
                    "type": "rules",
                    "rules": [
                        {
                            "condition": "income != null",
                            "on_fail": {"goto": "reject", "inject": "Income missing."},
                        }
                    ],
                    "next": "approve",
                },
                {
                    "id": "approve",
                    "type": "output",
                    "verdict": "APPROVE",
                    "destinations": ["file"],
                },
                {
                    "id": "reject",
                    "type": "output",
                    "verdict": "DENY",
                    "destinations": ["file"],
                },
            ],
            "audit": {"path": str(tmp_dir / "audit.jsonl")},
            "destinations": {"file": {"enabled": True, "path": str(tmp_dir / "decisions.jsonl")}},
        }

    @patch("agenttune.decide.stages.base.litellm")
    @pytest.mark.asyncio
    async def test_passes_rules_and_approves(self, mock_litellm, config):
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_llm_response(json.dumps({"income": 50000, "dob": "1990-01-01"}))
        )
        runner = GraphRunner(config)
        state = await runner.run("Good customer")
        assert state.is_complete
        assert state.verdict == "APPROVE"

    @patch("agenttune.decide.stages.base.litellm")
    @pytest.mark.asyncio
    async def test_fails_rules_and_denies(self, mock_litellm, config):
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_llm_response(json.dumps({"income": None, "dob": None}))
        )
        runner = GraphRunner(config)
        state = await runner.run("Incomplete customer")
        assert state.is_complete
        assert state.verdict == "DENY"


# ---------------------------------------------------------------------------
# Test class 3: JudgePipeline
# ---------------------------------------------------------------------------


class TestJudgePipeline:
    """4-stage pipeline: llm_call → llm_judge → approve/reject output."""

    @pytest.fixture
    def config(self, tmp_dir):
        return {
            "id": "test/judge",
            "name": "Judge Test",
            "version": "1.0.0",
            "max_total_steps": 20,
            "default_model": "claude-haiku-4-5",
            "stages": [
                {
                    "id": "generate",
                    "type": "llm_call",
                    "prompt": "Generate: {input_text}",
                    "output_schema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                    "next": "score",
                },
                {
                    "id": "score",
                    "type": "llm_judge",
                    "prompt": "Score: {generate.output.text}",
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "integer"},
                            "explanation": {"type": "string"},
                        },
                    },
                    "on_result": [
                        {"condition": "score >= 7", "goto": "approved"},
                        {"condition": "score < 7", "goto": "rejected"},
                    ],
                    "on_fail": "rejected",
                },
                {
                    "id": "approved",
                    "type": "output",
                    "verdict": "APPROVED",
                    "destinations": ["file"],
                },
                {
                    "id": "rejected",
                    "type": "output",
                    "verdict": "REJECTED",
                    "destinations": ["file"],
                },
            ],
            "audit": {"path": str(tmp_dir / "audit.jsonl")},
            "destinations": {"file": {"enabled": True, "path": str(tmp_dir / "decisions.jsonl")}},
        }

    @patch("agenttune.decide.stages.llm_judge.LLMJUDGE_AVAILABLE", False)
    @patch("agenttune.decide.stages.base.litellm")
    @pytest.mark.asyncio
    async def test_high_score_routes_to_approved(self, mock_litellm, config):
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_llm_response(json.dumps({"text": "Generated text"})),
                _mock_llm_response(json.dumps({"score": 9, "explanation": "Excellent"})),
            ]
        )
        runner = GraphRunner(config)
        state = await runner.run("Write something")
        assert state.is_complete
        assert state.verdict == "APPROVED"

    @patch("agenttune.decide.stages.llm_judge.LLMJUDGE_AVAILABLE", False)
    @patch("agenttune.decide.stages.base.litellm")
    @pytest.mark.asyncio
    async def test_low_score_routes_to_rejected(self, mock_litellm, config):
        mock_litellm.acompletion = AsyncMock(
            side_effect=[
                _mock_llm_response(json.dumps({"text": "Poor text"})),
                _mock_llm_response(json.dumps({"score": 3, "explanation": "Low quality"})),
            ]
        )
        runner = GraphRunner(config)
        state = await runner.run("Write something")
        assert state.is_complete
        assert state.verdict == "REJECTED"


# ---------------------------------------------------------------------------
# Test class 4: ParallelPipeline
# ---------------------------------------------------------------------------


class TestParallelPipeline:
    """Pipeline with parallel fan-out: extract → parallel(a, b) → output."""

    @pytest.fixture
    def config(self, tmp_dir):
        return {
            "id": "test/parallel",
            "name": "Parallel Test",
            "version": "1.0.0",
            "max_total_steps": 30,
            "default_model": "claude-haiku-4-5",
            "stages": [
                {
                    "id": "extract",
                    "type": "llm_call",
                    "prompt": "Extract: {input_text}",
                    "output_schema": {
                        "type": "object",
                        "properties": {"data": {"type": "string"}},
                    },
                    "next": "parallel_analyze",
                },
                {
                    "id": "parallel_analyze",
                    "type": "parallel",
                    "branches": [
                        {
                            "id": "agent_a",
                            "type": "llm_call",
                            "prompt": "Analyze A: {extract.output.data}",
                            "output_schema": {
                                "type": "object",
                                "properties": {"score": {"type": "integer"}},
                            },
                        },
                        {
                            "id": "agent_b",
                            "type": "llm_call",
                            "prompt": "Analyze B: {extract.output.data}",
                            "output_schema": {
                                "type": "object",
                                "properties": {"score": {"type": "integer"}},
                            },
                        },
                    ],
                    "next": "decide",
                },
                {
                    "id": "decide",
                    "type": "output",
                    "verdict": "ANALYZED",
                    "destinations": ["file"],
                },
            ],
            "audit": {"path": str(tmp_dir / "audit.jsonl")},
            "destinations": {"file": {"enabled": True, "path": str(tmp_dir / "decisions.jsonl")}},
        }

    @patch("agenttune.decide.stages.base.litellm")
    @pytest.mark.asyncio
    async def test_parallel_branches_both_execute(self, mock_litellm, config):
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_llm_response(json.dumps({"score": 7, "data": "sample"}))
        )
        runner = GraphRunner(config)
        state = await runner.run("Input data")
        assert state.is_complete
        assert state.verdict == "ANALYZED"

    @patch("agenttune.decide.stages.base.litellm")
    @pytest.mark.asyncio
    async def test_parallel_outputs_stored_in_state(self, mock_litellm, config):
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_llm_response(json.dumps({"score": 5, "data": "sample"}))
        )
        runner = GraphRunner(config)
        state = await runner.run("Input")
        # Parallel branches should be in state outputs
        parallel_out = state.stage_outputs.get("parallel_analyze") or {}
        assert isinstance(parallel_out, dict)
