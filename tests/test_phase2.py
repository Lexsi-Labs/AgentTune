"""
Phase 2 unit tests:
  - AuditWriter._compute_reward()       — all 4 reward functions
  - AuditWriter._compute_episode_reward() — weighted_mean aggregation
  - GraphRunner._validate_observation() — string / object / array schemas
  - GraphRunner.run_episodes()          — batched, shuffle, seed
  - CollectRunner._extract_record()     — BCO / DPO / GRPO formats
  - CollectRunner.run()                 — episode loop, min_samples cap
  - CollectRunner auto_train trigger    — calls bridge / skips when no config
  - EvalRunner                          — accuracy, latency, cost, judge_score_mean
  - EvalRunner threshold checking       — violations and passes
  - CLI --mode collect / eval / train   — dispatch and exit codes
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from agenttune.decide.audit import AuditWriter
from agenttune.decide.cli import decide_app
from agenttune.decide.collect_runner import CollectRunner
from agenttune.decide.eval_runner import EvalRunner
from agenttune.decide.graph_runner import GraphRunner
from agenttune.decide.state import PipelineState

cli_runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _state(
    pipeline_id="pipe-001",
    verdict="APPROVE",
    elapsed_seconds=1.0,
    config=None,
    stage_outputs=None,
    stage_iterations=None,
    is_complete=True,
    input_text="test input",
) -> PipelineState:
    return PipelineState(
        pipeline_id=pipeline_id,
        template_id="test/template",
        template_version="1.0.0",
        input_text=input_text,
        input_hash="deadbeef",
        stage_outputs=stage_outputs or {},
        stage_iterations=stage_iterations or {},
        stage_traces=[],
        step_count=3,
        step_history=[],
        verdict=verdict,
        verdict_label=(verdict.lower() if verdict else None),
        confidence=8,
        reason="Test reason",
        is_complete=is_complete,
        error=None,
        error_stage=None,
        timestamp_start="2026-04-27T10:00:00",
        timestamp_end="2026-04-27T10:00:01",
        elapsed_seconds=elapsed_seconds,
        config=config or {},
    )


def _reward_cfg(*stage_defs):
    """Build config dict with reward.stages from (id, fn, weight) tuples."""
    stages = {sid: {"fn": fn, "weight": w} for sid, fn, w in stage_defs}
    return {"reward": {"stages": stages, "final_fn": "weighted_mean"}}


def _minimal_graph_config():
    return {
        "id": "test/minimal",
        "name": "Minimal",
        "version": "1.0.0",
        "max_total_steps": 5,
        "stages": [
            {"id": "out", "type": "output", "verdict": "DONE", "destinations": []},
        ],
    }


# ---------------------------------------------------------------------------
# 1.  AuditWriter._compute_reward()
# ---------------------------------------------------------------------------


class TestComputeReward:

    def _writer(self, tmp_path) -> AuditWriter:
        return AuditWriter(str(tmp_path / "audit.jsonl"))

    # --- stage not in reward config ----------------------------------------

    def test_returns_none_for_unconfigured_stage(self, tmp_path):
        w = self._writer(tmp_path)
        s = _state(config={"reward": {"stages": {}}})
        assert w._compute_reward(s, {"id": "ghost_stage"}, {}) is None

    # --- verdict_binary -------------------------------------------------------

    @pytest.mark.parametrize("verdict", ["APPROVE", "PASS", "COMPLETE"])
    def test_verdict_binary_pass_verdicts_return_1(self, tmp_path, verdict):
        w = self._writer(tmp_path)
        s = _state(config=_reward_cfg(("s1", "verdict_binary", 1.0)), verdict=verdict)
        assert w._compute_reward(s, {"id": "s1"}, {}) == pytest.approx(1.0)

    @pytest.mark.parametrize("verdict", ["DENY", "FAIL", "REJECT", "ERROR"])
    def test_verdict_binary_fail_verdicts_return_0(self, tmp_path, verdict):
        w = self._writer(tmp_path)
        s = _state(config=_reward_cfg(("s1", "verdict_binary", 1.0)), verdict=verdict)
        assert w._compute_reward(s, {"id": "s1"}, {}) == pytest.approx(0.0)

    # --- judge_score ----------------------------------------------------------

    def test_judge_score_normalises_to_0_1(self, tmp_path):
        w = self._writer(tmp_path)
        s = _state(config=_reward_cfg(("judge", "judge_score", 1.0)))
        assert w._compute_reward(
            s, {"id": "judge"}, {"output": {"overall_score": 8}}
        ) == pytest.approx(0.8)

    def test_judge_score_uses_consensus_score_fallback(self, tmp_path):
        w = self._writer(tmp_path)
        s = _state(config=_reward_cfg(("judge", "judge_score", 1.0)))
        assert w._compute_reward(
            s, {"id": "judge"}, {"output": {"consensus_score": 5}}
        ) == pytest.approx(0.5)

    def test_judge_score_clamps_above_10(self, tmp_path):
        w = self._writer(tmp_path)
        s = _state(config=_reward_cfg(("judge", "judge_score", 1.0)))
        assert w._compute_reward(
            s, {"id": "judge"}, {"output": {"overall_score": 15}}
        ) == pytest.approx(1.0)

    def test_judge_score_returns_none_when_no_score_field(self, tmp_path):
        w = self._writer(tmp_path)
        s = _state(config=_reward_cfg(("judge", "judge_score", 1.0)))
        assert w._compute_reward(s, {"id": "judge"}, {"output": {}}) is None

    # --- rule_pass_rate -------------------------------------------------------

    def test_rule_pass_rate_partial(self, tmp_path):
        w = self._writer(tmp_path)
        s = _state(config=_reward_cfg(("rules", "rule_pass_rate", 1.0)))
        r = w._compute_reward(s, {"id": "rules"}, {"output": {"rules_passed": 3, "rules_total": 4}})
        assert r == pytest.approx(0.75)

    def test_rule_pass_rate_all_pass(self, tmp_path):
        w = self._writer(tmp_path)
        s = _state(config=_reward_cfg(("rules", "rule_pass_rate", 1.0)))
        r = w._compute_reward(s, {"id": "rules"}, {"output": {"rules_passed": 5, "rules_total": 5}})
        assert r == pytest.approx(1.0)

    # --- iteration_penalty ----------------------------------------------------

    def test_iteration_penalty_first_iter_is_1(self, tmp_path):
        w = self._writer(tmp_path)
        s = _state(
            config=_reward_cfg(("gen", "iteration_penalty", 1.0)), stage_iterations={"gen": 1}
        )
        r = w._compute_reward(s, {"id": "gen", "max_iterations": 3}, {})
        assert r == pytest.approx(1.0)

    def test_iteration_penalty_second_iter_is_discounted(self, tmp_path):
        w = self._writer(tmp_path)
        s = _state(
            config=_reward_cfg(("gen", "iteration_penalty", 1.0)), stage_iterations={"gen": 2}
        )
        r = w._compute_reward(s, {"id": "gen", "max_iterations": 3}, {})
        assert r == pytest.approx(1.0 - 1 / 3)

    # --- reward written to audit entry ----------------------------------------

    def test_reward_field_present_in_log_stage_output(self, tmp_path):
        w = AuditWriter(str(tmp_path / "audit.jsonl"))
        s = _state(
            config=_reward_cfg(("judge", "judge_score", 1.0)),
            stage_outputs={"judge": {"overall_score": 9}},
        )
        w.log_stage(s, {"id": "judge", "type": "llm_judge"}, {"output": {"overall_score": 9}})
        entries = [json.loads(l) for l in open(str(tmp_path / "audit.jsonl"))]
        assert entries[0]["reward"] == pytest.approx(0.9)

    def test_reward_is_null_for_unconfigured_stage_in_audit(self, tmp_path):
        w = AuditWriter(str(tmp_path / "audit.jsonl"))
        s = _state(config={"reward": {"stages": {}}})
        w.log_stage(s, {"id": "extract", "type": "llm_call"}, {"output": {}})
        entries = [json.loads(l) for l in open(str(tmp_path / "audit.jsonl"))]
        assert entries[0]["reward"] is None


# ---------------------------------------------------------------------------
# 2.  AuditWriter._compute_episode_reward()
# ---------------------------------------------------------------------------


class TestComputeEpisodeReward:

    def test_weighted_mean_two_stages(self, tmp_path):
        w = AuditWriter(str(tmp_path / "audit.jsonl"))
        reward_cfg = {
            "stages": {
                "judge": {"fn": "judge_score", "weight": 1.0},
                "generate": {"fn": "iteration_penalty", "weight": 0.2},
            }
        }
        s = _state(
            config={"reward": reward_cfg},
            stage_outputs={"judge": {"overall_score": 8}},
            stage_iterations={"generate": 1},
            verdict="COMPLETE",
        )
        ep = w._compute_episode_reward(s, reward_cfg)
        # judge: 0.8 * 1.0; generate: 1.0 * 0.2; total_weight = 1.2
        assert ep == pytest.approx((0.8 * 1.0 + 1.0 * 0.2) / 1.2, abs=0.001)

    def test_returns_none_when_no_stage_yields_reward(self, tmp_path):
        w = AuditWriter(str(tmp_path / "audit.jsonl"))
        reward_cfg = {"stages": {"judge": {"fn": "judge_score", "weight": 1.0}}}
        s = _state(config={"reward": reward_cfg}, stage_outputs={"judge": {}})
        assert w._compute_episode_reward(s, reward_cfg) is None

    def test_episode_reward_written_to_completion_entry(self, tmp_path):
        w = AuditWriter(str(tmp_path / "audit.jsonl"))
        reward_cfg = {
            "stages": {"judge": {"fn": "judge_score", "weight": 1.0}},
            "final_fn": "weighted_mean",
        }
        s = _state(
            config={"reward": reward_cfg},
            stage_outputs={"judge": {"overall_score": 10}},
            verdict="COMPLETE",
        )
        s.timestamp_end = "2026-04-27T10:00:01"
        w.write(s)
        entries = [json.loads(l) for l in open(str(tmp_path / "audit.jsonl"))]
        assert entries[0]["episode_reward"] == pytest.approx(1.0)

    def test_episode_reward_null_when_final_fn_not_weighted_mean(self, tmp_path):
        w = AuditWriter(str(tmp_path / "audit.jsonl"))
        s = _state(
            config={"reward": {"stages": {}, "final_fn": "mean"}},
            stage_outputs={},
            verdict="COMPLETE",
        )
        s.timestamp_end = "2026-04-27T10:00:01"
        w.write(s)
        entries = [json.loads(l) for l in open(str(tmp_path / "audit.jsonl"))]
        assert entries[0]["episode_reward"] is None


# ---------------------------------------------------------------------------
# 3.  GraphRunner._validate_observation()
# ---------------------------------------------------------------------------


class TestValidateObservation:

    def _runner(self) -> GraphRunner:
        r = GraphRunner.__new__(GraphRunner)
        r.config = _minimal_graph_config()
        return r

    # string schema
    def test_string_schema_accepts_str(self):
        self._runner()._validate_observation("hello", {"type": "string"})

    def test_empty_schema_skips_validation(self):
        self._runner()._validate_observation(42, {})  # no exception

    # object schema
    def test_object_schema_accepts_valid_json_object(self):
        self._runner()._validate_observation(
            '{"name": "Alice"}', {"type": "object", "required": ["name"]}
        )

    def test_object_schema_rejects_plain_string(self):
        with pytest.raises(ValueError, match="valid JSON"):
            self._runner()._validate_observation("not json", {"type": "object"})

    def test_object_schema_rejects_missing_required_field(self):
        with pytest.raises(ValueError, match="required field 'dob' missing"):
            self._runner()._validate_observation(
                '{"name": "Alice"}', {"type": "object", "required": ["name", "dob"]}
            )

    def test_object_schema_rejects_json_array(self):
        with pytest.raises(ValueError, match="JSON object"):
            self._runner()._validate_observation("[1,2,3]", {"type": "object"})

    # array schema
    def test_array_schema_accepts_json_array(self):
        self._runner()._validate_observation("[1,2,3]", {"type": "array"})

    def test_array_schema_rejects_json_object(self):
        with pytest.raises(ValueError, match="JSON array"):
            self._runner()._validate_observation('{"k":"v"}', {"type": "array"})

    def test_array_schema_rejects_plain_string(self):
        with pytest.raises(ValueError, match="valid JSON"):
            self._runner()._validate_observation("not json", {"type": "array"})


# ---------------------------------------------------------------------------
# 4.  GraphRunner.run_episodes()
# ---------------------------------------------------------------------------


class TestRunEpisodes:

    def _runner_with_mock_run(self, side_effect=None) -> GraphRunner:
        r = GraphRunner.__new__(GraphRunner)
        r.config = _minimal_graph_config()
        r.run = AsyncMock(side_effect=side_effect or (lambda inp: _state(input_text=inp)))
        return r

    @pytest.mark.asyncio
    async def test_returns_n_states(self):
        r = self._runner_with_mock_run()
        states = await r.run_episodes(["a", "b", "c"], n_episodes=5)
        assert len(states) == 5

    @pytest.mark.asyncio
    async def test_cycles_inputs_when_fewer_than_n_episodes(self):
        seen = []

        async def capture(inp):
            seen.append(inp)
            return _state(input_text=inp)

        r = self._runner_with_mock_run()
        r.run = capture
        await r.run_episodes(["x", "y"], n_episodes=5)
        assert seen == ["x", "y", "x", "y", "x"]

    @pytest.mark.asyncio
    async def test_shuffle_with_same_seed_is_reproducible(self):
        inputs = ["a", "b", "c", "d", "e"]

        async def make_runner():
            seen = []
            r = GraphRunner.__new__(GraphRunner)
            r.config = _minimal_graph_config()

            async def capture(inp):
                seen.append(inp)
                return _state(input_text=inp)

            r.run = capture
            return r, seen

        r1, seen1 = await make_runner()
        await r1.run_episodes(inputs, n_episodes=5, shuffle=True, seed=42)

        r2, seen2 = await make_runner()
        await r2.run_episodes(inputs, n_episodes=5, shuffle=True, seed=42)

        assert seen1 == seen2

    @pytest.mark.asyncio
    async def test_different_seeds_produce_different_orders(self):
        inputs = ["a", "b", "c", "d", "e"]

        async def make_runner():
            seen = []
            r = GraphRunner.__new__(GraphRunner)
            r.config = _minimal_graph_config()

            async def capture(inp):
                seen.append(inp)
                return _state(input_text=inp)

            r.run = capture
            return r, seen

        r1, seen1 = await make_runner()
        await r1.run_episodes(inputs, n_episodes=5, shuffle=True, seed=1)

        r2, seen2 = await make_runner()
        await r2.run_episodes(inputs, n_episodes=5, shuffle=True, seed=99)

        # Very unlikely to be identical with different seeds
        assert seen1 != seen2

    @pytest.mark.asyncio
    async def test_batch_size_runs_all_episodes(self):
        r = GraphRunner.__new__(GraphRunner)
        r.config = _minimal_graph_config()
        r.run = AsyncMock(return_value=_state())
        states = await r.run_episodes(["a", "b", "c"], n_episodes=6, batch_size=3)
        assert len(states) == 6
        assert r.run.call_count == 6


# ---------------------------------------------------------------------------
# 5.  CollectRunner._extract_record()
# ---------------------------------------------------------------------------


class TestCollectExtractRecord:

    def _cr(self, algorithm="bco") -> CollectRunner:
        mock_runner = MagicMock(spec=GraphRunner)
        mock_runner.config = {
            "id": "test/collect",
            "name": "T",
            "version": "1.0.0",
            "collect": {
                "algorithm": algorithm,
                "stage_id": "judge",
                "min_samples": 1,
                "output_path": "/tmp/x.jsonl",
            },
            "episode": {"n_episodes": 1, "batch_size": 1},
            "reward": {"stages": {"judge": {"fn": "judge_score", "weight": 1.0}}},
        }
        return CollectRunner(mock_runner)

    # BCO
    def test_bco_pass_label_is_1(self):
        rec = self._cr("bco")._extract_record(_state(verdict="APPROVE"), "judge", "bco")
        assert rec["label"] == 1

    def test_bco_fail_label_is_0(self):
        rec = self._cr("bco")._extract_record(_state(verdict="DENY"), "judge", "bco")
        assert rec["label"] == 0

    def test_bco_complete_is_pass(self):
        rec = self._cr("bco")._extract_record(_state(verdict="COMPLETE"), "judge", "bco")
        assert rec["label"] == 1

    def test_bco_record_has_input_and_pipeline_id(self):
        rec = self._cr("bco")._extract_record(_state(verdict="APPROVE"), "judge", "bco")
        assert "input" in rec
        assert "pipeline_id" in rec

    # DPO
    def test_dpo_pass_puts_output_in_chosen(self):
        s = _state(verdict="APPROVE", stage_outputs={"judge": {"score": 9}})
        rec = self._cr("dpo")._extract_record(s, "judge", "dpo")
        assert rec["chosen_output"] == {"score": 9}
        assert rec["rejected_output"] == {}

    def test_dpo_fail_puts_output_in_rejected(self):
        s = _state(verdict="DENY", stage_outputs={"judge": {"score": 2}})
        rec = self._cr("dpo")._extract_record(s, "judge", "dpo")
        assert rec["rejected_output"] == {"score": 2}
        assert rec["chosen_output"] == {}

    # GRPO
    def test_grpo_record_includes_reward(self):
        s = _state(
            verdict="APPROVE",
            stage_outputs={"judge": {"overall_score": 8}},
            config={"reward": {"stages": {"judge": {"fn": "judge_score", "weight": 1.0}}}},
        )
        rec = self._cr("grpo")._extract_record(s, "judge", "grpo")
        assert "reward" in rec
        assert rec["reward"] == pytest.approx(0.8)

    def test_grpo_binary_fallback_when_no_judge_score(self):
        s = _state(verdict="APPROVE", stage_outputs={"judge": {}})
        rec = self._cr("grpo")._extract_record(s, "judge", "grpo")
        assert rec["reward"] == pytest.approx(1.0)

    # Unknown algorithm
    def test_unknown_algorithm_returns_none(self):
        rec = self._cr()._extract_record(_state(), "judge", "unknown_algo")
        assert rec is None


# ---------------------------------------------------------------------------
# 6.  CollectRunner.run() — episode loop
# ---------------------------------------------------------------------------


class TestCollectRunnerRun:

    def _mock_runner(
        self,
        tmp_path,
        algorithm="bco",
        min_samples=3,
        n_episodes=3,
        auto_train=False,
        trainer_config=None,
    ):
        output_path = str(tmp_path / "out.jsonl")
        mock_runner = MagicMock(spec=GraphRunner)
        mock_runner.config = {
            "id": "test/collect",
            "name": "T",
            "version": "1.0.0",
            "collect": {
                "algorithm": algorithm,
                "stage_id": None,
                "min_samples": min_samples,
                "output_path": output_path,
                "auto_train": auto_train,
            },
            "episode": {"n_episodes": n_episodes, "batch_size": 1, "shuffle_inputs": False},
            "training": {"trainer_config": trainer_config},
        }
        mock_runner.run_episodes = AsyncMock(
            return_value=[
                _state(pipeline_id=f"p{i}", verdict="APPROVE" if i % 2 == 0 else "DENY")
                for i in range(n_episodes)
            ]
        )
        return mock_runner, output_path

    @pytest.mark.asyncio
    async def test_writes_bco_jsonl(self, tmp_path):
        mock_runner, output_path = self._mock_runner(tmp_path)
        n = await CollectRunner(mock_runner).run(["a", "b", "c"])
        assert n == 3
        records = [json.loads(l) for l in open(output_path)]
        assert all("input" in r and "label" in r for r in records)
        assert all(r["label"] in (0, 1) for r in records)

    @pytest.mark.asyncio
    async def test_caps_output_at_min_samples(self, tmp_path):
        mock_runner, output_path = self._mock_runner(tmp_path, min_samples=2, n_episodes=10)
        mock_runner.run_episodes = AsyncMock(
            return_value=[_state(pipeline_id=f"p{i}") for i in range(10)]
        )
        n = await CollectRunner(mock_runner).run(["x"])
        assert n == 2
        records = [json.loads(l) for l in open(output_path)]
        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_auto_train_calls_bridge_when_configured(self, tmp_path):
        mock_runner, _ = self._mock_runner(
            tmp_path, auto_train=True, trainer_config="/fake/trainer.yaml"
        )
        mock_trainer = MagicMock()
        mock_bridge = MagicMock()
        mock_bridge.build_trainer.return_value = mock_trainer

        with patch("agenttune.decide.collect_runner.TrainerConfigBridge", return_value=mock_bridge):
            await CollectRunner(mock_runner).run(["x"])

        mock_bridge.build_trainer.assert_called_once()
        mock_trainer.train.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_train_skipped_when_no_trainer_config(self, tmp_path):
        mock_runner, _ = self._mock_runner(tmp_path, auto_train=True, trainer_config=None)

        with patch("agenttune.decide.collect_runner.TrainerConfigBridge") as mock_cls:
            await CollectRunner(mock_runner).run(["x"])

        mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_train_skipped_when_auto_train_false(self, tmp_path):
        mock_runner, _ = self._mock_runner(
            tmp_path, auto_train=False, trainer_config="/fake/trainer.yaml"
        )

        with patch("agenttune.decide.collect_runner.TrainerConfigBridge") as mock_cls:
            await CollectRunner(mock_runner).run(["x"])

        mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# 7.  EvalRunner — metrics and thresholds
# ---------------------------------------------------------------------------


class TestEvalRunner:

    def _er(self, tmp_path, thresholds=None) -> EvalRunner:
        mock_runner = MagicMock(spec=GraphRunner)
        mock_runner.config = {
            "id": "test/eval",
            "name": "T",
            "version": "1.0.0",
            "audit": {"path": str(tmp_path / "audit.jsonl")},
            "reward": {"stages": {"judge": {"fn": "judge_score", "weight": 1.0}}},
            "eval": {"thresholds": thresholds or {}},
        }
        return EvalRunner(mock_runner)

    def _test_set(self, tmp_path, records) -> str:
        path = str(tmp_path / "test_set.jsonl")
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    def _append_audit(self, tmp_path, pipeline_id, cost_usd=None, reward=None):
        path = str(tmp_path / "audit.jsonl")
        entry = {"pipeline_id": pipeline_id, "stage_id": "judge"}
        if cost_usd is not None:
            entry["cost_usd"] = cost_usd
        if reward is not None:
            entry["reward"] = reward
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # accuracy ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_accuracy_all_correct(self, tmp_path):
        er = self._er(tmp_path)
        ts = self._test_set(
            tmp_path,
            [
                {"input": "a", "expected_verdict": "APPROVE"},
                {"input": "b", "expected_verdict": "DENY"},
            ],
        )
        er.runner.run = AsyncMock(
            side_effect=[
                _state(pipeline_id="p1", verdict="APPROVE"),
                _state(pipeline_id="p2", verdict="DENY"),
            ]
        )
        report = await er.run(ts)
        assert report["metrics"]["accuracy"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_accuracy_partial(self, tmp_path):
        er = self._er(tmp_path)
        ts = self._test_set(
            tmp_path,
            [
                {"input": "a", "expected_verdict": "APPROVE"},
                {"input": "b", "expected_verdict": "APPROVE"},
                {"input": "c", "expected_verdict": "APPROVE"},
            ],
        )
        er.runner.run = AsyncMock(
            side_effect=[
                _state(pipeline_id="p1", verdict="APPROVE"),
                _state(pipeline_id="p2", verdict="DENY"),  # wrong
                _state(pipeline_id="p3", verdict="APPROVE"),
            ]
        )
        report = await er.run(ts)
        assert report["metrics"]["accuracy"] == pytest.approx(2 / 3, abs=0.01)

    @pytest.mark.asyncio
    async def test_accuracy_case_insensitive(self, tmp_path):
        er = self._er(tmp_path)
        ts = self._test_set(tmp_path, [{"input": "a", "expected_verdict": "approve"}])
        er.runner.run = AsyncMock(return_value=_state(pipeline_id="p1", verdict="APPROVE"))
        report = await er.run(ts)
        assert report["metrics"]["accuracy"] == pytest.approx(1.0)

    # latency -----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_latency_percentiles_present(self, tmp_path):
        er = self._er(tmp_path)
        ts = self._test_set(tmp_path, [{"input": str(i)} for i in range(4)])
        er.runner.run = AsyncMock(
            side_effect=[
                _state(pipeline_id=f"p{i}", elapsed_seconds=float(i + 1)) for i in range(4)
            ]
        )
        report = await er.run(ts)
        assert "latency_p50" in report["metrics"]
        assert "latency_p95" in report["metrics"]

    # cost --------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cost_per_decision_mean_of_pipeline_costs(self, tmp_path):
        er = self._er(tmp_path)
        ts = self._test_set(tmp_path, [{"input": "a"}, {"input": "b"}])
        self._append_audit(tmp_path, "p1", cost_usd=0.02)
        self._append_audit(tmp_path, "p2", cost_usd=0.04)
        er.runner.run = AsyncMock(
            side_effect=[
                _state(pipeline_id="p1"),
                _state(pipeline_id="p2"),
            ]
        )
        report = await er.run(ts)
        assert report["metrics"]["cost_per_decision"] == pytest.approx(0.03, abs=0.001)

    # judge score mean --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_judge_score_mean_from_audit(self, tmp_path):
        er = self._er(tmp_path)
        ts = self._test_set(tmp_path, [{"input": "a"}, {"input": "b"}])
        self._append_audit(tmp_path, "p1", reward=0.6)
        self._append_audit(tmp_path, "p2", reward=0.8)
        er.runner.run = AsyncMock(
            side_effect=[
                _state(pipeline_id="p1"),
                _state(pipeline_id="p2"),
            ]
        )
        report = await er.run(ts)
        assert report["metrics"]["judge_score_mean"] == pytest.approx(0.7, abs=0.01)

    # threshold violations ----------------------------------------------------

    def test_lower_is_better_latency_violation(self, tmp_path):
        er = self._er(tmp_path, thresholds={"latency_p95": 1000})
        violations = er._check_thresholds({"latency_p95": 2000})
        assert any("latency_p95" in v for v in violations)

    def test_lower_is_better_latency_passes(self, tmp_path):
        er = self._er(tmp_path, thresholds={"latency_p95": 5000})
        assert er._check_thresholds({"latency_p95": 1000}) == []

    def test_higher_is_better_accuracy_violation(self, tmp_path):
        er = self._er(tmp_path, thresholds={"accuracy": 0.9})
        violations = er._check_thresholds({"accuracy": 0.5})
        assert any("accuracy" in v for v in violations)

    def test_higher_is_better_accuracy_passes(self, tmp_path):
        er = self._er(tmp_path, thresholds={"accuracy": 0.8})
        assert er._check_thresholds({"accuracy": 0.95}) == []

    def test_metric_absent_from_results_is_skipped(self, tmp_path):
        er = self._er(tmp_path, thresholds={"accuracy": 0.9})
        # accuracy not in metrics → no violation raised
        assert er._check_thresholds({"latency_p95": 100}) == []

    @pytest.mark.asyncio
    async def test_full_run_violation_sets_passed_false(self, tmp_path):
        er = self._er(tmp_path, thresholds={"accuracy": 0.9})
        ts = self._test_set(
            tmp_path,
            [
                {"input": "a", "expected_verdict": "APPROVE"},
                {"input": "b", "expected_verdict": "APPROVE"},
            ],
        )
        er.runner.run = AsyncMock(
            side_effect=[
                _state(pipeline_id="p1", verdict="APPROVE"),
                _state(pipeline_id="p2", verdict="DENY"),  # wrong → accuracy 0.5
            ]
        )
        report = await er.run(ts)
        assert not report["passed"]
        assert report["violations"]

    @pytest.mark.asyncio
    async def test_full_run_all_pass(self, tmp_path):
        er = self._er(tmp_path, thresholds={"accuracy": 0.5})
        ts = self._test_set(tmp_path, [{"input": "a", "expected_verdict": "APPROVE"}])
        er.runner.run = AsyncMock(return_value=_state(pipeline_id="p1", verdict="APPROVE"))
        report = await er.run(ts)
        assert report["passed"]
        assert report["violations"] == []

    @pytest.mark.asyncio
    async def test_empty_test_set_raises_value_error(self, tmp_path):
        er = self._er(tmp_path)
        ts = self._test_set(tmp_path, [])
        with pytest.raises(ValueError, match="No records"):
            await er.run(ts)

    @pytest.mark.asyncio
    async def test_n_evaluated_matches_test_set_size(self, tmp_path):
        er = self._er(tmp_path)
        ts = self._test_set(tmp_path, [{"input": str(i)} for i in range(4)])
        er.runner.run = AsyncMock(side_effect=[_state(pipeline_id=f"p{i}") for i in range(4)])
        report = await er.run(ts)
        assert report["n_evaluated"] == 4


# ---------------------------------------------------------------------------
# 8.  CLI --mode collect / eval / train
# ---------------------------------------------------------------------------


class TestCLIModes:

    def _base_cfg(self, tmp_path, extra=None):
        cfg = {
            "id": "test/cli_modes",
            "name": "T",
            "version": "1.0.0",
            "stages": [],
            "run_mode": "inference",
            "audit": {"path": str(tmp_path / "audit.jsonl")},
            "destinations": {"file": {"enabled": True, "path": str(tmp_path / "decisions.jsonl")}},
            "collect": {
                "algorithm": "bco",
                "stage_id": None,
                "min_samples": 3,
                "output_path": str(tmp_path / "out.jsonl"),
                "auto_train": False,
            },
            "episode": {"n_episodes": 3, "batch_size": 1},
            "eval": {"test_set": None, "thresholds": {}},
            "training": {},
        }
        if extra:
            cfg.update(extra)
        return cfg

    # --mode collect ----------------------------------------------------------

    @patch("agenttune.decide.collect_runner.CollectRunner.run")
    @patch("agenttune.decide.graph_runner.GraphRunner.from_template")
    def test_mode_collect_exits_0(self, mock_ft, mock_cr_run, tmp_path):
        mock_runner = MagicMock()
        mock_runner.config = self._base_cfg(tmp_path)
        mock_ft.return_value = mock_runner

        async def _run(inputs):
            return 3

        mock_cr_run.side_effect = _run
        result = cli_runner.invoke(
            decide_app,
            ["run", "--template", "test/cli", "--input", "hello", "--mode", "collect"],
        )
        assert result.exit_code == 0

    @patch("agenttune.decide.collect_runner.CollectRunner.run")
    @patch("agenttune.decide.graph_runner.GraphRunner.from_template")
    def test_mode_collect_reports_records_written(self, mock_ft, mock_cr_run, tmp_path):
        mock_runner = MagicMock()
        mock_runner.config = self._base_cfg(tmp_path)
        mock_ft.return_value = mock_runner

        async def _run(inputs):
            return 7

        mock_cr_run.side_effect = _run
        result = cli_runner.invoke(
            decide_app,
            ["run", "--template", "test/cli", "--input", "hello", "--mode", "collect"],
        )
        assert "7" in result.output

    # --mode eval -------------------------------------------------------------

    @patch("agenttune.decide.eval_runner.EvalRunner.run")
    @patch("agenttune.decide.graph_runner.GraphRunner.from_template")
    def test_mode_eval_exits_0_all_pass(self, mock_ft, mock_er_run, tmp_path):
        ts = tmp_path / "ts.jsonl"
        ts.write_text('{"input":"x","expected_verdict":"APPROVE"}\n')
        mock_runner = MagicMock()
        cfg = self._base_cfg(tmp_path)
        cfg["eval"]["test_set"] = str(ts)
        mock_runner.config = cfg
        mock_ft.return_value = mock_runner

        async def _run(path):
            return {
                "metrics": {"accuracy": 1.0},
                "passed": True,
                "violations": [],
                "n_evaluated": 1,
            }

        mock_er_run.side_effect = _run
        result = cli_runner.invoke(
            decide_app,
            ["run", "--template", "test/cli", "--input", "x", "--mode", "eval"],
        )
        assert result.exit_code == 0

    @patch("agenttune.decide.eval_runner.EvalRunner.run")
    @patch("agenttune.decide.graph_runner.GraphRunner.from_template")
    def test_mode_eval_exits_1_on_violation(self, mock_ft, mock_er_run, tmp_path):
        ts = tmp_path / "ts.jsonl"
        ts.write_text('{"input":"x","expected_verdict":"APPROVE"}\n')
        mock_runner = MagicMock()
        cfg = self._base_cfg(tmp_path)
        cfg["eval"]["test_set"] = str(ts)
        mock_runner.config = cfg
        mock_ft.return_value = mock_runner

        async def _run(path):
            return {
                "metrics": {"accuracy": 0.4},
                "passed": False,
                "violations": ["accuracy: 0.4 below min threshold 0.9"],
                "n_evaluated": 5,
            }

        mock_er_run.side_effect = _run
        result = cli_runner.invoke(
            decide_app,
            ["run", "--template", "test/cli", "--input", "x", "--mode", "eval"],
        )
        assert result.exit_code == 1

    @patch("agenttune.decide.graph_runner.GraphRunner.from_template")
    def test_mode_eval_exits_1_when_test_set_not_configured(self, mock_ft, tmp_path):
        mock_runner = MagicMock()
        cfg = self._base_cfg(tmp_path)
        cfg["eval"]["test_set"] = None
        mock_runner.config = cfg
        mock_ft.return_value = mock_runner
        result = cli_runner.invoke(
            decide_app,
            ["run", "--template", "test/cli", "--input", "x", "--mode", "eval"],
        )
        assert result.exit_code == 1
        assert "test_set" in result.output.lower() or "not configured" in result.output.lower()

    # --mode train ------------------------------------------------------------

    @patch("agenttune.decide.collect_runner.CollectRunner.run")
    @patch("agenttune.decide.graph_runner.GraphRunner.from_template")
    def test_mode_train_collects_and_warns_when_no_trainer_config(
        self, mock_ft, mock_cr_run, tmp_path
    ):
        mock_runner = MagicMock()
        mock_runner.config = self._base_cfg(tmp_path)  # training.trainer_config = None
        mock_ft.return_value = mock_runner

        async def _run(inputs):
            return 5

        mock_cr_run.side_effect = _run
        result = cli_runner.invoke(
            decide_app,
            ["run", "--template", "test/cli", "--input", "x", "--mode", "train"],
        )
        assert result.exit_code == 0
        assert "trainer_config" in result.output.lower() or "skipping" in result.output.lower()

    # unknown mode ------------------------------------------------------------

    @patch("agenttune.decide.graph_runner.GraphRunner.from_template")
    def test_unknown_mode_exits_1(self, mock_ft, tmp_path):
        mock_runner = MagicMock()
        mock_runner.config = self._base_cfg(tmp_path)
        mock_ft.return_value = mock_runner
        result = cli_runner.invoke(
            decide_app,
            ["run", "--template", "test/cli", "--input", "x", "--mode", "bogus_mode"],
        )
        assert result.exit_code == 1
