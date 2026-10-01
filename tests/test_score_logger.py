"""Verify the score/reason logging added in commit 45afe36 actually works.

Three layers:
  1. `ScoreLogger` itself — does `.log(...)` write a well-formed JSONL record
     (score + reasons + components + meta) to disk, and does `.get_last()`
     return it back.
  2. Every reward *function* across every reward module that imports
     `log_score` (agentic/rewards/builtin_rewards/*, rag/rewards/*,
     rag/memory/m2_decisions.py, rag/datagen.py, decide/rewards.py) — call
     each one for real and check it produced a matching JSONL line with a
     non-empty `reasons` trail, proving the wiring (not just the logger in
     isolation) works end to end.
  3. A JSONL trail file under `data/` (see `test_scores_jsonl_trail_is_human_checkable`)
     so the reasons can be eyeballed directly, not just asserted on.
"""

import json

import pytest

from agenttune.utils.score_logger import ScoreLogger, get_score_logger, log_score

# ---------------------------------------------------------------------------
# 1. ScoreLogger unit tests
# ---------------------------------------------------------------------------


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_log_writes_jsonl_record(tmp_path):
    out = tmp_path / "scores.jsonl"
    logger = ScoreLogger(output_file=str(out))

    logger.log("demo.reward", 0.75, reasons=["condition A -> +0.5", "condition B -> +0.25"])

    records = _read_jsonl(out)
    assert len(records) == 1
    record = records[0]
    assert record["name"] == "demo.reward"
    assert record["score"] == 0.75
    assert record["reasons"] == ["condition A -> +0.5", "condition B -> +0.25"]
    assert "ts" in record and record["ts"]


def test_log_defaults_reasons_components_meta_to_empty(tmp_path):
    out = tmp_path / "scores.jsonl"
    logger = ScoreLogger(output_file=str(out))

    logger.log("demo.reward", 1.0)

    record = _read_jsonl(out)[0]
    assert record["reasons"] == []
    assert record["components"] == {}
    assert record["meta"] == {}


def test_log_appends_multiple_calls(tmp_path):
    out = tmp_path / "scores.jsonl"
    logger = ScoreLogger(output_file=str(out))

    for i in range(5):
        logger.log("demo.reward", float(i), reasons=[f"call {i}"], meta={"index": i})

    records = _read_jsonl(out)
    assert len(records) == 5
    assert [r["meta"]["index"] for r in records] == [0, 1, 2, 3, 4]


def test_log_creates_parent_directories(tmp_path):
    out = tmp_path / "nested" / "dir" / "scores.jsonl"
    logger = ScoreLogger(output_file=str(out))

    logger.log("demo.reward", 0.5, reasons=["nested path works"])

    assert out.exists()
    assert _read_jsonl(out)[0]["score"] == 0.5


def test_get_last_returns_most_recent_record_per_name(tmp_path):
    logger = ScoreLogger(output_file=str(tmp_path / "scores.jsonl"))

    logger.log("reward.a", 0.1, reasons=["first"])
    logger.log("reward.b", 0.9, reasons=["other reward"])
    logger.log("reward.a", 0.2, reasons=["second"])

    last_a = logger.get_last("reward.a")
    assert last_a["score"] == 0.2
    assert last_a["reasons"] == ["second"]

    last_b = logger.get_last("reward.b")
    assert last_b["score"] == 0.9


def test_get_last_unknown_name_returns_none(tmp_path):
    logger = ScoreLogger(output_file=str(tmp_path / "scores.jsonl"))
    assert logger.get_last("never_logged") is None


def test_log_forwards_to_logging_manager(tmp_path):
    calls = []

    class FakeLoggingManager:
        def log_metrics(self, metrics, step=None):
            calls.append((metrics, step))

    logger = ScoreLogger(
        output_file=str(tmp_path / "scores.jsonl"), logging_manager=FakeLoggingManager()
    )
    logger.log("demo.reward", 0.42, reasons=["r"], step=7)

    assert calls == [({"demo.reward": 0.42}, 7)]


def test_log_survives_broken_logging_manager(tmp_path):
    """A misbehaving LoggingManager must not stop the JSONL trail from being written."""

    class BrokenLoggingManager:
        def log_metrics(self, metrics, step=None):
            raise RuntimeError("boom")

    out = tmp_path / "scores.jsonl"
    logger = ScoreLogger(output_file=str(out), logging_manager=BrokenLoggingManager())

    logger.log("demo.reward", 1.0, reasons=["still written"])

    assert _read_jsonl(out)[0]["score"] == 1.0


def test_log_survives_unwritable_output_path(tmp_path):
    """If the file write fails, log() must not raise (callers are reward fns mid-training)."""
    # output_file points at a directory, so open(..., "a") is guaranteed to fail
    # with IsADirectoryError regardless of platform/permissions.
    logger = ScoreLogger(output_file=str(tmp_path))

    logger.log("demo.reward", 1.0, reasons=["should not raise"])  # must not raise
    assert logger.get_last("demo.reward")["score"] == 1.0


def test_get_score_logger_returns_singleton():
    assert get_score_logger() is get_score_logger()


def test_module_level_log_score_uses_default_logger(monkeypatch, tmp_path):
    out = tmp_path / "scores.jsonl"
    monkeypatch.setattr(get_score_logger(), "output_file", str(out))

    log_score("demo.module_level", 0.3, reasons=["via module-level helper"])

    record = _read_jsonl(out)[0]
    assert record["name"] == "demo.module_level"
    assert record["score"] == 0.3


# ---------------------------------------------------------------------------
# 2. Smoke tests: every real reward function that calls log_score
# ---------------------------------------------------------------------------


@pytest.fixture
def score_log_file(monkeypatch, tmp_path):
    """Redirect the process-wide default ScoreLogger to a temp file for the
    duration of one test, and hand back a reader for it."""
    out = tmp_path / "scores.jsonl"
    monkeypatch.setattr(get_score_logger(), "output_file", str(out))

    def read():
        if not out.exists():
            return []
        return _read_jsonl(out)

    return read


def _assert_logged(records, name):
    matches = [r for r in records if r["name"] == name]
    assert (
        matches
    ), f"expected at least one log_score(...) call for {name!r}, got names: {[r['name'] for r in records]}"
    for r in matches:
        assert isinstance(r["score"], int | float)
        assert isinstance(r["reasons"], list) and all(isinstance(x, str) for x in r["reasons"])
        assert r["reasons"], f"{name!r} logged with an empty reasons list -> not explaining WHY"
    return matches


# --- agentic/rewards/builtin_rewards/sql.py -------------------------------


def test_sql_correctness_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.sql import correctness_reward

    correctness_reward(completions=["*yes*"], answer=["yes"])
    _assert_logged(score_log_file(), "sql.correctness_reward")


def test_sql_structure_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.sql import structure_reward

    completion = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "sql"}}]},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "final answer"},
    ]
    structure_reward(completions=[completion])
    _assert_logged(score_log_file(), "sql.structure_reward")


def test_sql_query_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.sql import query_reward

    completion = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"arguments": {"sql_command": "select * from t where x=1"}}}
            ],
        },
        {"role": "tool", "content": [{"id": 1}]},
    ]
    query_reward(completions=[completion], answer=["yes"])
    _assert_logged(score_log_file(), "sql.query_reward")


def test_sql_reward_correct_answer(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.sql import reward_correct_answer

    reward_correct_answer(completions=["the answer is 42"], answer=["42"])
    _assert_logged(score_log_file(), "sql.reward_correct_answer")


def test_sql_reward_tool_used(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.sql import reward_tool_used

    reward_tool_used(completions=[[{"role": "tool", "content": "x"}]])
    _assert_logged(score_log_file(), "sql.reward_tool_used")


def test_sql_reward_concise_answer(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.sql import reward_concise_answer

    reward_concise_answer(completions=["short reply"])
    _assert_logged(score_log_file(), "sql.reward_concise_answer")


def test_sql_format_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.sql import format_reward

    completion = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "sql"}}]},
        {"role": "tool", "content": "result"},
    ]
    format_reward(completions=[completion])
    _assert_logged(score_log_file(), "sql.format_reward")


# --- agentic/rewards/builtin_rewards/finqa.py ------------------------------


def test_finqa_format_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.finqa import format_reward

    format_reward(prompts=[""], completions=["<answer>42</answer>"])
    _assert_logged(score_log_file(), "finqa.format_reward")


def test_finqa_sql_grounding_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.finqa import sql_grounding_reward

    sql_grounding_reward(prompts=[""], completions=["called query_finqa_tables once"])
    _assert_logged(score_log_file(), "finqa.sql_grounding_reward")


def test_finqa_calculator_grounding_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.finqa import calculator_grounding_reward

    calculator_grounding_reward(prompts=[""], completions=["used the calculator tool"])
    _assert_logged(score_log_file(), "finqa.calculator_grounding_reward")


def test_finqa_placeholder_coverage_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.finqa import placeholder_coverage_reward

    placeholder_coverage_reward(
        prompts=[""], completions=["<answer>revenue was 2.5B</answer>"], answer=["2.5B"]
    )
    _assert_logged(score_log_file(), "finqa.placeholder_coverage_reward")


def test_finqa_answer_correctness_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.finqa import answer_correctness_reward

    answer_correctness_reward(prompts=[""], completions=["<answer>2.5B</answer>"], answer=["2.5B"])
    _assert_logged(score_log_file(), "finqa.answer_correctness_reward")


# --- agentic/rewards/builtin_rewards/use_case.py ---------------------------


def test_use_case_search_grounding_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.use_case import search_grounding_reward

    search_grounding_reward(prompts=[""], completions=["x"], tool_call_counts=[1])
    _assert_logged(score_log_file(), "use_case.search_grounding_reward")


def test_use_case_message_id_citation_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.use_case import message_id_citation_reward

    message_id_citation_reward(prompts=[""], completions=["see message_id 123"])
    _assert_logged(score_log_file(), "use_case.message_id_citation_reward")


def test_use_case_format_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.use_case import format_reward

    format_reward(prompts=[""], completions=["<answer>done</answer>"])
    _assert_logged(score_log_file(), "use_case.format_reward")


def test_use_case_answer_format_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.use_case import answer_format_reward

    answer_format_reward(prompts=[""], completions=["<answer>a long enough answer</answer>"])
    _assert_logged(score_log_file(), "use_case.answer_format_reward")


def test_use_case_answer_correctness_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.use_case import answer_correctness_reward

    answer_correctness_reward(
        prompts=[""], completions=["<answer>paris</answer>"], answer=["paris"]
    )
    _assert_logged(score_log_file(), "use_case.answer_correctness_reward")


def test_use_case_placeholder_coverage_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.use_case import placeholder_coverage_reward

    placeholder_coverage_reward(
        prompts=[""], completions=["<answer>2.5B</answer>"], answer=["2.5B"]
    )
    _assert_logged(score_log_file(), "use_case.placeholder_coverage_reward")


def test_use_case_template_structure_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.use_case import template_structure_reward

    template_structure_reward(
        prompts=[""],
        completions=["<answer>**Summary of findings** here</answer>"],
        answer=["**Summary of findings** goes here in gold"],
    )
    _assert_logged(score_log_file(), "use_case.template_structure_reward")


def test_use_case_computation_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.use_case import computation_reward

    computation_reward(prompts=[""], completions=["x"], tool_call_counts=[2])
    _assert_logged(score_log_file(), "use_case.computation_reward")


def test_use_case_numerical_match_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.use_case import numerical_match_reward

    numerical_match_reward(prompts=[""], completions=["<answer>42</answer>"], answer=["42"])
    _assert_logged(score_log_file(), "use_case.numerical_match_reward")


def test_use_case_exploration_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.use_case import exploration_reward

    exploration_reward(prompts=[""], completions=["x"], tool_call_counts=[3])
    _assert_logged(score_log_file(), "use_case.exploration_reward")


def test_use_case_tool_chain_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.use_case import tool_chain_reward

    tool_chain_reward(prompts=[""], completions=["list_dir then read_file then run_python"])
    _assert_logged(score_log_file(), "use_case.tool_chain_reward")


def test_use_case_summary_written_reward(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.use_case import summary_written_reward

    summary_written_reward(prompts=[""], completions=["x"], sample_dir=[None])
    _assert_logged(score_log_file(), "use_case.summary_written_reward")


# --- agentic/rewards/builtin_rewards/distilled_judge.py --------------------


def test_distilled_judge_dummy_mode(score_log_file):
    from agenttune.agentic.rewards.builtin_rewards.distilled_judge import DistilledJudge

    judge = DistilledJudge(model_path="this-model-does-not-exist")
    judge.pipeline = None  # force the "not loaded" fallback path deterministically
    judge(prompts=["hello"], completions=["world"])

    _assert_logged(score_log_file(), "distilled_judge_reward")


# --- rag/rewards/t3_rewards.py ----------------------------------------------


def test_t3_necessity_reward(score_log_file):
    from agenttune.rag.rewards.t3_rewards import necessity_reward

    necessity_reward(
        prompts=None,
        completions=["<answer>Paris</answer>"],
        tool_call_counts=[1],
        gold_answer=["Paris"],
        requires_search=[True],
        max_searches=6,
    )

    _assert_logged(score_log_file(), "necessity_reward")


def test_t3_frugality_reward(score_log_file):
    from agenttune.rag.rewards.t3_rewards import frugality_reward

    frugality_reward(
        prompts=None,
        completions=["<answer>Paris</answer>"],
        tool_call_counts=[2],
        gold_answer=["Paris"],
        requires_search=[True],
        optimal_search_count=[2],
    )

    _assert_logged(score_log_file(), "frugality_reward")


# --- rag/rewards/phase1_rewards.py ------------------------------------------


def test_phase1_termination_reward(score_log_file):
    from agenttune.rag.rewards.phase1_rewards import termination_reward

    termination_reward(prompts=None, completions=["<answer>Paris</answer>"], tool_call_counts=[1])
    _assert_logged(score_log_file(), "termination_reward")


def test_phase1_rag_correctness_reward(score_log_file):
    from agenttune.rag.rewards.phase1_rewards import rag_correctness_reward

    rag_correctness_reward(
        prompts=None, completions=["<answer>Paris is the capital</answer>"], gold_answer=["Paris"]
    )
    _assert_logged(score_log_file(), "rag_correctness_reward")


# --- rag/rewards/finder_rewards.py ------------------------------------------


def test_finder_numeric_correctness_reward(score_log_file):
    from agenttune.rag.rewards.finder_rewards import numeric_correctness_reward

    numeric_correctness_reward(
        prompts=None, completions=["<answer>2.5 billion</answer>"], gold_answer=["2.5B"]
    )
    _assert_logged(score_log_file(), "numeric_correctness_reward")


def test_finder_golden_chunk_recall_reward(score_log_file):
    from agenttune.rag.rewards.finder_rewards import golden_chunk_recall_reward

    golden_chunk_recall_reward(
        prompts=None,
        completions=["<answer>x</answer>"],
        retrieved_chunk_ids=[["doc1::0", "doc1::1"]],
        gold_chunk_ids=[["doc1::0"]],
    )
    _assert_logged(score_log_file(), "golden_chunk_recall_reward")


def test_finder_conciseness_reward(score_log_file):
    from agenttune.rag.rewards.finder_rewards import conciseness_reward

    conciseness_reward(prompts=None, completions=["<answer>a short answer</answer>"])
    _assert_logged(score_log_file(), "conciseness_reward")


# --- rag/memory/m2_decisions.py ---------------------------------------------


def test_m2_decision_reward(score_log_file):
    from agenttune.rag.memory.m2_decisions import decision_reward

    decision_reward(prompts=None, completions=["no decision token here"], tool_call_counts=[0])
    _assert_logged(score_log_file(), "decision_reward")


# --- rag/datagen.py ----------------------------------------------------------


def test_datagen_answer_correctness_reward_f1(score_log_file):
    from agenttune.rag.datagen import answer_correctness_reward

    answer_correctness_reward("Paris is the capital", "Paris", mode="f1")
    _assert_logged(score_log_file(), "datagen.answer_correctness_reward")


def test_datagen_answer_correctness_reward_em(score_log_file):
    from agenttune.rag.datagen import answer_correctness_reward

    answer_correctness_reward("Paris", "Paris", mode="em")
    _assert_logged(score_log_file(), "datagen.answer_correctness_reward")


# --- decide/rewards.py -------------------------------------------------------


def test_decide_fraud_detection_reward(score_log_file):
    from agenttune.decide.rewards import fraud_detection_reward

    class TrajectoryWithoutState:
        pass

    fraud_detection_reward(TrajectoryWithoutState())
    _assert_logged(score_log_file(), "fraud_detection_reward")


def test_decide_verdict_binary_reward(score_log_file):
    from agenttune.decide.rewards import verdict_binary_reward

    class State:
        verdict = "APPROVE"

    class Trajectory:
        state = State()

    verdict_binary_reward(Trajectory())
    _assert_logged(score_log_file(), "verdict_binary_reward")


def test_decide_risk_score_reward(score_log_file):
    from agenttune.decide.rewards import risk_score_reward

    class State:
        overall_score = 3.0

    class Trajectory:
        state = State()

    risk_score_reward(Trajectory())
    _assert_logged(score_log_file(), "risk_score_reward")


# ---------------------------------------------------------------------------
# 3. A real, human-checkable JSONL trail (not redirected to tmp_path)
# ---------------------------------------------------------------------------


def test_scores_jsonl_trail_is_human_checkable(tmp_path, monkeypatch):
    """End-to-end sanity check: run a handful of real reward functions against
    a single ScoreLogger and confirm the resulting file is exactly what a
    human would want to open and read -- one JSON object per line, each with
    a `name`, a numeric `score`, and a non-empty `reasons` trail explaining it.
    """
    from agenttune.agentic.rewards.builtin_rewards import finqa, sql, use_case

    out = tmp_path / "scores.jsonl"
    monkeypatch.setattr(get_score_logger(), "output_file", str(out))

    sql.correctness_reward(completions=["*yes*"], answer=["yes"])
    finqa.format_reward(prompts=[""], completions=["<answer>42</answer>"])
    use_case.numerical_match_reward(
        prompts=[""], completions=["<answer>42</answer>"], answer=["42"]
    )

    records = _read_jsonl(out)
    names = {r["name"] for r in records}
    assert names == {
        "sql.correctness_reward",
        "finqa.format_reward",
        "use_case.numerical_match_reward",
    }
    for r in records:
        assert set(r.keys()) == {"ts", "name", "score", "reasons", "components", "meta"}
        assert r["reasons"], f"{r['name']} has no reason trail"
