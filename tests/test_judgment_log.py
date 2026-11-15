"""Judge verdicts are written to a JSONL file only when asked (AGENTTUNE_JUDGMENT_LOG or an
explicit path), never by default into the working directory or the install tree."""

import asyncio
import json
from pathlib import Path

import pytest

import agenttune.eval.agentic.trajectory_eval as te
from agenttune.agentic.events.judgment_hook import JudgmentHook

VERDICT = json.dumps(
    {
        "goal_completion": 1.0,
        "tool_sequence_validity": 1.0,
        "unnecessary_steps": 1.0,
        "error_recovery": 1.0,
        "intent_action_alignment": 1.0,
        "evidence_grounding": 1.0,
        "overall_score": 0.9,
    }
)
TRAJ = {"trajectory_id": "t1", "query": "q", "final_answer": "a", "tool_calls": []}
PACKAGE_LOG = Path(te.__file__).parent / "../../../../data/judgments.jsonl"


class FakeEngine:
    async def generate_batch(self, messages, **kwargs):
        return [VERDICT for _ in messages]


def evaluate():
    return asyncio.run(te.TrajectoryEvaluator(engine=FakeEngine()).evaluate_batch([TRAJ]))


def size(path):
    return path.stat().st_size if path.exists() else -1


@pytest.fixture
def no_env(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTTUNE_JUDGMENT_LOG", raising=False)
    monkeypatch.delenv("JUDGMENTS_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_evaluate_batch_writes_nothing_by_default(no_env):
    before = size(PACKAGE_LOG)
    assert evaluate()[0].trajectory_id == "t1"
    assert list(no_env.iterdir()) == []
    assert size(PACKAGE_LOG) == before


@pytest.mark.parametrize("var", ["AGENTTUNE_JUDGMENT_LOG", "JUDGMENTS_FILE"])
def test_evaluate_batch_writes_where_asked(no_env, monkeypatch, var):
    out = no_env / "logs" / "judgments.jsonl"
    monkeypatch.setenv(var, str(out))
    evaluate()
    assert json.loads(out.read_text().splitlines()[0])["trajectory_id"] == "t1"


def test_judgment_hook_needs_a_path(no_env, monkeypatch):
    with pytest.raises(ValueError, match="AGENTTUNE_JUDGMENT_LOG"):
        JudgmentHook()
    assert list(no_env.iterdir()) == []
    monkeypatch.setenv("AGENTTUNE_JUDGMENT_LOG", str(no_env / "a" / "j.jsonl"))
    assert JudgmentHook().output_file == str(no_env / "a" / "j.jsonl")
    assert JudgmentHook("j.jsonl").output_file == "j.jsonl"  # no directory part: no crash
