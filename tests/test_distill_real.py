"""Phase 7 — agentic distillation: compress an expensive agent DESIGN into a small
student by SFT on the TEACHER's own trajectories (not weight-KD). Rides the same rails
proven in #23/#24: teacher full-tier trajectories -> SFT messages rows -> student trainer.

Discriminating (data-fit, GPU-free): the student trainer must receive (a) the student
model id and (b) a train_dataset derived from the TEACHER's trajectories that fits the real
SFT `messages` schema. Optionally collect teacher rollouts first via a real RolloutEngine.
"""

import pytest
from _real_backends import real_rollout_engine

from agenttune.agentic.events import EventLog
from agenttune.agentic.project import Project
from agenttune.agentic.trajectory.dataset import Step, Trajectory

# Loads Qwen2.5-0.5B via _real_backends (3-5GB RSS + ~1GB download);
# not for the 7.8GB CPU CI runner. Runs under -m qwen_e2e.
pytestmark = pytest.mark.qwen_e2e

# REAL teacher engine: agenttune's own `TransformersRolloutEngine` wrapping
# Qwen2.5-0.5B-Instruct, in place of the FakeTeacherEngine stand-in from
# tests/agentic/test_distill.py — teacher rollouts below are real generations.
FakeTeacherEngine = real_rollout_engine


class _RecordingTrainer:
    last_kwargs = None

    def __init__(self, **kwargs):
        _RecordingTrainer.last_kwargs = kwargs

    def train(self):
        return {"train_loss": 0.02}


def _teacher_log():
    traj = Trajectory(
        task="q",
        steps=[
            Step(
                step_number=0,
                state="s",
                action={"name": "search", "arguments": {"q": "x"}},
                observation="doc",
                thought="teacher chose to search",
            )
        ],
        reward=1.0,
        final_response="the answer",
        logprobs=[-0.1, -0.2],
    )
    return EventLog.from_trajectory(traj)


# ---- distill SFTs the student on teacher trajectories ----


def test_distill_sfts_student_on_teacher_trajectories():
    p = Project()
    p.add_trajectory(_teacher_log())

    result = p.distill("Qwen2.5-3B-Instruct", trainer_factory=_RecordingTrainer)

    assert result["train_loss"] == 0.02
    kw = _RecordingTrainer.last_kwargs
    assert kw["model"] == "Qwen2.5-3B-Instruct"  # the SMALL student
    ds = kw["train_dataset"]
    # dataset derived from the teacher, fitting the real SFT `messages` schema
    contents = [m["content"] for row in ds for m in row["messages"]]
    assert "teacher chose to search" in contents
    for row in ds:
        for m in row["messages"]:
            assert "role" in m and "content" in m
    stages = [(ev.stage, ev.kind) for ev in p.events()]
    assert ("distill", "started") in stages and ("distill", "done") in stages


# ---- distill can collect teacher rollouts first ----


def test_distill_collects_teacher_rollouts_first():
    p = Project()
    result = p.distill(
        "student-1.5B",
        trainer_factory=_RecordingTrainer,
        teacher_engine=FakeTeacherEngine(),
        tasks=["q1", "q2"],
        max_steps=1,
    )
    assert result["train_loss"] == 0.02
    assert len(p.native_trajectories) == 2  # teacher rollouts were collected
    assert len(_RecordingTrainer.last_kwargs["train_dataset"]) == 2


# ---- honest guards ----


def test_distill_requires_teacher_data():
    p = Project()
    with pytest.raises(ValueError, match="teacher"):
        p.distill("student", trainer_factory=_RecordingTrainer)


def test_distill_teacher_engine_needs_tasks():
    p = Project()
    with pytest.raises(ValueError, match="tasks"):
        p.distill("student", trainer_factory=_RecordingTrainer, teacher_engine=FakeTeacherEngine())
