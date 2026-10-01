"""Phase 6 (train wiring) — the discriminating test: does the dataset the SPINE
produces from its own trajectories match what the REAL SFT trainer consumes?

The SFT trainer (backends/.../sft.py, CHAT_COMPLETION branch) reads a ``messages``
column, requires each value be a list of ``{role, content}`` dicts, and feeds it to
``apply_chat_template``. ``EventLog.as_dataset_rows('sft')`` emits exactly that. So the
load-bearing test records the dataset Project hands the trainer and asserts it satisfies
that consumption contract — proving data-fit, not merely that ``.train()`` was called.

RL/GRPO needs native-Trajectory logprobs from a real rollout harness (Phase 2b) and
stays an honest NotImplementedError; light-tier toy traces cannot be trained on.
"""

import pytest

from agenttune.agentic.events import EventLog
from agenttune.agentic.harness import DictToolHarness
from agenttune.agentic.project import Project
from agenttune.agentic.strategy import ReActStrategy
from agenttune.agentic.trajectory.dataset import Step, Trajectory


class _RecordingTrainer:
    """Stand-in for TRLSFTTrainer: records the dataset it is handed, mimics .train()."""

    last_dataset = None

    def __init__(self, *, train_dataset=None, **kw):
        _RecordingTrainer.last_dataset = train_dataset
        self.train_dataset = train_dataset

    def train(self):
        return {"train_loss": 0.123, "n_rows": len(self.train_dataset)}


def _teacher_log():
    traj = Trajectory(
        task="q",
        steps=[
            Step(
                step_number=0,
                state="s",
                action={"name": "search", "arguments": {"q": "x"}},
                observation="doc1",
                thought="I should search",
            )
        ],
        reward=1.0,
        final_response="the answer",
        logprobs=[-0.1, -0.2],
    )
    return EventLog.from_trajectory(traj)  # full tier (carries logprobs)


# ---- discriminating test: spine dataset fits the real SFT schema ----


def test_train_feeds_sft_schema_from_spine_trajectories():
    p = Project()
    p.add_trajectory(_teacher_log())

    result = p.train(trainer_factory=_RecordingTrainer)

    # the trainer ran and returned its metrics through the spine
    assert result["train_loss"] == 0.123
    # Project fed the trainer rows derived from ITS OWN trajectories...
    ds = _RecordingTrainer.last_dataset
    assert ds and isinstance(ds, list)
    # ...and those rows satisfy the real SFT CHAT_COMPLETION contract exactly
    for row in ds:
        assert "messages" in row
        assert isinstance(row["messages"], list)
        for m in row["messages"]:
            assert "role" in m and "content" in m  # the access the trainer makes
    # the teacher's reasoning made it into the SFT signal
    contents = [m["content"] for row in ds for m in row["messages"]]
    assert "I should search" in contents

    # lifecycle events were emitted around the training call
    stages = [(ev.stage, ev.kind) for ev in p.events()]
    assert ("train", "started") in stages and ("train", "done") in stages


# ---- honest guards ----


def test_train_rejects_light_tier_only():
    """Observational light-tier traces (DictToolHarness) cannot be trained on."""
    calls = iter([{"name": "finish", "arguments": {"answer": "a"}}])
    p = Project(
        strategy=ReActStrategy(policy=lambda s: next(calls)),
        harness=DictToolHarness({"noop": lambda: ""}),
    )
    p.infer("q")  # light-tier trajectory only
    with pytest.raises(ValueError, match="full-tier"):
        p.train(trainer_factory=_RecordingTrainer)


def test_train_grpo_needs_a_rollout_engine():
    """RL is on-policy: fmt='grpo' wires a rollout engine (Phase 2b), not a precollected
    SFT buffer. Without an engine it must fail clearly rather than silently SFT."""
    p = Project()
    p.add_trajectory(_teacher_log())
    with pytest.raises(ValueError, match="rollout_engine"):
        p.train(trainer_factory=_RecordingTrainer, fmt="grpo")
