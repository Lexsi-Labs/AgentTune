"""Phase 2b — rollout harness: the capability gate that makes GRPO + distillation real.

A real RolloutEngine produces FULL-tier trajectories (carrying logprobs). This wires that
producer into Project so:
  - collect_rollout(engine, tasks) yields full-tier EventLogs (SFT/distill/eval) AND retains
    native Trajectories (the RL substrate: to_trl_format);
  - train(fmt='grpo', rollout_engine=engine) un-stubs on-policy GRPO by wiring the real
    create_rollout_fn as the trainer's rollout_func.

Discriminating (data-fit, GPU-free): one rollout call must feed BOTH the GRPO on-policy
contract (env_mask/logprobs/prompt_ids) and the SFT `messages` schema — proven with a fake
engine driving the REAL rollout_factory assembly code (no reimplementation).
"""

import pytest

from agenttune.agentic.project import Project
from agenttune.agentic.rollout_engines.base import RolloutEngine


class _FakeTok:
    def apply_chat_template(self, *a, **k):
        raise RuntimeError("no template")

    def encode(self, text, add_special_tokens=True):
        return []


class FakeEngine(RolloutEngine):
    """A GPU-free stand-in producing a fixed completion + logprobs, so the REAL
    rollout_factory trajectory-assembly runs unchanged."""

    def _get_tokenizer(self):
        return _FakeTok()

    def generate(self, prompts, tools, gen_cfg):
        return {
            "completions": ["The answer is 42."],
            "logprobs": [[-0.1, -0.2, -0.3]],
            "metadata": {"backend": "fake"},
        }


class _RecordingTrainer:
    last_kwargs = None

    def __init__(self, **kwargs):
        _RecordingTrainer.last_kwargs = kwargs

    def train(self):
        return {"train_loss": 0.01}


# ---- collect_rollout feeds BOTH the SFT path and retains the RL substrate ----


def test_collect_rollout_yields_full_tier_and_native_trajectories():
    p = Project()
    logs = p.collect_rollout(FakeEngine(), ["what is 6x7?"], max_steps=1)

    assert len(logs) == 1
    log = logs[0]
    # full tier — carries the logprobs only the rollout path can produce
    assert log.tier == "full"
    assert log.masked_tokens() is not None  # full-tier only, must not raise
    # the SAME rollout is retained as a native Trajectory (RL substrate)
    assert len(p.native_trajectories) == 1
    assert hasattr(p.native_trajectories[0], "to_trl_format")
    assert p.native_trajectories[0].logprobs  # (logprob, token_id) tuples
    # and it fits the SFT `messages` schema (the #23 consumer)
    rows = log.as_dataset_rows("sft")
    for m in rows[0]["messages"]:
        assert "role" in m and "content" in m
    # lifecycle events emitted
    stages = [(ev.stage, ev.kind) for ev in p.events()]
    assert ("collect_rollout", "started") in stages and ("collect_rollout", "done") in stages


def test_collect_rollout_then_train_sft_end_to_end():
    """The gate: rollouts become training data through the spine, no GPU."""
    p = Project()
    p.collect_rollout(FakeEngine(), ["q1", "q2"], max_steps=1)
    result = p.train(trainer_factory=_RecordingTrainer)  # fmt='sft' default
    assert result["train_loss"] == 0.01
    ds = _RecordingTrainer.last_kwargs["train_dataset"]
    assert len(ds) == 2  # one row per rollout


# ---- train(fmt='grpo') is now reachable and wires the REAL rollout_func ----


def test_train_grpo_wires_real_rollout_func():
    p = Project()
    result = p.train(
        fmt="grpo", rollout_engine=FakeEngine(), trainer_factory=_RecordingTrainer, max_steps=1
    )
    assert result["train_loss"] == 0.01
    rollout_func = _RecordingTrainer.last_kwargs.get("rollout_func")
    assert callable(rollout_func)
    # data-fit: invoking it yields the GRPO on-policy contract, not a toy
    batch = rollout_func(["what is 6x7?"])
    assert {"env_mask", "logprobs", "prompt_ids", "completion_ids", "rewards"} <= set(batch)


def test_train_grpo_requires_engine():
    p = Project()
    with pytest.raises(ValueError, match="rollout_engine"):
        p.train(fmt="grpo", trainer_factory=_RecordingTrainer)
