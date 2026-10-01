"""
Week 2 Path B unit tests:
- drop-rate buffer-health gate in RetrainingTrigger
- adapter-only retrain config / dataset conversion (retrain_config)
- background retrain runner lifecycle (retrain_runner)

No GPU / no real model: the retrain job is a stub callable, and dataset
conversion is checked structurally.
"""

import threading
import time

import pytest

from agenttune.decide.closed_loop.contracts import TrainingExample
from agenttune.decide.closed_loop.retrain_config import (
    RetrainConfig,
    build_retrain_config,
    examples_to_bco_dataset,
    examples_to_dpo_dataset,
)
from agenttune.decide.closed_loop.retrain_runner import (
    BackgroundRetrainer,
    RetrainResult,
)
from agenttune.decide.closed_loop.retraining_trigger import (
    RetrainingTrigger,
    TriggerConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_example(root_cause="wrong_tool", traj_id="t", chosen=True, rejected=True):
    return TrainingExample(
        trajectory_id=traj_id,
        original_failure_type="tool_crash",
        root_cause=root_cause,
        prompt=[{"role": "user", "content": "do a task"}],
        chosen=[{"role": "assistant", "content": "correct"}] if chosen else None,
        rejected=[{"role": "assistant", "content": "wrong"}] if rejected else None,
    )


# ---------------------------------------------------------------------------
# Drop-rate buffer-health gate (Week 2)
# ---------------------------------------------------------------------------


def test_trigger_blocked_by_high_drop_rate():
    """A buffer with an excessive drop rate must skip the retrain."""
    cfg = TriggerConfig(
        total_failures_threshold=3,  # would fire on volume
        dominance_ratio=2.0,
        min_examples_ready=999,
        max_drop_rate=0.50,
        min_attempts_for_drop_gate=4,
    )
    trig = RetrainingTrigger(config=cfg)
    # 2 good (accepted), 4 no-signal (dropped) → drop_rate = 4/6 = 0.67 > 0.50
    trig.buffer.add(make_example(traj_id="g0"))
    trig.buffer.add(make_example(traj_id="g1"))
    for i in range(4):
        trig.buffer.add(make_example(traj_id=f"b{i}", chosen=False, rejected=False))

    fire, reason = trig.should_trigger()
    assert fire is False
    assert reason == "drop_rate_too_high"
    # And check_and_fire must NOT drain when blocked.
    assert trig.check_and_fire() is None
    assert len(trig.buffer) == 6


def test_trigger_not_blocked_when_drop_rate_ok():
    """Healthy buffer (low drop rate) still fires normally."""
    cfg = TriggerConfig(
        total_failures_threshold=3,
        dominance_ratio=2.0,
        min_examples_ready=999,
        max_drop_rate=0.50,
        min_attempts_for_drop_gate=4,
    )
    trig = RetrainingTrigger(config=cfg)
    for i in range(5):
        trig.buffer.add(
            make_example(root_cause=["wrong_tool", "loop_collapse"][i % 2], traj_id=f"g{i}")
        )
    fire, reason = trig.should_trigger()
    assert fire is True
    assert reason == "total_failures_exceeded"


def test_drop_rate_gate_ignored_below_min_attempts():
    """With too few attempts, the drop-rate gate does not apply yet."""
    cfg = TriggerConfig(
        total_failures_threshold=2,
        dominance_ratio=2.0,
        min_examples_ready=999,
        max_drop_rate=0.50,
        min_attempts_for_drop_gate=10,  # high → gate inactive at small N
    )
    trig = RetrainingTrigger(config=cfg)
    # 2 dropped examples → drop_rate=1.0, but only 2 attempts (< 10) so gate is off.
    trig.buffer.add(make_example(traj_id="b0", chosen=False, rejected=False))
    trig.buffer.add(make_example(traj_id="b1", chosen=False, rejected=False))
    fire, reason = trig.should_trigger()
    assert fire is True
    assert reason == "total_failures_exceeded"


# ---------------------------------------------------------------------------
# Dataset conversion (retrain_config)
# ---------------------------------------------------------------------------


def test_examples_to_dpo_dataset_pairs_only():
    examples = [
        make_example(traj_id="a"),  # full pair
        make_example(traj_id="b", chosen=False),  # negative-only → skipped
        make_example(traj_id="c"),  # full pair
    ]
    rows = examples_to_dpo_dataset(examples)
    assert len(rows) == 2
    for r in rows:
        assert set(r.keys()) == {"prompt", "chosen", "rejected"}
        assert r["prompt"] and r["chosen"] and r["rejected"]


def test_examples_to_bco_dataset_labels():
    examples = [
        make_example(traj_id="a"),  # chosen+rejected → 2 rows
        make_example(traj_id="b", chosen=False),  # rejected only   → 1 row (False)
    ]
    rows = examples_to_bco_dataset(examples)
    assert len(rows) == 3
    labels = sorted(r["label"] for r in rows)
    assert labels == [False, False, True]
    for r in rows:
        assert set(r.keys()) == {"prompt", "completion", "label"}


def test_build_retrain_config_dpo_attaches_lora():
    examples = [make_example(traj_id=f"t{i}") for i in range(3)]
    cfg = RetrainConfig(model="sshleifer/tiny-gpt2", algorithm="dpo", lora_r=8, max_steps=2)
    kwargs = build_retrain_config(examples, cfg)
    assert kwargs["model"] == "sshleifer/tiny-gpt2"
    assert "peft_config" in kwargs  # adapter-only
    assert kwargs["peft_config"]["r"] == 8
    assert kwargs["max_steps"] == 2
    assert kwargs["train_dataset"] is not None
    assert len(kwargs["train_dataset"]) == 3


def test_build_retrain_config_raises_on_no_usable_rows():
    # All negative-only → no DPO pairs.
    examples = [make_example(traj_id=f"n{i}", chosen=False) for i in range(3)]
    cfg = RetrainConfig(model="sshleifer/tiny-gpt2", algorithm="dpo")
    with pytest.raises(ValueError):
        build_retrain_config(examples, cfg)


# ---------------------------------------------------------------------------
# Bridge: multi-completion (Path A's output) -> DPO/BCO
# ---------------------------------------------------------------------------


def _completion_example(tid="c", rewards=(0.1, 0.9)):
    """An example in Path A's emitted form: completions/rewards, no chosen/rejected."""
    return TrainingExample(
        trajectory_id=tid,
        original_failure_type="tool_crash",
        root_cause="wrong_tool",
        prompt=[{"role": "user", "content": "task"}],
        completions=[[{"role": "assistant", "content": f"comp_{i}"}] for i in range(len(rewards))],
        rewards=list(rewards),
    )


def test_dpo_bridges_completions_to_pair():
    ex = _completion_example(rewards=(0.1, 0.5, 0.9))
    rows = examples_to_dpo_dataset([ex])
    assert len(rows) == 1
    assert "comp_2" in rows[0]["chosen"]  # highest reward → chosen
    assert "comp_0" in rows[0]["rejected"]  # lowest reward → rejected


def test_dpo_skips_single_completion():
    # Only 1 completion → no pair derivable → skipped.
    ex = _completion_example(rewards=(0.5,))
    assert examples_to_dpo_dataset([ex]) == []


def test_dpo_skips_equal_reward_completions():
    # All equal rewards → no meaningful preference → skipped.
    ex = _completion_example(rewards=(0.5, 0.5))
    assert examples_to_dpo_dataset([ex]) == []


def test_bco_thresholds_completions_by_mean_reward():
    ex = _completion_example(rewards=(0.1, 0.9))  # mean 0.5
    rows = examples_to_bco_dataset([ex])
    assert len(rows) == 2
    by_label = {r["label"]: r["completion"] for r in rows}
    assert "comp_1" in by_label[True]  # 0.9 >= mean → desirable
    assert "comp_0" in by_label[False]  # 0.1 <  mean → undesirable


def test_build_retrain_config_rejects_unknown_algorithm():
    examples = [make_example(traj_id="t")]
    cfg = RetrainConfig(model="x", algorithm="grpo")
    with pytest.raises(ValueError):
        build_retrain_config(examples, cfg)


# ---------------------------------------------------------------------------
# Background retrain runner (Week 2 design + tests)
# ---------------------------------------------------------------------------


def test_background_retrainer_runs_and_signals_done():
    rec = {}

    def job(x):
        time.sleep(0.05)
        return {"trained": x}

    runner = BackgroundRetrainer(on_done=lambda r: rec.update(done=r))
    started = runner.start(job, 42)
    assert started is True
    # Non-blocking: returns immediately while the job runs.
    result = runner.wait(timeout=5)
    assert isinstance(result, RetrainResult)
    assert result.success is True
    assert result.result == {"trained": 42}
    assert result.duration_s >= 0.0
    assert runner.is_done() is True
    assert rec["done"].success is True


def test_background_retrainer_captures_failure():
    def bad_job():
        raise RuntimeError("boom")

    runner = BackgroundRetrainer()
    runner.start(bad_job)
    result = runner.wait(timeout=5)
    assert result.success is False
    assert "boom" in result.error


def test_background_retrainer_no_double_retrain():
    """A second start() while running is refused (no double-retrain)."""
    release = threading.Event()

    def slow_job():
        release.wait(timeout=5)
        return "ok"

    runner = BackgroundRetrainer()
    assert runner.start(slow_job) is True
    assert runner.is_running() is True
    # Second start refused while the first is still running.
    assert runner.start(slow_job) is False
    release.set()
    runner.wait(timeout=5)
    assert runner.result.result == "ok"


def test_background_retrainer_drives_trigger_flag():
    """Runner sets/clears the trigger's retrain_in_progress flag."""
    trig = RetrainingTrigger(config=TriggerConfig(total_failures_threshold=1))
    trig.buffer.add(make_example())
    release = threading.Event()

    def slow_job():
        release.wait(timeout=5)
        return "done"

    runner = BackgroundRetrainer(trigger=trig)
    runner.start(slow_job)

    # While running: flag is SET → trigger refuses to fire again.
    assert trig.is_retrain_running() is True
    fire, reason = trig.should_trigger()
    assert fire is False
    assert reason == "retrain_already_running"

    # Finish: flag cleared, last_retrain_time recorded.
    release.set()
    runner.wait(timeout=5)
    assert trig.is_retrain_running() is False
    assert trig._last_retrain_time is not None
