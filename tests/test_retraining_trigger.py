"""
Tests for Path B retraining trigger: TrainingBuffer, RewardDriftTracker,
RetrainingTrigger.

Scope: Week 1. The drop-rate *gate* test is deferred to Week 2 (the gate is a
Week 2 deliverable); Week 1 only verifies drop_rate is correctly computed.
"""

import threading
from datetime import UTC

import pytest

from agenttune.decide.closed_loop.contracts import TrainingExample
from agenttune.decide.closed_loop.retraining_trigger import (
    RetrainingTrigger,
    RewardDriftTracker,
    TrainingBuffer,
    TriggerConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_example(
    root_cause: str = "wrong_tool",
    traj_id: str = "traj_x",
    with_chosen: bool = True,
    with_rejected: bool = True,
) -> TrainingExample:
    return TrainingExample(
        trajectory_id=traj_id,
        original_failure_type="tool_crash",
        root_cause=root_cause,
        prompt=[{"role": "user", "content": "do a task"}],
        chosen=[{"role": "assistant", "content": "correct"}] if with_chosen else None,
        rejected=[{"role": "assistant", "content": "wrong"}] if with_rejected else None,
    )


# ---------------------------------------------------------------------------
# TrainingBuffer
# ---------------------------------------------------------------------------


def test_buffer_add_and_drain():
    buf = TrainingBuffer()
    for i in range(5):
        buf.add(make_example(traj_id=f"t{i}"))
    assert len(buf) == 5

    drained = buf.drain()
    assert len(drained) == 5
    assert len(buf) == 0
    # Second drain returns nothing.
    assert buf.drain() == []


def test_buffer_thread_safety():
    buf = TrainingBuffer(max_size=10_000)
    n_threads, per_thread = 10, 100

    def worker(tid):
        for i in range(per_thread):
            buf.add(make_example(traj_id=f"{tid}_{i}"))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(buf) == n_threads * per_thread


def test_buffer_max_size_eviction():
    buf = TrainingBuffer(max_size=3)
    for i in range(5):
        buf.add(make_example(traj_id=f"t{i}"))
    # Capacity is capped; oldest evicted FIFO.
    assert len(buf) == 3
    drained = buf.drain()
    ids = [ex.trajectory_id for ex in drained]
    assert ids == ["t2", "t3", "t4"]


def test_buffer_novel_causes():
    buf = TrainingBuffer()
    buf.add(make_example(root_cause="wrong_tool"))
    buf.add(make_example(root_cause="loop_collapse"))
    # Nothing drained yet -> everything currently present is "novel".
    assert buf.novel_causes() == {"wrong_tool", "loop_collapse"}

    buf.drain()  # marks wrong_tool + loop_collapse as seen
    buf.add(make_example(root_cause="wrong_tool"))  # already seen
    buf.add(make_example(root_cause="hallucinated_output"))  # new
    assert buf.novel_causes() == {"hallucinated_output"}


# ---------------------------------------------------------------------------
# RewardDriftTracker
# ---------------------------------------------------------------------------


def test_reward_tracker_baseline_lock():
    tracker = RewardDriftTracker(window_size=10, baseline_size=10)
    assert not tracker.baseline_locked
    for _ in range(9):
        tracker.push(0.8)
    assert not tracker.baseline_locked  # not enough yet
    tracker.push(0.8)
    assert tracker.baseline_locked  # locked at 10th push
    # Baseline is a stable, healthy reward — pushing more healthy values
    # must NOT flag drift.
    for _ in range(10):
        tracker.push(0.8)
    assert tracker.is_drifting(sigma=1.5) is False


def test_reward_tracker_detects_drift():
    tracker = RewardDriftTracker(window_size=10, baseline_size=20)
    # Baseline with variance around ~0.8.
    import random

    rng = random.Random(0)
    for _ in range(20):
        tracker.push(0.8 + rng.uniform(-0.05, 0.05))
    assert tracker.baseline_locked
    # Now the model degrades sharply.
    for _ in range(10):
        tracker.push(0.2)
    assert tracker.is_drifting(sigma=1.5) is True


# ---------------------------------------------------------------------------
# RetrainingTrigger — triggers
# ---------------------------------------------------------------------------


def test_trigger_total_failures():
    cfg = TriggerConfig(total_failures_threshold=5, min_examples_ready=999, dominance_ratio=2.0)
    trig = RetrainingTrigger(config=cfg)
    for i in range(5):
        # Vary type so dominance does not fire first.
        rc = "wrong_tool" if i % 2 == 0 else "loop_collapse"
        trig.buffer.add(make_example(root_cause=rc, traj_id=f"t{i}"))
    fire, reason = trig.should_trigger()
    assert fire is True
    assert reason == "total_failures_exceeded"


def test_trigger_dominance():
    cfg = TriggerConfig(
        total_failures_threshold=999,
        min_examples_ready=999,
        dominance_ratio=0.70,
    )
    trig = RetrainingTrigger(config=cfg)
    for i in range(8):
        trig.buffer.add(make_example(root_cause="wrong_tool", traj_id=f"t{i}"))
    for i in range(2):
        trig.buffer.add(make_example(root_cause="loop_collapse", traj_id=f"l{i}"))
    fire, reason = trig.should_trigger()
    assert fire is True
    assert reason == "dominance:wrong_tool"


def test_trigger_enough_examples():
    cfg = TriggerConfig(
        total_failures_threshold=999,
        dominance_ratio=2.0,
        min_examples_ready=3,
    )
    trig = RetrainingTrigger(config=cfg)
    for i in range(3):
        rc = ["wrong_tool", "loop_collapse", "wrong_routing"][i]
        trig.buffer.add(make_example(root_cause=rc, traj_id=f"t{i}"))
    fire, reason = trig.should_trigger()
    assert fire is True
    assert reason == "enough_examples"


def test_trigger_reward_drift():
    cfg = TriggerConfig(
        total_failures_threshold=999,
        min_examples_ready=999,
        dominance_ratio=2.0,
        reward_drift_window=10,
        reward_drift_baseline_size=20,
        reward_drift_sigma=1.5,
    )
    trig = RetrainingTrigger(config=cfg)
    import random

    rng = random.Random(1)
    for _ in range(20):
        trig.push_reward(0.8 + rng.uniform(-0.05, 0.05))
    for _ in range(10):
        trig.push_reward(0.1)
    fire, reason = trig.should_trigger()
    assert fire is True
    assert reason == "reward_drift"


def test_trigger_novel_failure_type():
    cfg = TriggerConfig(
        total_failures_threshold=999,
        min_examples_ready=999,
        dominance_ratio=2.0,
        novel_type_min_count=3,
    )
    trig = RetrainingTrigger(config=cfg)
    # Seed + drain so these become "seen" and won't be novel.
    trig.buffer.add(make_example(root_cause="wrong_tool"))
    trig.buffer.drain()
    # A brand new type appears 3 times.
    for i in range(3):
        trig.buffer.add(make_example(root_cause="hallucinated_output", traj_id=f"h{i}"))
    fire, reason = trig.should_trigger()
    assert fire is True
    assert reason == "novel_failure:hallucinated_output"


def test_trigger_staleness():
    from datetime import datetime, timedelta

    cfg = TriggerConfig(
        total_failures_threshold=999,
        min_examples_ready=999,
        dominance_ratio=2.0,
        staleness_window_hours=6.0,
        min_stale_buffer_size=3,
    )
    trig = RetrainingTrigger(config=cfg)
    # Pretend a retrain finished 7 hours ago.
    trig._last_retrain_time = datetime.now(UTC) - timedelta(hours=7)
    for i in range(3):
        rc = ["wrong_tool", "loop_collapse", "wrong_routing"][i]
        trig.buffer.add(make_example(root_cause=rc, traj_id=f"t{i}"))
    fire, reason = trig.should_trigger()
    assert fire is True
    assert reason == "staleness"


# ---------------------------------------------------------------------------
# RetrainingTrigger — gate + health + fire
# ---------------------------------------------------------------------------


def test_trigger_blocked_by_retrain_in_progress():
    cfg = TriggerConfig(total_failures_threshold=1)
    trig = RetrainingTrigger(config=cfg)
    trig.buffer.add(make_example())  # would fire T1 immediately
    trig.mark_retrain_started()
    fire, reason = trig.should_trigger()
    assert fire is False
    assert reason == "retrain_already_running"
    # After finishing, the same condition fires.
    trig.mark_retrain_finished()
    fire, reason = trig.should_trigger()
    assert fire is True


def test_health_computation():
    buf = TrainingBuffer()
    # 3 accepted, 1 dropped (no chosen, no rejected).
    buf.add(make_example(root_cause="wrong_tool"))
    buf.add(make_example(root_cause="wrong_tool"))
    buf.add(make_example(root_cause="loop_collapse"))
    buf.add(make_example(root_cause="wrong_tool", with_chosen=False, with_rejected=False))

    health = buf.health()
    assert health.total_examples_attempted == 4
    assert health.total_examples_accepted == 3
    assert health.drop_rate == pytest.approx(0.25)
    assert health.buffer_size == 4
    assert health.dominant_failure_type == "wrong_tool"
    assert health.dominant_failure_ratio == pytest.approx(0.75)


def test_check_and_fire_drains_on_fire():
    cfg = TriggerConfig(total_failures_threshold=3, dominance_ratio=2.0, min_examples_ready=999)
    trig = RetrainingTrigger(config=cfg)
    for i in range(3):
        rc = ["wrong_tool", "loop_collapse", "wrong_routing"][i]
        trig.buffer.add(make_example(root_cause=rc, traj_id=f"t{i}"))
    examples = trig.check_and_fire()
    assert examples is not None
    assert len(examples) == 3
    assert len(trig.buffer) == 0  # drained


def test_check_and_fire_returns_none_below_threshold():
    cfg = TriggerConfig(total_failures_threshold=99, dominance_ratio=2.0, min_examples_ready=99)
    trig = RetrainingTrigger(config=cfg)
    trig.buffer.add(make_example())
    assert trig.check_and_fire() is None
    assert len(trig.buffer) == 1  # not drained
