import pytest

from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.memory import (
    BaseMemory,
    InContextMemory,
    MemoryItem,
    MemoryKind,
    Scope,
    TrajectoryStore,
)
from agenttune.agentic.trajectory.dataset import Step, Trajectory

# ---------- schema ----------


def test_memory_kinds_and_defaults():
    for name in ["WORKING", "EPISODIC", "SEMANTIC", "PROCEDURAL", "ENTITY"]:
        assert hasattr(MemoryKind, name)
    item = MemoryItem(content="x")
    assert item.kind is MemoryKind.EPISODIC
    assert item.scope == Scope()
    assert item.id


def test_basememory_is_abstract():
    with pytest.raises(TypeError):
        BaseMemory()


# ---------- InContextMemory ----------


def test_incontext_write_read_recent_k():
    m = InContextMemory()
    for i in range(5):
        m.write(MemoryItem(content=f"m{i}"))
    got = m.read(k=2)
    assert [i.content for i in got] == ["m3", "m4"]


def test_incontext_scope_and_kind_filter():
    m = InContextMemory()
    a, b = Scope(agent_id="a"), Scope(agent_id="b")
    m.write(MemoryItem(content="fa", kind=MemoryKind.SEMANTIC), scope=a)
    m.write(MemoryItem(content="fb", kind=MemoryKind.EPISODIC), scope=b)
    assert [i.content for i in m.read(scope=a)] == ["fa"]
    assert [i.content for i in m.read(kind=MemoryKind.EPISODIC)] == ["fb"]


def test_incontext_update_and_delete():
    m = InContextMemory()
    mid = m.write(MemoryItem(content="old"))
    m.update(mid, {"content": "new"})
    assert m.read(k=1)[0].content == "new"
    m.delete(mid)
    assert m.read() == []


def test_incontext_consolidate_and_forget_policies_are_trainable_hooks():
    m = InContextMemory()
    for i in range(4):
        m.write(MemoryItem(content=i))
    # a "policy" that keeps only even-valued items — stands in for a learned policy
    m.forget(policy=lambda items: [i for i in items if i.content % 2 == 0])
    assert sorted(i.content for i in m.read(k=99)) == [0, 2]
    m.consolidate(policy=lambda items: items[:1])
    assert len(m.read(k=99)) == 1


def test_incontext_snapshot_restore():
    m = InContextMemory()
    m.write(MemoryItem(content="a"))
    snap = m.snapshot()
    m.write(MemoryItem(content="b"))
    assert len(m.read(k=99)) == 2
    m.restore(snap)
    assert [i.content for i in m.read(k=99)] == ["a"]


# ---------- TrajectoryStore ----------


def _full_log():
    traj = Trajectory(
        task="t",
        steps=[
            Step(
                step_number=0,
                state="s",
                action={"name": "search", "arguments": {}},
                observation="obs",
                thought="think",
                reward=0.5,
            )
        ],
        reward=1.0,
        final_response="final",
        logprobs=[-0.1, -0.2],
    )
    return EventLog.from_trajectory(traj)


def test_trajectorystore_rejects_non_eventlog():
    s = TrajectoryStore()
    with pytest.raises(TypeError):
        s.write(MemoryItem(content="not a log"))


def test_trajectorystore_as_dataset_from_full_logs():
    s = TrajectoryStore()
    s.write(MemoryItem(content=_full_log()))
    # a light log should contribute nothing to the dataset
    light = EventLog(events=[Event(EventKind.TEXT, {"text": "x"})], tier="light")
    s.write(MemoryItem(content=light))
    rows = s.as_dataset("sft")
    assert len(rows) == 1  # only the full-tier log
    assert "messages" in rows[0]


def test_exports():
    from agenttune.agentic import InContextMemory as IC
    from agenttune.agentic import MemoryKind as MK
    from agenttune.agentic import TrajectoryStore as TS

    assert IC is InContextMemory and TS is TrajectoryStore and MK is MemoryKind
