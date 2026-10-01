import pytest

from agenttune.agentic.memory import MemoryItem, MemoryKind, Scope
from agenttune.agentic.memory_graph import GraphMemory

# ---------- relational retrieval (the discriminating test) ----------


def test_neighbors_traverses_graph_by_depth():
    m = GraphMemory()
    a = m.write(MemoryItem(content="A"))
    b = m.write(MemoryItem(content="B"))
    c = m.write(MemoryItem(content="C"))
    m.write(MemoryItem(content="D"))  # unrelated, unlinked
    m.link(a, b, "knows")
    m.link(b, c, "knows")
    # depth 1: only the immediate neighbor B
    assert [i.content for i in m.neighbors(a, k=1)] == ["B"]
    # depth 2: B (1 hop) and C (2 hops), never the unrelated D
    assert sorted(i.content for i in m.neighbors(a, k=2)) == ["B", "C"]
    assert "D" not in {i.content for i in m.neighbors(a, k=9)}


def test_read_returns_graph_neighborhood_of_matched_node():
    m = GraphMemory()
    a = m.write(MemoryItem(content="Alice"))
    b = m.write(MemoryItem(content="Bob"))
    m.write(MemoryItem(content="Dave"))  # unrelated node D
    m.link(a, b, "friend")
    got = m.read("Alice", k=5)  # match A -> return connected items
    assert [i.content for i in got] == ["Bob"]  # neighborhood, not a flat list
    assert "Dave" not in {i.content for i in got}


def test_delete_prunes_incoming_edges_so_node_is_no_longer_a_neighbor():
    m = GraphMemory()
    a = m.write(MemoryItem(content="A"))
    b = m.write(MemoryItem(content="B"))
    c = m.write(MemoryItem(content="C"))
    m.link(a, b, "knows")
    m.link(b, c, "knows")
    m.delete(b)  # must prune the A->B incoming edge
    names = {i.content for i in m.neighbors(a, k=9)}
    assert "B" not in names  # deleted node not returned
    assert "C" not in names  # C only reachable through B
    # the incoming A->b edge itself must be pruned from the adjacency structure,
    # not merely filtered at read time — no adjacency list may still point at b.
    assert all(b not in [dst for dst, _rel in adj] for adj in m._edges.values())


# ---------- link/neighbors API guards ----------


def test_link_requires_existing_nodes():
    m = GraphMemory()
    a = m.write(MemoryItem(content="A"))
    with pytest.raises(KeyError):
        m.link(a, "does-not-exist", "rel")


def test_neighbors_of_unknown_or_isolated_node_is_empty():
    m = GraphMemory()
    a = m.write(MemoryItem(content="A"))
    assert m.neighbors("missing", k=1) == []
    assert m.neighbors(a, k=1) == []


# ---------- scope / kind filters ----------


def test_read_scope_and_kind_filter():
    m = GraphMemory()
    sa, sb = Scope(agent_id="a"), Scope(agent_id="b")
    seed = m.write(MemoryItem(content="seed"))
    na = m.write(MemoryItem(content="cat", kind=MemoryKind.SEMANTIC), scope=sa)
    nb = m.write(MemoryItem(content="cat", kind=MemoryKind.EPISODIC), scope=sb)
    m.link(seed, na, "rel")
    m.link(seed, nb, "rel")
    assert [i.scope for i in m.read("seed", scope=sa)] == [sa]
    assert [i.kind for i in m.read("seed", kind=MemoryKind.EPISODIC)] == [MemoryKind.EPISODIC]


def test_read_without_query_returns_recent_k():
    m = GraphMemory()
    for i in range(4):
        m.write(MemoryItem(content=f"n{i}"))
    got = m.read(k=2)
    assert [i.content for i in got] == ["n2", "n3"]


# ---------- CRUD ----------


def test_update_and_delete():
    m = GraphMemory()
    mid = m.write(MemoryItem(content="old"))
    m.update(mid, {"content": "new"})
    assert m.read(k=1)[0].content == "new"
    m.delete(mid)
    assert m.read(k=99) == []


# ---------- snapshot / restore ----------


def test_snapshot_restore_round_trip_preserves_graph_and_edges():
    m = GraphMemory()
    a = m.write(MemoryItem(content="A"))
    b = m.write(MemoryItem(content="B"))
    m.link(a, b, "knows")
    snap = m.snapshot()
    c = m.write(MemoryItem(content="C"))
    m.link(a, c, "knows")
    assert len(m.neighbors(a, k=1)) == 2
    m.restore(snap)
    # graph + edges restored exactly
    assert [i.content for i in m.read(k=99)] == ["A", "B"]
    assert [i.content for i in m.neighbors(a, k=1)] == ["B"]


# ---------- trainable consolidate / forget hooks prune edges ----------


def test_forget_policy_prunes_edges():
    m = GraphMemory()
    a = m.write(MemoryItem(content="cat"))
    b = m.write(MemoryItem(content="dog"))
    c = m.write(MemoryItem(content="cat too"))
    m.link(a, b, "rel")
    m.link(a, c, "rel")
    # drop everything without "cat" -> B removed, and the A->B edge must be pruned
    m.forget(policy=lambda items: [i for i in items if "cat" in str(i.content)])
    assert sorted(i.content for i in m.read(k=99)) == ["cat", "cat too"]
    names = {i.content for i in m.neighbors(a, k=9)}
    assert names == {"cat too"}  # dangling A->B edge is gone


def test_consolidate_policy_prunes_edges():
    m = GraphMemory()
    a = m.write(MemoryItem(content="A"))
    b = m.write(MemoryItem(content="B"))
    m.link(a, b, "rel")
    m.consolidate(policy=lambda items: items[:1])  # keep only A
    assert [i.content for i in m.read(k=99)] == ["A"]
    assert m.neighbors(a, k=9) == []  # edge to dropped B pruned


def test_export():
    from agenttune.agentic import GraphMemory as G

    assert G is GraphMemory


# ---------- temporal decay / recency (Graphiti/Zep, arXiv:2501.13956) ----------


def test_recency_weighted_read_ranks_newer_neighbor_first():
    m = GraphMemory()
    s = m.write(MemoryItem(content="seed"))
    old = m.write(MemoryItem(content="old note"))  # written first -> older
    new = m.write(MemoryItem(content="new note"))  # written last  -> newer
    m.link(s, old, "rel")
    m.link(s, new, "rel")
    # default: insertion (edge) order, oldest neighbor first
    assert [i.content for i in m.read("seed")] == ["old note", "new note"]
    # recency-weighted: the more recent write outranks the older one
    ranked = m.read("seed", recency_weighted=True)
    assert [i.content for i in ranked] == ["new note", "old note"]


def test_injected_clock_makes_timestamps_deterministic():
    ticks = iter([10, 20, 30])
    m = GraphMemory(clock=lambda: next(ticks))
    s = m.write(MemoryItem(content="seed"))
    a = m.write(MemoryItem(content="a"))  # ts=20
    b = m.write(MemoryItem(content="b"))  # ts=30 (newer)
    m.link(s, a, "rel")
    m.link(s, b, "rel")
    assert [i.content for i in m.read("seed", recency_weighted=True)] == ["b", "a"]


def test_forget_older_than_drops_stale_nodes_and_prunes_dangling_edges():
    m = GraphMemory()
    a = m.write(MemoryItem(content="A"))  # ts=1 (stale)
    b = m.write(MemoryItem(content="B"))  # ts=2
    c = m.write(MemoryItem(content="C"))  # ts=3 (now)
    m.link(b, a, "knows")  # incoming edge onto stale A
    m.link(b, c, "knows")
    m.forget_older_than(1)  # now=3; drop age>1 -> drops A (age 2)
    contents = {i.content for i in m.read(k=99)}
    assert contents == {"B", "C"}  # stale A dropped, fresh kept
    # the dangling B->A edge must be pruned from the adjacency structure itself,
    # not merely filtered at read time.
    assert all("A" not in [dst for dst, _rel in adj] for adj in m._edges.values())
    a_id = a
    assert a_id not in m._edges  # outgoing edges gone too
    assert all(a_id != i.id for i in m.read(k=99))


def test_snapshot_restore_preserves_timestamps():
    m = GraphMemory()
    a = m.write(MemoryItem(content="A"))
    b = m.write(MemoryItem(content="B"))
    m.link(a, b, "rel")
    snap = m.snapshot()
    m.write(MemoryItem(content="C"))  # advance logical time
    m.restore(snap)
    # after restore, recency ranking must reflect the snapshotted timestamps
    seed = m.write(MemoryItem(content="seed"))  # newest
    m.link(seed, a, "rel")
    m.link(seed, b, "rel")
    assert [i.content for i in m.read("seed", recency_weighted=True)] == ["B", "A"]
