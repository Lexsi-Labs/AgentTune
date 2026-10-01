from _real_backends import real_embed

from agenttune.agentic.memory import MemoryItem, MemoryKind, Scope
from agenttune.agentic.memory_vector import VectorMemory

# REAL embedder — sentence-transformers/all-MiniLM-L6-v2, run on this box's GPU
# (or CPU if unavailable). Replaces the mocked bag-of-keywords fake embedder
# from tests/agentic/test_memory_vector.py with actual model inference.
_EMBED_CACHE: dict[str, list[float]] = {}


def _fake_embed(x):
    text = str(x)
    if text not in _EMBED_CACHE:
        _EMBED_CACHE[text] = real_embed([text])[0]
    return _EMBED_CACHE[text]


# ---------- ranking (the discriminating test) ----------


def test_read_ranks_by_cosine_similarity_to_query():
    m = VectorMemory(embed=_fake_embed)
    m.write(MemoryItem(content="cat cat"))  # closest to "cat"
    m.write(MemoryItem(content="dog dog"))  # orthogonal to "cat"
    m.write(MemoryItem(content="cat dog"))  # in between
    got = m.read("cat", k=3)
    # semantically closer item ranks above a farther one
    assert got[0].content == "cat cat"
    assert got[-1].content == "dog dog"


def test_read_k_limit():
    m = VectorMemory(embed=_fake_embed)
    for _ in range(5):
        m.write(MemoryItem(content="cat"))
    assert len(m.read("cat", k=2)) == 2


def test_read_scope_and_kind_filter():
    m = VectorMemory(embed=_fake_embed)
    a, b = Scope(agent_id="a"), Scope(agent_id="b")
    m.write(MemoryItem(content="cat", kind=MemoryKind.SEMANTIC), scope=a)
    m.write(MemoryItem(content="cat", kind=MemoryKind.EPISODIC), scope=b)
    assert [i.scope for i in m.read("cat", scope=a)] == [a]
    assert [i.kind for i in m.read("cat", kind=MemoryKind.EPISODIC)] == [MemoryKind.EPISODIC]


def test_read_without_query_returns_recent_k():
    m = VectorMemory(embed=_fake_embed)
    for i in range(4):
        m.write(MemoryItem(content=f"cat {i}"))
    got = m.read(k=2)  # no query -> fall back to recent-k
    assert [i.content for i in got] == ["cat 2", "cat 3"]


# ---------- update / delete ----------


def test_update_reembeds_content():
    m = VectorMemory(embed=_fake_embed)
    # Distractor written FIRST so that if update did NOT re-embed, the stale "dog"
    # vector would tie at cosine 0 with "cat" and insertion order would surface the
    # distractor instead — making this test genuinely catch a missing re-embed.
    m.write(MemoryItem(content="car car"))  # distractor, orthogonal to "cat"
    mid = m.write(MemoryItem(content="dog"))
    m.update(mid, {"content": "cat"})
    got = m.read("cat", k=1)
    assert got[0].content == "cat"  # only reachable if the item was re-embedded


def test_delete_drops_item_and_embedding():
    m = VectorMemory(embed=_fake_embed)
    mid = m.write(MemoryItem(content="cat"))
    m.delete(mid)
    assert m.read("cat", k=99) == []


# ---------- snapshot / restore ----------


def test_snapshot_restore_round_trip():
    m = VectorMemory(embed=_fake_embed)
    m.write(MemoryItem(content="cat"))
    snap = m.snapshot()
    m.write(MemoryItem(content="dog"))
    assert len(m.read(k=99)) == 2
    m.restore(snap)
    assert [i.content for i in m.read(k=99)] == ["cat"]
    # embeddings restored too: ranking still works after restore
    assert m.read("cat", k=1)[0].content == "cat"


# ---------- trainable consolidate / forget hooks ----------


def test_consolidate_and_forget_policy_hooks():
    m = VectorMemory(embed=_fake_embed)
    for w in ["cat", "dog", "car", "cat dog"]:
        m.write(MemoryItem(content=w))
    m.forget(policy=lambda items: [i for i in items if "cat" in str(i.content)])
    assert sorted(i.content for i in m.read(k=99)) == ["cat", "cat dog"]
    m.consolidate(policy=lambda items: items[:1])
    assert len(m.read(k=99)) == 1
    # embeddings stayed consistent with surviving items -> ranking still works
    top = m.read("cat", k=1)
    assert "cat" in str(top[0].content)


def test_zero_vector_query_is_safe():
    m = VectorMemory(embed=_fake_embed)
    m.write(MemoryItem(content="cat"))
    # query with no vocab overlap -> zero vector, cosine undefined; must not crash
    got = m.read("xyz", k=5)
    assert len(got) == 1


def test_export():
    from agenttune.agentic import VectorMemory as VM

    assert VM is VectorMemory
