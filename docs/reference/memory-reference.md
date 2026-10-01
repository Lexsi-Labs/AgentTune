# Python API: Memory

`agenttune.agentic.memory` / `memory_vector.py` / `memory_graph.py`: four `BaseMemory`
implementations (in-context, trajectory replay, vector-similarity, graph). All pure Python,
no GPU/API key required for any of them; `VectorMemory` needs an `embed` callable you
supply, which *can* call a real model but doesn't have to. See
[Python API: Agentic Spine](../user-guide/agentic-spine.md) for the
one-paragraph overview and how `MemoryReActStrategy` reads/writes memory during a rollout.

!!! note "Not the same thing as `agenttune.rag.memory`"
    `agenttune.rag.memory` (MEM1-style, `m1_rewrite.py`/`m2_decisions.py`) is a completely
    different concept: it compresses a *single rollout's* running conversation state between
    turns so context stays roughly constant length, not a persistent store queried across
    episodes. See [Python API: RAG & Data Synthesis](../user-guide/rag-and-synthesis.md) for
    that system; it isn't covered further here.

## `BaseMemory`: the shared contract

```python
class MemoryKind(Enum):
    WORKING = "working"; EPISODIC = "episodic"; SEMANTIC = "semantic"
    PROCEDURAL = "procedural"; ENTITY = "entity"

@dataclass(frozen=True)
class Scope:
    agent_id: str = "default"
    namespace: str = "default"

@dataclass
class MemoryItem:
    content: Any
    kind: MemoryKind = MemoryKind.EPISODIC
    scope: Scope = field(default_factory=Scope)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)
```

`BaseMemory` (`agentic/memory.py`) declares 6 abstract methods every backend must implement,
plus 2 concrete no-op methods every backend may override:

| Method | Signature | Abstract? |
|---|---|---|
| `write` | `(item: MemoryItem, *, scope: Optional[Scope] = None) -> str` | Yes |
| `read` | `(query: Any = None, *, k: int = 5, scope: Optional[Scope] = None, kind: Optional[MemoryKind] = None) -> list[MemoryItem]` | Yes |
| `update` | `(id: str, patch: dict) -> None` | Yes |
| `delete` | `(id: str) -> None` | Yes |
| `snapshot` | `() -> Any` | Yes |
| `restore` | `(state: Any) -> None` | Yes |
| `consolidate` | `(*, policy: Optional[Callable] = None) -> None` | **No**; default is a no-op |
| `forget` | `(*, policy: Optional[Callable] = None) -> None` | **No**; default is a no-op |

`consolidate`/`forget` are the injectable, trainable-policy hooks (Memory-R1 pattern): a
`policy` is any `Callable[[list[MemoryItem]], list[MemoryItem]]`; pass one in and the
backend replaces its item list with whatever the policy returns; pass nothing and the call
does nothing at all. There is no built-in policy shipped anywhere in this module; you write
the `policy` callable yourself (e.g. "keep the 20 most recent" or a trained scorer).

## `InContextMemory`: recent-k, list-backed

```python
InContextMemory()
```

The ReAct scratchpad. `write` appends; `read(query=None, k=5, scope=None, kind=None)`
ignores `query` entirely and returns `items[-k:]` after filtering by `scope`/`kind`, a pure
recency window, no relevance ranking. `update(id, patch)` replaces `content` (from
`patch["content"]`, if present) and merges `patch.get("metadata", {})`. `snapshot()`/
`restore(state)` deep-copy the item list both ways.

## `TrajectoryStore`: episodic replay of full-tier `EventLog`s

```python
TrajectoryStore()
```

`write(item, *, scope=None)` **requires** `item.content` to be an `EventLog` instance,
raises `TypeError` otherwise, and always forces `kind=MemoryKind.EPISODIC` regardless of
what the `MemoryItem` was constructed with.

Two real gaps worth knowing before you rely on this class:

- **`read()` accepts a `kind` parameter but silently ignores it.** The signature is
  `read(query=None, *, k=5, scope=None, kind=None)`; it filters by `scope` and returns
  `items[-k:]`, but the `kind` filter present in `InContextMemory`/`VectorMemory`/
  `GraphMemory` is not applied here.
- **`update(id, patch)` only ever touches `metadata`.** Unlike `InContextMemory.update`, a
  `patch={"content": ...}` is silently dropped; `TrajectoryStore` has no supported way to
  edit a stored trajectory's content after the fact, only its metadata.

```python
def as_dataset(self, fmt: str = "sft") -> list[dict]
```

Iterates stored items and, for every one whose `content.tier == "full"`, extends the output
with `content.as_dataset_rows(fmt)` (an `EventLog` method; see `agentic/events.py`).
Light-tier logs (e.g. anything projected from a DECIDE `PipelineState` via
`EventLog.from_pipeline_state`) are silently skipped: zero rows, no error.
`EventLog.as_dataset_rows` itself only supports `fmt="sft"` today; anything else raises
`ValueError` inside that call.

`snapshot()` returns `list(self._items)`, a shallow copy of the *list*, not a deep copy of
the `MemoryItem` objects it contains (unlike every other backend on this page). Mutating a
restored item's `.metadata` in place will also mutate the object still referenced by the
original snapshot.

## `VectorMemory`: cosine-similarity semantic recall

```python
VectorMemory(embed: Callable[[Any], list[float]])
```

`embed` is **required** and is the only place a model can enter this class; it's called as
`self._embed(item.content)` on every `write()` and as `self._embed(query)` on every `read()`
with a non-`None` query, so it must accept whatever type your `content`/`query` values are
(typically `str`) and return a plain `list[float]`. A toy embedder that satisfies the type
needs nothing extra; a real one would typically wrap `sentence-transformers`. There is no
dimensionality check anywhere; an `embed` that returns vectors of inconsistent length will
silently truncate to the shorter one inside `zip()` in the cosine calculation rather than
raising.

```python
def read(self, query: Any = None, *, k: int = 5, scope=None, kind=None) -> list[MemoryItem]
```

`query=None` → same recency-window fallback as `InContextMemory` (`items[-k:]`, after
scope/kind filtering). Otherwise: embeds `query`, then ranks the scope/kind-filtered items
by a hand-rolled cosine similarity (`_cosine(a, b)`: plain-Python dot product / norms, no
numpy; returns `0.0` if either vector is all-zero) and returns the top `k`.

`update(id, patch)` re-embeds automatically whenever `patch` contains `"content"`.
`delete(id)` also drops the id from the internal embedding map. `consolidate`/`forget` apply
the policy like `InContextMemory` and then call an internal `_prune_vecs()` to drop
embeddings for items the policy removed. `snapshot()` returns a 2-tuple
`(deepcopy(items), deepcopy(vecs))`, a different shape from `InContextMemory`'s plain list,
and `restore(state)` unpacks that same 2-tuple.

## `GraphMemory`: typed-edge relational memory with temporal decay

```python
GraphMemory(*, clock: Optional[Callable[[], Any]] = None)
```

`clock` is an injectable logical-time source (must return a value that never decreases
across calls); the default is an internal monotonic counter (`_tick`, starting at 1). Every
`write()` stamps the new node with `self._clock()` in an internal `id -> timestamp` map and
seeds an empty adjacency list for it.

```python
def link(self, src_id: str, dst_id: str, relation: str) -> None
```

Adds one directed, typed edge `src -> dst`. Raises `KeyError` if either id isn't a known
node (both must already have been `write()`-ed).

```python
def neighbors(self, id: str, k: int = 1) -> list[MemoryItem]
```

BFS outward along outgoing edges up to `k` **hops** (start node excluded; returns `[]` for
an unknown id). This `k` is a traversal depth, deliberately different from `read()`'s `k`
below, which is a result count.

```python
def read(self, query=None, *, k=5, scope=None, kind=None, recency_weighted=False, hops=1) -> list[MemoryItem]
```

- `query=None` → the same recency-window fallback as every other backend.
- `query` given → matches "seed" nodes by a plain substring test against the **full**
  graph, unfiltered by scope/kind (`str(query).lower() in str(item.content).lower()`, so an
  out-of-scope node can still act as a bridge in the path); for each seed, walks
  `neighbors(seed.id, k=hops)`. **Traversal depth is controlled by the separate `hops`
  keyword (default `1`, matching the original behavior), not by `read()`'s own `k`
  argument**, which stays a result-count cap. Then filters those neighbors down to the
  scope/kind-allowed set, de-duplicates, and truncates to `k` results.
- `recency_weighted=True` re-sorts the (already-computed) result list by
  `_recency_score(id)` descending instead of leaving it in traversal order.
  `_recency_score` is `0.5 ** age` where `age = now - timestamp`: 1.0 for the newest node,
  decaying toward 0 for older ones. `now` is `max()` of every timestamp ever assigned, never
  a fresh call to `clock()`; reading doesn't advance logical time.

For a multi-hop traversal, call `.neighbors(id, k=N)` directly rather than `read()`.

```python
def forget_older_than(self, age: Any) -> None
```

Drops every node whose logical age (`now - timestamp`) **strictly exceeds** `age` (a node
exactly at the boundary is kept), then prunes any edge pointing at or from a dropped node so
the graph never keeps a dangling reference.

`delete(id)` removes the node, its outgoing edges, its timestamp, and scrubs incoming edges
from every other node (`_drop_incoming`). `consolidate`/`forget` follow the same
policy-driven pattern as the other backends, additionally pruning edges via `_prune_edges()`
after the policy runs. `update(id, patch)` only ever touches `content`/`metadata`; it does
not touch edges or the timestamp.

`snapshot()` returns a 4-tuple: `(deepcopy(items), deepcopy(edges), dict(timestamps),
counter)`. `restore(state)` is tolerant of an older 2-tuple `(items, edges)`; if timestamps
are absent it rebuilds them as `1..N` in list order and sets the counter to `len(items)`.

## See also

- [Python API: Agentic Spine](../user-guide/agentic-spine.md): the
  one-paragraph overview and how `MemoryReActStrategy(policy, memory, recall_k=5)` uses
  `.read()`/`.write()` during a rollout.
- [Python API: RAG & Data Synthesis](../user-guide/rag-and-synthesis.md): the unrelated
  MEM1-style per-rollout state-compression system mentioned above.
