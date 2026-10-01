"""GraphMemory — entity/relationship (knowledge-graph) memory driver (additive).

Grounded in Zep/Graphiti (arXiv:2501.13956) and A-MEM (arXiv:2502.12110): items
are stored as nodes and connected by typed, directed edges, so retrieval is
RELATIONAL rather than flat. `read(query)` matches a node by content and returns
its graph neighborhood (the items connected to it), and `neighbors(id, k)` walks
outgoing edges up to `k` hops. Stdlib-only — plain adjacency dicts, NO networkx,
NO numpy, NO model/GPU/network — so it is fully unit-testable. Scope/kind filters,
CRUD, snapshot/restore and consolidate/forget policies match BaseMemory /
VectorMemory exactly; delete/forget/consolidate prune edges (both directions) so
the graph never keeps a dangling reference to a removed node.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from agenttune.agentic.memory import BaseMemory, MemoryItem, MemoryKind, Scope


class GraphMemory(BaseMemory):
    """Node/edge memory. Keeps items plus an adjacency map of typed, directed
    edges (`src_id -> [(dst_id, relation), ...]`) so retrieval can follow
    relationships instead of returning a flat recent-k list."""

    def __init__(self, *, clock: Callable[[], Any] | None = None) -> None:
        self._items: list[MemoryItem] = []
        self._edges: dict[str, list[tuple[str, str]]] = {}  # src -> [(dst, relation)]
        # Temporal-knowledge-graph bookkeeping (Graphiti/Zep, arXiv:2501.13956):
        # every node carries a LOGICAL timestamp assigned at write. With no wall
        # clock we default to a monotonic counter; an injected `clock` (returning
        # an int/float that never decreases) keeps it deterministic and testable.
        # Timestamps live in a parallel dict keyed by id so MemoryItem's shape is
        # unchanged. "now" is the newest assigned timestamp (max of `_ts`), never a
        # fresh clock() call — reading the clock must not advance logical time.
        self._ts: dict[str, Any] = {}  # id -> logical timestamp
        self._counter: int = 0  # monotonic default clock state
        self._clock: Callable[[], Any] = clock if clock is not None else self._tick

    def _tick(self) -> int:
        self._counter += 1
        return self._counter

    def _now(self) -> Any:
        """Newest logical timestamp seen so far (0 when the graph is empty)."""
        return max(self._ts.values()) if self._ts else 0

    # ---- CRUD -------------------------------------------------------------

    def write(self, item: MemoryItem, *, scope: Scope | None = None) -> str:
        if scope is not None:
            item = MemoryItem(
                content=item.content,
                kind=item.kind,
                scope=scope,
                id=item.id,
                metadata=item.metadata,
            )
        self._items.append(item)
        self._edges.setdefault(item.id, [])
        self._ts[item.id] = self._clock()  # stamp with logical time
        return item.id

    def read(
        self,
        query: Any = None,
        *,
        k: int = 5,
        scope: Scope | None = None,
        kind: MemoryKind | None = None,
        recency_weighted: bool = False,
        hops: int = 1,
    ) -> list[MemoryItem]:
        items = self._items
        if scope is not None:
            items = [i for i in items if i.scope == scope]
        if kind is not None:
            items = [i for i in items if i.kind is kind]
        if query is None:
            return items[-k:]  # no query -> recent-k fallback
        # Relational retrieval: match seed node(s) by content over the FULL graph
        # (so an out-of-scope intermediate node can't sever a path), traverse the
        # `hops`-hop neighborhood (1 hop by default, unchanged from before), then
        # apply the scope/kind filter to the results.
        allowed = {i.id for i in items}
        seeds = [i for i in self._items if self._match(query, i)]
        out: list[MemoryItem] = []
        seen: set[str] = set()
        for s in seeds:
            for n in self.neighbors(s.id, k=hops):
                if n.id in allowed and n.id not in seen:
                    seen.add(n.id)
                    out.append(n)
        if recency_weighted:
            # Rank by recency-decayed score (higher = more recent) instead of
            # traversal/insertion order; ties keep their original relative order.
            out.sort(key=lambda i: self._recency_score(i.id), reverse=True)
        return out[:k]

    def update(self, id: str, patch: dict) -> None:
        for i in self._items:
            if i.id == id:
                if "content" in patch:
                    i.content = patch["content"]
                i.metadata.update(patch.get("metadata", {}))
                return

    def delete(self, id: str) -> None:
        self._items = [i for i in self._items if i.id != id]
        self._edges.pop(id, None)  # drop outgoing edges
        self._ts.pop(id, None)  # drop its logical timestamp
        self._drop_incoming({id})  # drop incoming edges

    # ---- relationship API -------------------------------------------------

    def link(self, src_id: str, dst_id: str, relation: str) -> None:
        """Add a typed, directed edge `src -> dst`. Both nodes must already exist."""
        for nid in (src_id, dst_id):
            if nid not in self._edges:
                raise KeyError(f"unknown node id: {nid}")
        self._edges[src_id].append((dst_id, relation))

    def neighbors(self, id: str, k: int = 1) -> list[MemoryItem]:
        """Items reachable from `id` within `k` HOPS along outgoing edges (BFS,
        start node excluded). NB: here `k` is a traversal depth, unlike `read`'s
        `k` which is a result count."""
        pool = {i.id: i for i in self._items}
        if id not in pool:
            return []
        seen = {id}
        frontier = [id]
        order: list[str] = []
        for _ in range(max(0, k)):
            nxt: list[str] = []
            for node in frontier:
                for dst, _rel in self._edges.get(node, []):
                    if dst not in seen:
                        seen.add(dst)
                        order.append(dst)
                        nxt.append(dst)
            frontier = nxt
            if not frontier:
                break
        return [pool[d] for d in order if d in pool]

    # ---- trainable policies ----------------------------------------------

    def consolidate(self, *, policy: Callable | None = None) -> None:
        if policy is not None:
            self._items = list(policy(self._items))
            self._prune_edges()

    def forget(self, *, policy: Callable | None = None) -> None:
        if policy is not None:
            self._items = list(policy(self._items))
            self._prune_edges()

    def forget_older_than(self, age: Any) -> None:
        """Temporal-decay forget: drop every node whose logical age (`now - ts`)
        strictly exceeds `age`, then prune the edges that pointed at (or out of)
        the dropped nodes so the graph keeps no dangling references. Reuses the
        same `_prune_edges` helper as the policy-based `forget`/`consolidate`."""
        now = self._now()
        self._items = [i for i in self._items if (now - self._ts.get(i.id, now)) <= age]
        self._prune_edges()

    def _recency_score(self, id: str) -> float:
        """Monotonic recency score in (0, 1]: 1.0 for the newest node, decaying
        as a node gets logically older. Any monotonic function of age gives the
        same ranking; this mirrors the Graphiti/Zep exponential-decay framing."""
        age = self._now() - self._ts.get(id, self._now())
        return 0.5 ** float(age)

    # ---- snapshot / restore ----------------------------------------------

    def snapshot(self):
        return (
            copy.deepcopy(self._items),
            copy.deepcopy(self._edges),
            dict(self._ts),
            self._counter,
        )

    def restore(self, state) -> None:
        items, edges = state[0], state[1]
        self._items = copy.deepcopy(items)
        self._edges = copy.deepcopy(edges)
        # Tolerant of pre-temporal 2-tuple snapshots: rebuild timestamps if absent.
        self._ts = (
            dict(state[2]) if len(state) > 2 else {i.id: n for n, i in enumerate(self._items, 1)}
        )
        self._counter = state[3] if len(state) > 3 else len(self._items)

    # ---- internals --------------------------------------------------------

    @staticmethod
    def _match(query: Any, item: MemoryItem) -> bool:
        return str(query).lower() in str(item.content).lower()

    def _drop_incoming(self, gone: set[str]) -> None:
        for src, adj in self._edges.items():
            self._edges[src] = [(d, r) for (d, r) in adj if d not in gone]

    def _prune_edges(self) -> None:
        keep = {i.id for i in self._items}
        self._edges = {
            src: [(d, r) for (d, r) in adj if d in keep]
            for src, adj in self._edges.items()
            if src in keep
        }
        self._ts = {i: t for i, t in self._ts.items() if i in keep}
