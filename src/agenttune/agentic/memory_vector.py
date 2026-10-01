"""VectorMemory — semantic-retrieval memory driver (memory pillar, additive).

Unlike InContextMemory's recent-k, `read(query, k=...)` returns the k items whose
embedding is most cosine-similar to the query embedding. The embedding function is
INJECTED (`VectorMemory(embed=...)`) so this stays stdlib-only and testable with a
tiny deterministic fake embedder — NO real model, NO GPU, NO network. Cosine is
implemented here by hand (no numpy). Scope/kind filters and CRUD + snapshot/restore
+ consolidate/forget policies match BaseMemory / InContextMemory exactly.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import Any

from agenttune.agentic.memory import BaseMemory, MemoryItem, MemoryKind, Scope


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is the zero vector)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class VectorMemory(BaseMemory):
    """Embedding-backed semantic/entity memory. Keeps items plus a parallel map of
    item-id -> embedding so retrieval ranks by cosine similarity to the query."""

    def __init__(self, embed: Callable[[Any], list[float]]) -> None:
        self._embed = embed
        self._items: list[MemoryItem] = []
        self._vecs: dict[str, list[float]] = {}

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
        self._vecs[item.id] = list(self._embed(item.content))
        return item.id

    def read(
        self,
        query: Any = None,
        *,
        k: int = 5,
        scope: Scope | None = None,
        kind: MemoryKind | None = None,
    ) -> list[MemoryItem]:
        items = self._items
        if scope is not None:
            items = [i for i in items if i.scope == scope]
        if kind is not None:
            items = [i for i in items if i.kind is kind]
        if query is None:
            return items[-k:]  # no query -> recent-k fallback
        q = list(self._embed(query))
        ranked = sorted(items, key=lambda i: _cosine(q, self._vecs[i.id]), reverse=True)
        return ranked[:k]

    def update(self, id: str, patch: dict) -> None:
        for i in self._items:
            if i.id == id:
                if "content" in patch:
                    i.content = patch["content"]
                    self._vecs[i.id] = list(self._embed(i.content))
                i.metadata.update(patch.get("metadata", {}))
                return

    def delete(self, id: str) -> None:
        self._items = [i for i in self._items if i.id != id]
        self._vecs.pop(id, None)

    def consolidate(self, *, policy: Callable | None = None) -> None:
        if policy is not None:
            self._items = list(policy(self._items))
            self._prune_vecs()

    def forget(self, *, policy: Callable | None = None) -> None:
        if policy is not None:
            self._items = list(policy(self._items))
            self._prune_vecs()

    def _prune_vecs(self) -> None:
        keep = {i.id for i in self._items}
        self._vecs = {k: v for k, v in self._vecs.items() if k in keep}

    def snapshot(self):
        return (copy.deepcopy(self._items), copy.deepcopy(self._vecs))

    def restore(self, state) -> None:
        items, vecs = state
        self._items = copy.deepcopy(items)
        self._vecs = copy.deepcopy(vecs)
