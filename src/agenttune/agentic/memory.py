"""Agent memory — pluggable memory subsystem (Phase 4 of the spine).

Taxonomy (CoALA 2309.02427): working / episodic / semantic / procedural / entity.
BaseMemory exposes CRUD plus injectable, trainable `consolidate`/`forget` policies
(Memory-R1 2508.19828 pattern) and snapshot/restore for rollout reset/replay.
Drivers: InContextMemory (ReAct scratchpad) and TrajectoryStore (RL replay +
distillation dataset). Vector/graph drivers land later. Pure-Python, additive.
"""

from __future__ import annotations

import copy
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenttune.agentic.events import EventLog


class MemoryKind(Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    ENTITY = "entity"


@dataclass(frozen=True)
class Scope:
    agent_id: str = "default"
    namespace: str = "default"  # per-agent private vs shared pool


@dataclass
class MemoryItem:
    content: Any
    kind: MemoryKind = MemoryKind.EPISODIC
    scope: Scope = field(default_factory=Scope)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)


class BaseMemory(ABC):
    @abstractmethod
    def write(self, item: MemoryItem, *, scope: Scope | None = None) -> str: ...

    @abstractmethod
    def read(
        self,
        query: Any = None,
        *,
        k: int = 5,
        scope: Scope | None = None,
        kind: MemoryKind | None = None,
    ) -> list[MemoryItem]: ...

    @abstractmethod
    def update(self, id: str, patch: dict) -> None: ...

    @abstractmethod
    def delete(self, id: str) -> None: ...

    # Injectable, trainable policies (default no-ops). A policy is
    # Callable[[list[MemoryItem]], list[MemoryItem]] returning the items to keep.
    def consolidate(self, *, policy: Callable | None = None) -> None:
        return None

    def forget(self, *, policy: Callable | None = None) -> None:
        return None

    @abstractmethod
    def snapshot(self): ...

    @abstractmethod
    def restore(self, state) -> None: ...


class InContextMemory(BaseMemory):
    """List-backed working/episodic memory — the ReAct scratchpad. Keeps raw items
    (per 2605.12978, raw episodes are a safe default over lossy consolidation)."""

    def __init__(self) -> None:
        self._items: list[MemoryItem] = []

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
        return items[-k:]  # most-recent-k

    def update(self, id: str, patch: dict) -> None:
        for i in self._items:
            if i.id == id:
                i.content = patch.get("content", i.content)
                i.metadata.update(patch.get("metadata", {}))
                return

    def delete(self, id: str) -> None:
        self._items = [i for i in self._items if i.id != id]

    def consolidate(self, *, policy: Callable | None = None) -> None:
        if policy is not None:
            self._items = list(policy(self._items))

    def forget(self, *, policy: Callable | None = None) -> None:
        if policy is not None:
            self._items = list(policy(self._items))

    def snapshot(self):
        return copy.deepcopy(self._items)

    def restore(self, state) -> None:
        self._items = copy.deepcopy(state)


class TrajectoryStore(BaseMemory):
    """Episodic replay store of full-tier EventLogs; emits distillation datasets."""

    def __init__(self) -> None:
        self._items: list[MemoryItem] = []

    def write(self, item: MemoryItem, *, scope: Scope | None = None) -> str:
        if not isinstance(item.content, EventLog):
            raise TypeError("TrajectoryStore stores MemoryItem whose content is an EventLog.")
        item = MemoryItem(
            content=item.content,
            kind=MemoryKind.EPISODIC,
            scope=scope or item.scope,
            id=item.id,
            metadata=item.metadata,
        )
        self._items.append(item)
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
        return items[-k:]

    def update(self, id: str, patch: dict) -> None:
        for i in self._items:
            if i.id == id:
                if "content" in patch:
                    i.content = patch["content"]
                i.metadata.update(patch.get("metadata", {}))
                return

    def delete(self, id: str) -> None:
        self._items = [i for i in self._items if i.id != id]

    def as_dataset(self, fmt: str = "sft") -> list[dict]:
        """Emit SFT/DPO rows from stored full-tier trajectories (distillation bridge)."""
        rows: list[dict] = []
        for item in self._items:
            log: EventLog = item.content
            if log.tier == "full":
                rows.extend(log.as_dataset_rows(fmt))
        return rows

    def snapshot(self):
        return list(self._items)

    def restore(self, state) -> None:
        self._items = list(state)
