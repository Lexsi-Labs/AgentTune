"""
agenttune.agentic.scenarios.collection
=======================================
Lightweight data classes for generated scenarios.

No external dependencies — pure stdlib.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    task: str
    difficulty: int  # 1 (easy) to 5 (hard)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"task": self.task, "difficulty": self.difficulty}
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class ScenarioCollection:
    """
    An ordered collection of Scenario objects with convenience helpers.

    Compatible with the old GeneratedScenarioCollection interface so existing
    code that calls .preview() / .print_difficulty_distribution() continues
    to work unchanged.
    """

    def __init__(self, scenarios: list[Scenario]):
        self._scenarios = list(scenarios)

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    @classmethod
    def from_dicts(cls, dicts: list[dict[str, Any]]) -> ScenarioCollection:
        return cls(
            [
                Scenario(
                    task=d["task"],
                    difficulty=int(d.get("difficulty", 3)),
                    metadata={k: v for k, v in d.items() if k not in ("task", "difficulty")},
                )
                for d in dicts
            ]
        )

    @classmethod
    def from_json(cls, path: str) -> ScenarioCollection:
        with open(path) as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("scenarios", [])
        return cls.from_dicts(items)

    # -------------------------------------------------------------------------
    # Container protocol
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._scenarios)

    def __iter__(self) -> Iterator[Scenario]:
        return iter(self._scenarios)

    def __getitem__(self, idx) -> Scenario:
        return self._scenarios[idx]

    def __repr__(self) -> str:
        return f"ScenarioCollection({len(self._scenarios)} scenarios)"

    # -------------------------------------------------------------------------
    # Serialisation
    # -------------------------------------------------------------------------

    def to_dicts(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._scenarios]

    def to_json(self, path: str, indent: int = 2) -> None:
        with open(path, "w") as f:
            json.dump({"scenarios": self.to_dicts()}, f, indent=indent)

    # -------------------------------------------------------------------------
    # Display helpers
    # -------------------------------------------------------------------------

    def preview(self, n: int = 5) -> None:
        """Print a short preview of the first n scenarios."""
        n = min(n, len(self._scenarios))
        logger.info(f"\n-- Scenario preview (first {n}) " + "-" * 30)
        for i, s in enumerate(self._scenarios[:n]):
            task_preview = s.task[:120].strip()
            ellipsis = "..." if len(s.task) > 120 else ""
            logger.info(f"  {i + 1}. [{s.difficulty}/5] {task_preview}{ellipsis}")
        logger.info()

    def print_difficulty_distribution(self) -> None:
        """Print a simple histogram of difficulty levels."""
        from collections import Counter

        counts = Counter(s.difficulty for s in self._scenarios)
        total = len(self._scenarios)
        logger.info("\n-- Difficulty distribution " + "-" * 35)
        for level in range(1, 6):
            bar_len = int((counts.get(level, 0) / max(total, 1)) * 30)
            bar = "#" * bar_len
            logger.info(f"  {level}/5  {bar:<30}  {counts.get(level, 0):>3}")
        logger.info()

    # -------------------------------------------------------------------------
    # Filtering / slicing
    # -------------------------------------------------------------------------

    def filter_by_difficulty(
        self,
        min_difficulty: int = 1,
        max_difficulty: int = 5,
    ) -> ScenarioCollection:
        """Return a new collection filtered to the given difficulty range."""
        return ScenarioCollection(
            [s for s in self._scenarios if min_difficulty <= s.difficulty <= max_difficulty]
        )

    def tasks(self) -> list[str]:
        """Return just the task strings -- ready to pass as prompts to rollout_fn."""
        return [s.task for s in self._scenarios]
