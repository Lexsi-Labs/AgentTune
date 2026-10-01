"""Structured logger for reward/eval scores that records WHY a score landed
where it did, not just the number.

Every reward or eval function builds up a small `reasons: list[str]` as it
evaluates its conditions (e.g. "no verdict found -> 0.0", "confidence=high ->
+0.3") and calls `log_score(...)` right before each `return`. Output is an
always-on JSONL trail (same convention as `agentic.events.judgment_hook`) plus
an optional forward to an existing `LoggingManager` (W&B/TensorBoard) for
scalar tracking across steps.
"""

import json
import logging
import os
import threading
from datetime import UTC, datetime, timezone  # noqa: F401
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PATH = os.environ.get("AGENTTUNE_SCORE_LOG", "data/scores.jsonl")


class ScoreLogger:
    """Records a score plus the human-readable trail of why it has that value."""

    def __init__(self, output_file: str = _DEFAULT_PATH, logging_manager: Any | None = None):
        self.output_file = output_file
        self.logging_manager = logging_manager
        self._lock = threading.Lock()
        self._last: dict[str, dict[str, Any]] = {}

    def log(
        self,
        name: str,
        score: float,
        reasons: list[str] | None = None,
        components: dict[str, float] | None = None,
        meta: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> None:
        """Record one score. `reasons` is the ordered list of explanations for
        why the score is what it is (append one string per condition/branch
        that contributed, in the order they were evaluated)."""
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "name": name,
            "score": score,
            "reasons": reasons or [],
            "components": components or {},
            "meta": meta or {},
        }
        self._last[name] = record

        try:
            out_dir = os.path.dirname(os.path.abspath(self.output_file))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with self._lock, open(self.output_file, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning(f"ScoreLogger failed to write '{name}': {e}")

        if reasons:
            logger.debug(f"[{name}] score={score}: " + " | ".join(reasons))

        if self.logging_manager is not None:
            try:
                self.logging_manager.log_metrics({name: score}, step=step)
            except Exception as e:
                logger.warning(f"ScoreLogger forward to LoggingManager failed for '{name}': {e}")

    def get_last(self, name: str) -> dict[str, Any] | None:
        """Most recent record logged under `name` (mirrors the old per-file
        `get_last_*_component_scores()` helpers, generically)."""
        return self._last.get(name)


_default_logger: ScoreLogger | None = None
_default_lock = threading.Lock()


def get_score_logger() -> ScoreLogger:
    """Process-wide default ScoreLogger (JSONL only, no LoggingManager)."""
    global _default_logger
    if _default_logger is None:
        with _default_lock:
            if _default_logger is None:
                _default_logger = ScoreLogger()
    return _default_logger


def log_score(
    name: str,
    score: float,
    reasons: list[str] | None = None,
    components: dict[str, float] | None = None,
    meta: dict[str, Any] | None = None,
    step: int | None = None,
) -> None:
    """Convenience call into the process-wide default ScoreLogger."""
    get_score_logger().log(
        name, score, reasons=reasons, components=components, meta=meta, step=step
    )
