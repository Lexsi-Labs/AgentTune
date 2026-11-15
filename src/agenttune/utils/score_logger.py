"""Structured logger for reward/eval scores that records WHY a score landed
where it did, not just the number.

Every reward or eval function builds up a small `reasons: list[str]` as it
evaluates its conditions (e.g. "no verdict found -> 0.0", "confidence=high ->
+0.3") and calls `log_score(...)` right before each `return`. The last record
per name is kept in memory (`get_last`). A JSONL trail is opt-in: set
`AGENTTUNE_SCORE_LOG=path/to/scores.jsonl`, or pass `output_file` to your own
`ScoreLogger`. Records can also be forwarded to an existing `LoggingManager`
(W&B/TensorBoard) for scalar tracking across steps.
"""

import json
import logging
import os
import threading
from datetime import UTC, datetime, timezone  # noqa: F401
from typing import Any

logger = logging.getLogger(__name__)


class ScoreLogger:
    """Records a score plus the human-readable trail of why it has that value."""

    def __init__(self, output_file: str | None = None, logging_manager: Any | None = None):
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

        if self.output_file:
            self._write(name, record)

        if reasons:
            logger.debug(f"[{name}] score={score}: " + " | ".join(reasons))

        if self.logging_manager is not None:
            try:
                self.logging_manager.log_metrics({name: score}, step=step)
            except Exception as e:
                logger.warning(f"ScoreLogger forward to LoggingManager failed for '{name}': {e}")

    def _write(self, name: str, record: dict[str, Any]) -> None:
        try:
            out_dir = os.path.dirname(os.path.abspath(self.output_file))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with self._lock, open(self.output_file, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning(f"ScoreLogger failed to write '{name}': {e}")

    def get_last(self, name: str) -> dict[str, Any] | None:
        """Most recent record logged under `name` (mirrors the old per-file
        `get_last_*_component_scores()` helpers, generically)."""
        return self._last.get(name)


_default_logger: ScoreLogger | None = None
_default_lock = threading.Lock()


def get_score_logger() -> ScoreLogger:
    """Process-wide default ScoreLogger. It writes a JSONL trail only when
    `AGENTTUNE_SCORE_LOG` names a file (read when the logger is first used)."""
    global _default_logger
    if _default_logger is None:
        with _default_lock:
            if _default_logger is None:
                _default_logger = ScoreLogger(os.environ.get("AGENTTUNE_SCORE_LOG") or None)
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
