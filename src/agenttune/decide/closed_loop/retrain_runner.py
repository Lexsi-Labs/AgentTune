"""
Background Retrain Runner — Path B, Week 2 (design + tests only)
===============================================================

Non-blocking execution wrapper around a retrain job.  This is the *interface*
that the closed-loop runner will use in Week 3; here we deliver the design and
its tests, not the full wiring into the live loop.

Contract
--------
- ``start(job_fn, *args, **kwargs)`` launches ``job_fn`` on a background thread
  and returns IMMEDIATELY (non-blocking) — control goes straight back to the
  caller's loop.
- While the job runs, ``is_running()`` is True and the bound
  ``RetrainingTrigger.retrain_in_progress`` Event is SET, so the trigger will
  not fire a second retrain (no double-retrain).
- When the job finishes (success OR failure), the runner posts a "done" signal:
  the ``done`` Event is SET, the trigger's flag is CLEARED, and an optional
  ``on_done`` callback fires with the ``RetrainResult``.
- ``wait(timeout)`` blocks for completion (used by tests / synchronous callers).

The runner deliberately knows nothing about *what* the job does — it only
manages the thread lifecycle and the signalling.  The actual training is the
``job_fn`` (e.g. ``retrain_config.run_retrain``), injected by the caller.  This
keeps the runner testable with a trivial stub job and no GPU.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrainResult:
    """Outcome of a background retrain job."""

    success: bool
    result: Any = None  # whatever job_fn returned (e.g. trainer stats)
    error: str | None = None  # str(exception) on failure
    started_at: str | None = None  # ISO 8601
    finished_at: str | None = None
    duration_s: float = 0.0


class BackgroundRetrainer:
    """Runs one retrain job at a time on a background thread.

    Bind a :class:`RetrainingTrigger` so the runner can drive its
    ``retrain_in_progress`` flag automatically (set on start, clear on done),
    which is what prevents the trigger from firing a second retrain.
    """

    def __init__(
        self,
        trigger: Any | None = None,
        on_done: Callable[[RetrainResult], None] | None = None,
    ) -> None:
        self._trigger = trigger
        self._on_done = on_done
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._lock = threading.RLock()
        self._result: RetrainResult | None = None

    # -- lifecycle ----------------------------------------------------------

    def start(self, job_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
        """Launch ``job_fn`` on a background thread. Returns immediately.

        Returns False (and does nothing) if a retrain is already running — this
        is the no-double-retrain guard at the runner level, complementing the
        trigger's own ``retrain_in_progress`` check.
        """
        with self._lock:
            if self.is_running():
                logger.warning("BackgroundRetrainer.start ignored: a retrain is already running.")
                return False

            self._done.clear()
            self._result = None
            if self._trigger is not None:
                self._trigger.mark_retrain_started()

            self._thread = threading.Thread(
                target=self._run,
                args=(job_fn, args, kwargs),
                name="BackgroundRetrainer",
                daemon=True,
            )
            self._thread.start()
            logger.info("BackgroundRetrainer: retrain job started (non-blocking).")
            return True

    def _run(self, job_fn: Callable[..., Any], args: tuple, kwargs: dict) -> None:
        started = datetime.now(UTC)
        t0 = time.monotonic()
        try:
            out = job_fn(*args, **kwargs)
            result = RetrainResult(success=True, result=out)
        except Exception as exc:  # noqa: BLE001 — we record every failure
            logger.error("BackgroundRetrainer job failed: %s\n%s", exc, traceback.format_exc())
            result = RetrainResult(success=False, error=str(exc))
        finally:
            t1 = time.monotonic()
            finished = datetime.now(UTC)

        result.started_at = started.isoformat()
        result.finished_at = finished.isoformat()
        result.duration_s = round(t1 - t0, 3)

        # Store result and clear the retrain-in-progress flag first
        with self._lock:
            self._result = result
            if self._trigger is not None:
                self._trigger.mark_retrain_finished()

        logger.info(
            "BackgroundRetrainer: retrain %s in %.2fs.",
            "succeeded" if result.success else "FAILED",
            result.duration_s,
        )

        # Run on_done BEFORE signalling _done so that wait() callers always
        # see the side-effects of on_done (e.g. last_decision) already set.
        if self._on_done is not None:
            try:
                self._on_done(result)
            except Exception as cb_exc:  # noqa: BLE001
                logger.error("BackgroundRetrainer on_done callback raised: %s", cb_exc)

        # Signal done only after on_done has finished
        self._done.set()

    # -- status / sync ------------------------------------------------------

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_done(self) -> bool:
        """True once a started job has posted its done signal."""
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> RetrainResult | None:
        """Block until the job finishes (or ``timeout`` elapses).

        Returns the :class:`RetrainResult`, or ``None`` if it timed out.
        """
        finished = self._done.wait(timeout=timeout)
        if not finished:
            return None
        with self._lock:
            return self._result

    @property
    def result(self) -> RetrainResult | None:
        with self._lock:
            return self._result
