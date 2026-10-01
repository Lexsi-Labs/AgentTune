"""Crash-safe incremental checkpoints for the LLM-heavy pipeline stages.

The big runs cost real money and real wall-clock. If the process dies at the
3,000th of 6,000 samples, restarting must not regenerate everything from
scratch — that would re-spend the whole stage. This module gives every
completed sample a durable, append-only record the moment it finishes, so a
crash loses only the handful of in-flight calls.

Design:
  - Append-only length-framed pickle log. Each `append` is thread-safe and
    flushed to the OS immediately; a crash mid-write leaves at most a partial
    trailing frame, which `load` skips.
  - The file starts with a length-framed `meta` header (model, prompt
    fingerprint, chunking, seed, hop/endpoint knobs). `load` returns [] when
    the file is absent OR the meta no longer matches — so a changed
    prompt/model/seed never silently resumes against stale artifacts (the same
    staleness rule `stage0_cache.pkl` follows). A mismatched existing file is
    truncated on the next `append`, not appended to.
  - Records are pickled Python objects (QASample / LLMCallRecord / tuples).
    This is our own data written to our own run dir, so pickle is safe here.

Callers:
  - `generate_batch` and `verify_batch` accept `checkpoint_path=`; they load
    completed records, process only the missing samples, and append each
    result as it completes. `run_pipeline` just passes `out_dir`-based paths —
    restarting the same command resumes automatically.
"""

from __future__ import annotations

import os
import pickle
import struct
import threading
from typing import Any

_MAGIC = b"ATCKPT1"
_FRAME = struct.Struct("<I")


def fingerprint(text: str) -> str:
    """Short stable hash used to invalidate checkpoints when the thing they
    depend on (prompt text, input sample set) changes."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def stage_meta(
    stage: str,
    *,
    model: str,
    prompt: str | None = None,
    thresholds: dict[str, Any] | None = None,
    samples=None,
) -> dict[str, Any]:
    """Run knobs a stage's checkpoint must be invalidated on.

    A changed model, prompt template, verification threshold, or input sample
    set must never silently resume against stale artifacts — the same rule
    `stage0_cache.pkl` follows. `samples` fingerprints the ordered
    (sample_id, question, answer) tuples, so a regenerated Stage 2 output
    automatically invalidates the Stage 3 checkpoint.
    """
    m: dict[str, Any] = {"stage": stage, "model": model}
    if prompt is not None:
        m["prompt"] = fingerprint(prompt)
    if thresholds:
        m["thresholds"] = dict(sorted(thresholds.items()))
    if samples is not None:
        seq = "\n".join(f"{s.sample_id}|{s.question}|{s.answer}" for s in samples)
        m["samples"] = fingerprint(seq)
    return m


def _frame(data: bytes) -> bytes:
    return _FRAME.pack(len(data)) + data


class CheckpointLog:
    """Thread-safe, crash-tolerant append-only log of pickled records.

    File layout: `_MAGIC` (8 bytes) + length-framed meta pickle + one
    length-framed pickle per record. `load` reads complete records and
    silently stops at a partial trailing frame (a crash mid-write).
    """

    def __init__(self, path: str, *, meta: dict[str, Any] | None = None):
        self.path = str(path)
        self._meta = meta or {}
        self._lock = threading.Lock()
        self._fh: object | None = None

    def _header(self) -> bytes:
        return _MAGIC + _frame(pickle.dumps(self._meta, protocol=pickle.HIGHEST_PROTOCOL))

    def _ensure_open(self) -> None:
        if self._fh is not None:
            return
        header = self._header()
        if not os.path.exists(self.path):
            with open(self.path, "wb") as f:
                f.write(header)
                f.flush()
        else:
            if self._read_meta() is None:
                # stale or corrupt checkpoint: start over (never append to it)
                with open(self.path, "wb") as f:
                    f.write(header)
                    f.flush()
        self._fh = open(self.path, "ab")

    def _read_meta(self) -> dict[str, Any] | None:
        """Meta from the file header, or None if absent/stale/corrupt."""
        try:
            with open(self.path, "rb") as f:
                magic = f.read(len(_MAGIC))
                if magic != _MAGIC:
                    return None
                head = f.read(_FRAME.size)
                if len(head) < _FRAME.size:
                    return None
                (n,) = _FRAME.unpack(head)
                try:
                    meta = pickle.loads(f.read(n))
                except Exception:
                    return None
                return meta if meta == self._meta else None
        except OSError:
            return None

    def append(self, record: Any) -> None:
        """Record one completed sample. Thread-safe; flushed per record."""
        data = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
        with self._lock:
            self._ensure_open()
            self._fh.write(_frame(data))
            self._fh.flush()

    def load(self) -> list[Any]:
        """All complete records, in order. Partial trailing frame → skipped.

        Returns [] if the file is absent or stale (meta mismatch) — the
        caller then treats the stage as not-yet-started.
        """
        if self._read_meta() is None:
            return []
        try:
            with open(self.path, "rb") as f:
                f.read(len(_MAGIC))  # magic
                head = f.read(_FRAME.size)  # meta frame
                (n,) = _FRAME.unpack(head)
                f.read(n)  # meta payload
                out: list[Any] = []
                while True:
                    head = f.read(_FRAME.size)
                    if len(head) < _FRAME.size:
                        break
                    (n,) = _FRAME.unpack(head)
                    data = f.read(n)
                    if len(data) < n:  # partial trailing frame (crash)
                        break
                    try:
                        out.append(pickle.loads(data))
                    except Exception:
                        break
                return out
        except (OSError, EOFError, struct.error):
            return []

    @property
    def exists(self) -> bool:
        return os.path.exists(self.path) and self._read_meta() is not None

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
