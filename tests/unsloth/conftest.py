"""Gate for the Unsloth-backend test suite.

These tests must NEVER run in the same pytest process as the main suite:
Unsloth monkey-patches transformers' model classes (Qwen2Attention,
Qwen2RotaryEmbedding, ...) at the class level the moment it's imported, and
that patch is process-global -- any later test in the same process that
loads a plain (non-Unsloth) model breaks. The parent tests/conftest.py sets
TRL_ONLY_MODE=1 by default for exactly this reason.

Run this directory on its own, in its own process, after the main suite:
    RUN_UNSLOTH_TESTS=1 pytest tests/unsloth -q --no-cov

Collecting it as part of a full `pytest` run is intentionally a no-op (each
test skips) unless RUN_UNSLOTH_TESTS=1 is set, so accidentally including
tests/unsloth in a full-suite invocation can't silently re-poison the
process.
"""

import os

import pytest

if os.environ.get("RUN_UNSLOTH_TESTS", "0") != "1":
    collect_ignore_glob = ["*"]
else:
    # Explicit opt-in: this process is meant to test Unsloth, so undo the
    # parent conftest's TRL-only defaults before anything imports unsloth.
    for _var in ("TRL_ONLY_MODE", "DISABLE_UNSLOTH_FOR_TRL", "PURE_TRL_MODE"):
        os.environ.pop(_var, None)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_UNSLOTH_TESTS", "0") != "1":
        skip = pytest.mark.skip(
            reason="Set RUN_UNSLOTH_TESTS=1 to run the Unsloth backend suite (run in its own process, separate from the main suite)"
        )
        for item in items:
            if "unsloth" in str(item.fspath):
                item.add_marker(skip)
