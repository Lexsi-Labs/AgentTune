"""Regression tests for PassAtKMetric.

Guards against the double-counting bug where a passing candidate incremented
the correct-count `c` twice (once in the pass-check block and again in the
stats-gathering block), so `c` could reach 2n and inflate pass@k.

REAL executor: uses agenttune's own `SafeCodeExecutor` (real subprocess-isolated
Python execution) with real candidate source code and real `assert` test cases,
in place of the `_FakeExecutor` stand-in from tests/eval/test_pass_at_k.py that
just string-compared code to the literal "PASS"/"FAIL".
"""

from agenttune.eval.metrics.code import PassAtKMetric

_CORRECT_CANDIDATE = "def add(a, b):\n    return a + b\n"
_WRONG_CANDIDATE = "def add(a, b):\n    return a - b\n"
_TEST_CASES = ["assert add(2, 3) == 5"]


def _metric():
    # Default executor is already the real SafeCodeExecutor — no fake to inject.
    return PassAtKMetric(k_list=[1, 2])


def test_pass_at_k_not_doubled():
    """1 of 2 candidates passing must give pass@1 = 0.5, not 1.0."""
    m = _metric()
    res = m.compute([[_CORRECT_CANDIDATE, _WRONG_CANDIDATE]], [_TEST_CASES])
    assert res["pass@1"] == 0.5
    assert res["pass@2"] == 1.0


def test_pass_at_k_all_pass():
    m = _metric()
    res = m.compute([[_CORRECT_CANDIDATE, _CORRECT_CANDIDATE]], [_TEST_CASES])
    assert res["pass@1"] == 1.0
    assert res["pass@2"] == 1.0


def test_pass_at_k_none_pass():
    m = _metric()
    res = m.compute([[_WRONG_CANDIDATE, _WRONG_CANDIDATE]], [_TEST_CASES])
    assert res["pass@1"] == 0.0
    assert res["pass@2"] == 0.0
