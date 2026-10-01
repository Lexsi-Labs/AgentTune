"""Regression tests for PassAtKMetric.

Guards against the double-counting bug where a passing candidate incremented
the correct-count `c` twice (once in the pass-check block and again in the
stats-gathering block), so `c` could reach 2n and inflate pass@k.
"""

from types import SimpleNamespace

from agenttune.eval.metrics.code import PassAtKMetric


class _FakeExecutor:
    """Deterministic executor: a candidate passes iff its code == "PASS"."""

    def extract_code(self, code):
        return code

    def execute(self, code, test_cases=None):
        passed = code == "PASS"
        return SimpleNamespace(
            success=True,
            test_passed=passed,
            error="",
            output="",
            test_results=[],
            execution_time=0.0,
        )


def _metric():
    m = PassAtKMetric(k_list=[1, 2])
    m.executor = _FakeExecutor()
    return m


def test_pass_at_k_not_doubled():
    """1 of 2 candidates passing must give pass@1 = 0.5, not 1.0."""
    m = _metric()
    res = m.compute([["PASS", "FAIL"]], [["assert True"]])
    assert res["pass@1"] == 0.5
    assert res["pass@2"] == 1.0


def test_pass_at_k_all_pass():
    m = _metric()
    res = m.compute([["PASS", "PASS"]], [["assert True"]])
    assert res["pass@1"] == 1.0
    assert res["pass@2"] == 1.0


def test_pass_at_k_none_pass():
    m = _metric()
    res = m.compute([["FAIL", "FAIL"]], [["assert True"]])
    assert res["pass@1"] == 0.0
    assert res["pass@2"] == 0.0
