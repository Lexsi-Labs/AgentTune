"""MathAccuracyMetric takes the last conclusion, like its LaTeX and equation patterns."""

import pytest

from agenttune.eval.metrics.math import MathAccuracyMetric


def _accuracy(pred, ref):
    return MathAccuracyMetric().compute([pred], [ref])["math_accuracy"]


@pytest.mark.parametrize(
    "pred,ref",
    [
        ("So the total is 16 eggs. She sells 9 of them. Therefore, she makes $18", "#### 18"),
        ("The total is 16. Then the answer is 18", "18"),
        ("So the result is 3. Wait, let me recheck. So the result is 7", "7"),
        ("Janet makes 5 every day. Bob makes 9 every week.", "9"),
        ("It takes 2 bolts for the frame. It takes 3 bolts for the base.", "3"),
    ],
)
def test_later_conclusion_replaces_earlier_one(pred, ref):
    assert _accuracy(pred, ref) == 1.0


def test_intermediate_value_is_not_the_answer():
    pred = "So the total is 18 eggs. Half are sold. Therefore, she makes $9"
    assert _accuracy(pred, "#### 18") == 0.0


@pytest.mark.parametrize(
    "pred,ref",
    [
        ("Therefore, Janet makes $18", "18"),
        ("The answer is 1,000", "1000"),
        ("THE ANSWER IS 18, not 5", "18"),
        ("#### 72", "#### 72"),
    ],
)
def test_single_conclusion_is_unchanged(pred, ref):
    assert _accuracy(pred, ref) == 1.0


def test_conclusion_drops_thousands_separators():
    assert MathAccuracyMetric()._extract_answer("The answer is 1,000") == "1000"
