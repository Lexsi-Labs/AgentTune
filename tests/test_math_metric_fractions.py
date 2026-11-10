"""MathAccuracyMetric compares a fraction by its value, not by one of its digits."""

import pytest

from agenttune.eval.metrics.math import MathAccuracyMetric


def _accuracy(pred, ref):
    return MathAccuracyMetric().compute([pred], [ref])["math_accuracy"]


@pytest.mark.parametrize(
    "pred,ref",
    [
        (r"\boxed{\frac{1}{2}}", r"\boxed{\frac{1}{3}}"),
        (r"\boxed{\frac{3}{4}}", r"\boxed{\frac{3}{7}}"),
        (r"\boxed{\frac{1}{2}}", r"\boxed{\frac{3}{2}}"),
        (r"\boxed{-\frac{1}{2}}", r"\boxed{\frac{1}{2}}"),
        (r"\boxed{\frac{-1}{2}}", r"\boxed{\frac{-3}{2}}"),
        (r"\boxed{\frac{1.5}{2}}", r"\boxed{\frac{2.5}{2}}"),
        (r"\boxed{1/2}", r"\boxed{3/2}"),
        (r"\boxed{\frac{1}{2}\text{ cup}}", r"\boxed{\frac{3}{2}\text{ cup}}"),
        ("<SOLUTION>1/2</SOLUTION>", "<SOLUTION>3/2</SOLUTION>"),
        ("1/2", "2"),
    ],
)
def test_different_fractions_do_not_match(pred, ref):
    assert _accuracy(pred, ref) == 0.0


@pytest.mark.parametrize(
    "pred,ref",
    [
        (r"\boxed{\frac{1}{2}}", r"\boxed{\frac{1}{2}}"),
        (r"\boxed{\dfrac{1}{2}}", r"\boxed{\tfrac{2}{4}}"),
        (r"\boxed{\frac{1}{2}}", "0.5"),
        (r"\boxed{\frac{1.5}{2}}", "0.75"),
        (r"\boxed{1/2}", "0.5"),
        (r"\boxed{1/2}", "1/2"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"The answer is \boxed{ \frac{3}{4} }", "#### 0.75"),
        ("<SOLUTION>3/4</SOLUTION>", "0.75"),
    ],
)
def test_equal_fractions_match(pred, ref):
    assert _accuracy(pred, ref) == 1.0


@pytest.mark.parametrize(
    "pred,ref",
    [
        (r"The answer is \boxed{42}", "42"),
        (r"\boxed{1,000}", "1000"),
        (r"\boxed{18 \text{ dollars}}", "#### 18"),
        (r"\boxed{\text{5 apples}}", "5"),
        ("<SOLUTION>He eats 3/4 of 20, which is 15</SOLUTION>", "15"),
    ],
)
def test_other_answers_keep_their_last_number(pred, ref):
    assert _accuracy(pred, ref) == 1.0


def test_zero_denominator_is_not_a_fraction():
    assert MathAccuracyMetric()._extract_answer(r"\boxed{\frac{1}{0}}") == "0"


def test_nested_braces_stay_inside_the_box():
    text = r"The answer is 5, so \boxed{\frac{\sqrt{2}}{3}}"
    assert MathAccuracyMetric()._extract_answer(text) == "3"
