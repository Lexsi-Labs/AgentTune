"""FinQA number extraction on tokens that sit next to punctuation or letters."""

import pytest

from agenttune.agentic.rewards.builtin_rewards.finqa import (
    _extract_numbers,
    answer_correctness_reward,
)


def _correctness(gold: str, pred: str) -> float:
    return answer_correctness_reward([None], [f"<answer>{pred}</answer>"], answer=[gold])[0]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("5, costs 6", [5.0, 6.0]),
        ("1,234, up from 1,000", [1234.0, 1000.0]),
        ("$5, then $6", [5.0, 6.0]),
    ],
)
def test_number_before_a_comma_is_kept(text, expected):
    assert _extract_numbers(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("5th", [5.0]),
        ("the 10th year", [10.0]),
        ("the 5TH year", [5.0]),
    ],
)
def test_ordinal_is_not_a_trillion(text, expected):
    assert _extract_numbers(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("-$5", [-5.0]),
        ("-€5", [-5.0]),
        ("a loss of -$1,234.50", [-1234.5]),
        ("(-$5)", [-5.0]),
    ],
)
def test_sign_before_currency_is_kept(text, expected):
    assert _extract_numbers(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$5", [5.0]),
        ("$-5", [-5.0]),
        ("--5", [-5.0]),
        ("-$-5", [-5.0]),
        ("$10-$20", [10.0, 20.0]),
        ("1,234,567", [1234567.0]),
        ("12.5, which is higher", [12.5]),
        ("5.", [5.0]),
        ("(1,234)", [-1234.0]),
        ("($1,234.5)", [-1234.5]),
        ("(5T)", [-5e12]),
        ("12.5%", [0.125]),
        ("2.5B", [2.5e9]),
        ("5K", [5e3]),
        ("5M", [5e6]),
        ("5T", [5e12]),
        ("5bn", [5e9]),
        ("5mn", [5e6]),
        ("5mm", [5e6]),
        ("5tn", [5e12]),
    ],
)
def test_existing_tokens_keep_their_value(text, expected):
    assert _extract_numbers(text) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        ("5", "5, up from 4", 1.0),
        ("1,234", "1,234, up from 1,000", 1.0),
        ("-$5", "-5", 1.0),
        ("-$5", "$5", 0.0),
        ("10", "the 10th year", 1.0),
        ("5T", "the 5th year", 0.0),
    ],
)
def test_answer_correctness_on_these_tokens(gold, pred, expected):
    assert _correctness(gold, pred) == expected
