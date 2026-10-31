"""Numeric rewards must read 1,234 as one number, not as 1 and 234."""

import pytest

from agenttune.agentic.rewards.builtin_rewards.sql import reward_correct_answer
from agenttune.agentic.rewards.builtin_rewards.use_case import numerical_match_reward


def _numerical(gold: str, pred: str) -> float:
    return numerical_match_reward([None], [f"<answer>{pred}</answer>"], answer=[gold])[0]


@pytest.mark.parametrize(
    "gold,pred",
    [
        ("1,234", "1,999"),
        ("$2,500", "$2,900"),
        ("1,000,000", "1,000"),
        ("-1,234", "-1,999"),
    ],
)
def test_numerical_match_does_not_reward_a_different_number(gold, pred):
    assert _numerical(gold, pred) == 0.0


@pytest.mark.parametrize(
    "gold,pred",
    [
        ("1,234", "1234"),
        ("1234", "1,234"),
        ("1,234.50", "1234.5"),
        ("1,000,000", "1000000"),
        ("-1,234", "-1234"),
    ],
)
def test_numerical_match_accepts_the_same_number_with_or_without_commas(gold, pred):
    assert _numerical(gold, pred) == 1.0


def test_numerical_match_tolerance_uses_the_whole_number():
    assert _numerical("1,000", "1,040") == 0.5


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        ("5", "5", 1.0),
        ("5", "5.", 1.0),
        ("5", "5, up from 4", 1.0),
        ("12.5", "12.5", 1.0),
        ("10", "10,20", 1.0),
        ("10", "11", 0.0),
    ],
)
def test_numerical_match_keeps_plain_numbers(gold, pred, expected):
    assert _numerical(gold, pred) == expected


@pytest.mark.parametrize(
    "gold,text",
    [
        ("1234", "the total is 1,234"),
        ("1234.5", "the total is 1,234.50"),
        ("1000000", "the total is 1,000,000 rows"),
        ("-1234", "the total is -1,234"),
    ],
)
def test_sql_reward_reads_the_whole_number(gold, text):
    assert reward_correct_answer([text], answer=[gold]) == [1.5]


@pytest.mark.parametrize(
    "gold,text",
    [
        ("234", "the total is 1,234"),
        ("1", "the total is 1,234"),
    ],
)
def test_sql_reward_does_not_match_a_fragment_of_the_number(gold, text):
    assert reward_correct_answer([text], answer=[gold]) == [0.0]


def test_sql_reward_keeps_plain_numbers():
    assert reward_correct_answer(["there are 42 rows, 7 of them new"], answer=["7"]) == [1.5]
    assert reward_correct_answer(["there are 42 rows"], answer=["7"]) == [0.0]
    assert reward_correct_answer(["the values are 10,20"], answer=["20"]) == [1.5]
