"""SQL rewards: `where 1 = 1` earns no WHERE bonus, tool output written as text is read,
and `result_match_reward` compares the returned rows to the gold rows (issue #11)."""

import pytest

from agenttune.agentic.rewards import REWARD_REGISTRY
from agenttune.agentic.rewards.builtin_rewards import sql as R


def traj(q, rows, final):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "sql_database", "arguments": {"sql_command": q}}}],
        },
        {"role": "tool", "content": rows},
        {"role": "assistant", "content": final},
    ]


def total(t, ans="yes"):
    return sum(
        f([t], answer=[ans])[0] for f in (R.correctness_reward, R.query_reward, R.structure_reward)
    )


HONEST = "select * from customers where country = 'Peru'"


@pytest.mark.parametrize(
    "sql",
    [
        "select * from customers where 1 = 1",
        "select * from customers where 'a' = 'a'",
        "select * from customers where true",
        "select * from customers where 1=1 and 'x' like 'x%' order by id",
        "select * from customers where not null is null",
        "select * from customers",
        "select * from customers where_x",
    ],
)
def test_where_that_names_no_column_is_not_a_filter(sql):
    assert not R._has_real_where(sql)


@pytest.mark.parametrize(
    "sql",
    [
        HONEST,
        "SELECT * FROM t WHERE Country='Peru'",
        "select * from t where id in (select id from u where x = 1)",
        "select * from t where name = 'where 1=1' and age > 3",
        "select * from t\nwhere\n  amount > 10 group by country",
    ],
)
def test_where_that_names_a_column_is_a_filter(sql):
    assert R._has_real_where(sql)


def test_where_one_equals_one_no_longer_ties_the_honest_query():
    honest = total(traj(HONEST, [[7, "Ana", "Peru"]], "*yes*"))
    hack = total(traj("select * from customers where 1 = 1", [[1, "Bo", "Chile"]], "*yes*"))
    assert honest == pytest.approx(3.1)
    assert hack == pytest.approx(2.2)


def test_query_reward_reads_tool_output_written_as_text():
    # Rollouts store str(result) in the tool message.
    hit = R.query_reward([traj(HONEST, "[(7, 'Ana', 'Peru')]", "*yes*")], answer=["yes"])[0]
    miss = R.query_reward([traj(HONEST, "[]", "*no*")], answer=["yes"])[0]
    error = R.query_reward([traj(HONEST, "{'error': 'no such table'}", "*no*")], answer=["no"])[0]
    assert hit == pytest.approx(2.4)
    assert miss == pytest.approx(-1.1)
    assert error == pytest.approx(-1.6)


def test_tool_output_parsing():
    assert R._tool_output("[(1, 'a')]") == [(1, "a")]
    assert R._tool_output("((1,),)") == [(1,)]
    assert R._tool_output("{'error': 'x'}") == {"error": "x"}
    assert R._tool_output("Error: boom") is None
    assert R._tool_output("42") is None
    assert R._tool_output([[1]]) == [[1]]


GOLD = [[7, "Ana", "Peru"], [9, "Luz", "Peru"]]


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([[9, "Luz", "Peru"], [7, "Ana", "Peru"]], 2.0),  # order ignored
        ("[(7, 'Ana', 'Peru'), (9, 'Luz', 'Peru')]", 2.0),  # text, tuples
        ([[7, "Ana", "Peru"]], -1.0),  # a row missing
        ([[7, "Ana", "Peru"], [7, "Ana", "Peru"], [9, "Luz", "Peru"]], -1.0),  # duplicates count
        ([[1, "Bo", "Chile"], [7, "Ana", "Peru"], [9, "Luz", "Peru"]], -1.0),  # where id > 0
        ("{'error': 'syntax error'}", -1.0),
        ("Error: boom", -1.0),
    ],
)
def test_result_match_reward(rows, expected):
    score = R.result_match_reward([traj(HONEST, rows, "*yes*")], gold_rows=[GOLD])[0]
    assert score == expected


def test_result_match_reward_uses_the_last_tool_result():
    t = traj("select * from customers", [[1, "Bo", "Chile"]], "")
    t += traj(HONEST, GOLD, "*yes*")
    assert R.result_match_reward([t], gold_rows=[GOLD]) == [2.0]
    assert R.result_match_reward([list(reversed(t))], gold_rows=[GOLD]) == [-1.0]


def test_result_match_reward_without_gold_or_tool_result():
    t = traj(HONEST, GOLD, "*yes*")
    assert R.result_match_reward([t, t]) == [0.0, 0.0]
    assert R.result_match_reward([t, t], gold_rows=[GOLD, None]) == [2.0, 0.0]
    no_tool = [{"role": "assistant", "content": "*yes*"}]
    assert R.result_match_reward([no_tool], gold_rows=[GOLD]) == [-1.0]
    assert R.result_match_reward([[]], gold_rows=[[]]) == [-1.0]
    # Gold of no rows matches an empty result.
    assert R.result_match_reward([traj(HONEST, "[]", "*no*")], gold_rows=[[]]) == [2.0]


def test_result_match_reward_is_registered():
    assert REWARD_REGISTRY["result_match_reward"] is R.result_match_reward
