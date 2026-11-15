"""`result_match_reward` scores the rows of the last query that ran, and only when that
query read the database: rows a query writes out itself (literals, VALUES) prove nothing."""

import pytest

from agenttune.agentic.rewards.builtin_rewards import sql as R

GOLD = [["Joe", 31], ["Ann", 45]]
GOOD = "select name, age from singer where age > 30"


def call(sql):
    return {"function": {"name": "run_sql", "arguments": {"sql_command": sql}}}


def turns(*steps):
    """steps: (sql, tool result) pairs, one assistant call and one tool turn each."""
    out = []
    for sql, result in steps:
        out.append({"role": "assistant", "content": "", "tool_calls": [call(sql)]})
        out.append({"role": "tool", "name": "run_sql", "content": result})
    return out


def score(conv, gold=GOLD, **kw):
    return R.result_match_reward([""], gold_rows=[gold], conversations=[conv], **kw)[0]


ERROR = "{'error': 'no such column: agee'}"


@pytest.mark.parametrize(
    "steps, expected",
    [
        ([(GOOD, GOLD), ("select agee from singer", ERROR)], 2.0),  # failed look-around after
        ([("select agee from singer", ERROR), (GOOD, GOLD)], 2.0),
        ([(GOOD, GOLD), ("select name from singer", [["Joe"]])], -1.0),  # last that ran
        ([("select agee from singer", ERROR)], -1.0),
        ([(GOOD, GOLD), ("select agee from singer", "Error: boom")], 2.0),  # text error
    ],
)
def test_last_query_that_ran_counts(steps, expected):
    assert score(turns(*steps)) == expected


@pytest.mark.parametrize(
    "sql",
    [
        "select 'Joe', 31 union all select 'Ann', 45",
        "values ('Joe', 31), ('Ann', 45)",
        "with t(a, b) as (values ('Joe', 31), ('Ann', 45)) select * from t",
        "with singer as (select 'Joe' as name, 31 as age union all select 'Ann', 45) "
        "select * from singer",
        "select * from (select 'Joe', 31 union all select 'Ann', 45)",
        "select 'Joe', 31 -- from singer\nunion all select 'Ann', 45",
        "select 'Joe', 31 from sqlite_master union all select 'Ann', 45 from sqlite_schema",
        "select 'Joe', 31 where exists (select 1 from pragma_table_list) union all select 'Ann', 45",
    ],
)
def test_rows_written_by_the_query_do_not_match(sql):
    assert score(turns((sql, GOLD))) == -1.0


@pytest.mark.parametrize(
    "sql",
    [
        GOOD,
        "select age, name from singer where age > 30",
        # literals mixed into a query that reads the table
        "select name, 31 from singer where name = 'Joe' union all "
        "select name, age from singer where name = 'Ann'",
        "select * from adults",  # a view
        "SELECT name, age FROM main.singer WHERE age > 30;",
        "select 'Joe'::text, 31",  # not SQLite: cannot tell, the rows decide
        "...",  # no query text to check: the rows decide
    ],
)
def test_queries_that_read_tables_still_match(sql):
    assert score(turns((sql, GOLD))) == 2.0


def test_result_is_paired_with_its_own_call():
    # Two calls in one turn: the literal select's rows must not borrow the real query.
    conv = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "run_sql", "arguments": {"sql_command": "select 'x', 1"}}},
                {"function": {"name": "nosuch", "arguments": {"sql_command": GOOD}}},
            ],
        },
        {"role": "tool", "name": "run_sql", "content": GOLD},
        {"role": "tool", "name": "nosuch", "content": "{'error': \"Tool 'nosuch' not found\"}"},
    ]
    assert score(conv) == -1.0
