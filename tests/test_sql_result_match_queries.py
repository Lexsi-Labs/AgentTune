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


# ── gold_sql: row order, and comparison databases ────────────────────────────


def make_db(path, rows):
    import sqlite3

    con = sqlite3.connect(path)
    con.execute("create table singer(name text, age int)")
    con.executemany("insert into singer values (?, ?)", rows)
    con.execute("create view adults as select name, age from singer where age >= 18")
    con.commit()
    con.close()
    return str(path)


@pytest.fixture
def dbs(tmp_path):
    """The training database has nobody aged 30; the comparison databases do, and one
    holds other people altogether."""
    return [
        make_db(tmp_path / "b.db", [("Joe", 31), ("Ann", 45), ("Bob", 30)]),
        make_db(tmp_path / "c.db", [("Zed", 50), ("Kim", 12)]),
    ]


GOLD_SQL = "select name, age from singer where age > 30"
UNION_HACK = (
    "select 'Joe', 31 union all select 'Ann', 45 union all select name, age from singer where 0"
)
EXISTS_HACK = (
    "select 'Joe', 31 where exists (select 1 from singer) "
    "union all select 'Ann', 45 where exists (select 1 from singer)"
)


@pytest.mark.parametrize(
    "sql, alone, with_dbs",
    [
        (GOOD, 2.0, 2.0),
        ("select age, name from singer where age between 31 and 200", 2.0, 2.0),
        ("select name, age from singer where age >= 30", 2.0, -1.0),  # issue #20 (b)
        (UNION_HACK, 2.0, -1.0),
        (EXISTS_HACK, 2.0, -1.0),
        ("select name, age from singer where name in ('Joe', 'Ann')", 2.0, -1.0),
    ],
)
def test_comparison_databases(dbs, sql, alone, with_dbs):
    conv = turns((sql, GOLD))
    assert score(conv) == alone
    assert score(conv, gold_sql=[GOLD_SQL], db_paths=[dbs]) == with_dbs


def test_comparison_databases_are_read_only(dbs):
    import sqlite3

    conv = turns(("delete from singer", "[]"))
    gold_sql = "select name from singer where age > 100"
    assert score(conv, gold=[], gold_sql=[gold_sql], db_paths=[dbs]) == -1.0
    assert sqlite3.connect(dbs[0]).execute("select count(*) from singer").fetchone() == (3,)


def test_runaway_query_is_stopped(dbs, monkeypatch):
    monkeypatch.setattr(R, "_QUERY_SECONDS", 0.2)
    forever = (
        "with recursive c(x) as (select 1 union all select x + 1 from c) "
        "select name, age from singer where age > 30 and (select count(*) from c) > 0"
    )
    assert score(turns((forever, GOLD)), gold_sql=[GOLD_SQL], db_paths=[dbs]) == -1.0


def test_broken_gold_sql_is_reported(dbs):
    with pytest.raises(ValueError, match="gold_sql"):
        score(turns((GOOD, GOLD)), gold_sql=["select nope from singer"], db_paths=[dbs])
    with pytest.raises(ValueError, match="gold_sql"):
        score(turns((GOOD, GOLD)), db_paths=[dbs])


@pytest.mark.parametrize(
    "gold_sql, expected",
    [
        ("select name, age from singer where age > 30 order by age desc", -1.0),
        ("select name, age from singer where age > 30", 2.0),
        (None, 2.0),
    ],
)
def test_row_order_counts_when_gold_orders(gold_sql, expected):
    gold = [["Ann", 45], ["Joe", 31]]  # in the gold query's order
    assert score(turns((GOOD, [["Joe", 31], ["Ann", 45]])), gold, gold_sql=[gold_sql]) == expected
    assert score(turns((GOOD, gold)), gold, gold_sql=[gold_sql]) == 2.0


@pytest.mark.parametrize(
    "sql, ordered",
    [
        ("select a from t order by a", True),
        ("SELECT a FROM t\nORDER\n BY a DESC LIMIT 3", True),
        ("select a from t union select b from u order by 1", True),
        ("select a from (select a from t order by a)", False),  # subquery only
        ("select a, rank() over (order by b) from t", False),  # window
        ("select 'order by' from t", False),  # in a literal
        ("select a from t -- order by a", False),  # in a comment
        ('select "order by" from t', False),  # quoted identifier
        ("select a from t", False),
    ],
)
def test_top_level_order_by(sql, ordered):
    assert R._has_top_level_order_by(sql) is ordered


def test_rows_from_a_tool_that_runs_no_sql_do_not_match():
    # echo(text) hands back whatever the model wrote; its rows prove nothing.
    rows = "[('Joe', 31), ('Ann', 45)]"
    conv = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "echo", "arguments": {"text": rows}}}],
        },
        {"role": "tool", "name": "echo", "content": rows},
    ]
    assert score(conv) == -1.0
    assert score(turns((GOOD, GOLD)) + conv) == 2.0  # the SQL result before it still counts


def last_reason():
    from agenttune.utils.score_logger import get_score_logger

    return get_score_logger().get_last("sql.result_match_reward")["reasons"][0]


WIDE = ", ".join(f"printf('%.*c', 999999, 'x') as c{i}" for i in range(40))


@pytest.mark.parametrize(
    "sql, reason",
    [
        # endless rows, streamed from the table
        (
            "with recursive c(n, a) as (select name, age from singer where age > 30 "
            "union all select n, a from c) select n, a from c",
            "more rows",
        ),
        ("select name, zeroblob(1000000000) from singer where age > 30", "fails"),
        # SQLite's printf gives NULL past the length limit
        ("select name, printf('%.*c', 1000000000, 'x') from singer where age > 30", "differs"),
        # 40 columns of almost 1 MB each, per row
        (f"select {WIDE} from singer where age > 30", "columns"),
        # right width, rows of almost 1 MB each
        (
            "select name, printf('%.*c', 999999, 'x') from singer s, "
            "(with recursive c(x) as (select 1 union all select x + 1 from c) select x from c)",
            "more rows",
        ),
    ],
)
def test_huge_results_stop_early(dbs, sql, reason):
    import resource
    import sys
    import time

    def peak_kib():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024 if sys.platform == "darwin" else peak

    before, start = peak_kib(), time.perf_counter()
    assert score(turns((sql, GOLD)), gold_sql=[GOLD_SQL], db_paths=[dbs]) == -1.0
    assert reason in last_reason()
    assert time.perf_counter() - start < 2.0
    assert peak_kib() - before < 100 * 1024  # under 100 MiB


def test_large_gold_still_bounds_the_query(tmp_path):
    db = make_db(tmp_path / "big.db", [(f"p{i}", i) for i in range(1000)])
    gold_sql = "select name, age from singer"
    gold = [[f"p{i}", i] for i in range(1000)]
    assert score(turns((gold_sql, gold)), gold, gold_sql=[gold_sql], db_paths=[[db]]) == 2.0
    big = "select name, printf('%.*c', 999999, 'x') from singer"
    assert score(turns((big, gold)), gold, gold_sql=[gold_sql], db_paths=[[db]]) == -1.0
    assert "more data" in last_reason()
