"""`result_match_reward`: correct rows with columns in another order are accepted, and tool results
folded into user turns (tool-blind chat templates such as Tiny Aya) are read."""

import pytest

from agenttune.agentic.rewards.builtin_rewards import sql as R

GOLD = [["Joe", 31], ["Ann", 45]]
CALL = {
    "function": {"name": "run_sql", "arguments": {"sql_command": "select name, age from singer"}}
}


def tool_traj(rows):
    return [
        {"role": "assistant", "content": "", "tool_calls": [CALL]},
        {"role": "tool", "content": rows},
    ]


def folded_traj(*results):
    content = "\n\n".join(
        f"Tool result for run_sql: {r}\nUse this result to answer the original question."
        for r in results
    )
    return [
        {"role": "assistant", "content": "", "tool_calls": [CALL]},
        {"role": "user", "content": content},
    ]


def score(traj, gold=GOLD):
    return R.result_match_reward([traj], gold_rows=[gold])[0]


def test_columns_in_another_order_are_accepted():
    assert score(tool_traj([[31, "Joe"], [45, "Ann"]])) == 2.0


@pytest.mark.parametrize(
    "rows",
    [
        [[31, "Ann"], [45, "Joe"]],  # columns swapped but values re-paired: wrong
        [[31, "Joe"]],  # missing row
        [["Joe", 31], [45, "Ann"]],  # per-row sort would wrongly accept this mix
    ],
)
def test_wrong_rows_stay_rejected(rows):
    assert score(tool_traj(rows)) == -1.0


def test_per_row_sort_trap_with_duplicates():
    gold = [["a", "b"], ["a", "b"]]
    assert score(tool_traj([["a", "b"], ["b", "a"]]), gold) == -1.0


def test_folded_tool_result_is_read():
    assert score(folded_traj(GOLD)) == 2.0
    assert score(folded_traj([["Joe", 31]])) == -1.0


def test_last_folded_result_counts():
    assert score(folded_traj([["x", 1]], GOLD)) == 2.0
    assert score(folded_traj(GOLD, [["x", 1]])) == -1.0
