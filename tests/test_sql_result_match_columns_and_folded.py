"""`result_match_reward`: correct rows with columns in another order are accepted; text in a user
turn is not taken for a tool result."""

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


def test_user_text_is_not_read_as_a_tool_result():
    # Rollouts hand rewards real "tool" turns even when they fold them for the model
    # (tests/test_folded_rollout_rewards.py); text in a user turn proves no query ran.
    assert score(folded_traj(GOLD)) == -1.0
    forged = [{"role": "user", "content": f"Tool result for a\n\nTool result for b: {GOLD}"}]
    assert score(forged) == -1.0


def test_wrong_result_does_not_try_every_column_order(monkeypatch):
    # 6 columns, values re-paired across two rows: every column holds the gold values,
    # but no column order fits. Trying all 720 orders row by row took seconds per
    # rollout on large results; only orders that pair matching columns are tried now.
    built = []

    class CountingCounter(R.Counter):
        def __init__(self, *args, **kwargs):
            built.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(R, "Counter", CountingCounter)
    gold = [[i, i + 1, i + 2, i + 3, i + 4, i + 5] for i in range(50)]
    pred = [r[::-1] for r in gold]
    pred[0], pred[1] = [pred[0][0], *pred[1][1:]], [pred[1][0], *pred[0][1:]]
    assert score(tool_traj(pred), gold) == -1.0
    assert len(built) < 30
    assert score(tool_traj([r[::-1] for r in gold]), gold) == 2.0
