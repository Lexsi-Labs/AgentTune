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
    full_row_checks = []
    same_rows = R._same_rows

    def counting(pred, gold, ordered):
        if pred and len(pred[0]) > 1:
            full_row_checks.append(1)
        return same_rows(pred, gold, ordered)

    monkeypatch.setattr(R, "_same_rows", counting)
    gold = [[i, i + 1, i + 2, i + 3, i + 4, i + 5] for i in range(50)]
    pred = [r[::-1] for r in gold]
    pred[0], pred[1] = [pred[0][0], *pred[1][1:]], [pred[1][0], *pred[0][1:]]
    assert score(tool_traj(pred), gold) == -1.0
    assert len(full_row_checks) <= 3
    assert score(tool_traj([r[::-1] for r in gold]), gold) == 2.0


@pytest.mark.parametrize(
    "rows, expected",
    [
        ([{"name": "Joe", "age": 31}, {"name": "Ann", "age": 45}], 2.0),  # dict rows
        ([{"age": 31, "name": "Joe"}, {"age": 45, "name": "Ann"}], 2.0),  # in another order
        ([{"name": "Joe", "age": 31}], -1.0),
        ([["Joe", {"a": [1]}], ["Ann", 45]], -1.0),  # unhashable cell, no TypeError
    ],
)
def test_dict_and_nested_rows(rows, expected):
    assert score(tool_traj(rows)) == expected


def test_nested_cells_compare_by_value():
    gold = [["Joe", {"tags": ["a", "b"], "n": 1}]]
    assert score(tool_traj([["Joe", {"n": 1, "tags": ["a", "b"]}]]), gold) == 2.0
    assert score(tool_traj([["Joe", {"n": 1, "tags": ["b", "a"]}]]), gold) == -1.0


@pytest.mark.parametrize(
    "pred, gold, expected",
    [
        ([["x", 0.6000000000000001]], [["x", 0.6]], 2.0),  # float noise
        ([["x", 0.1 + 0.2], ["y", 1.0]], [["y", 1], ["x", 0.3]], 2.0),  # 1.0 == 1, any order
        ([["x", 0.3000001]], [["x", 0.3]], 2.0),  # within abs 1e-6
        ([["x", 0.300002]], [["x", 0.3]], -1.0),
        ([["x", 1e12 + 1e-4]], [["x", 1e12]], 2.0),  # within rel 1e-9
        ([["x", 1000000001]], [["x", 1000000000]], -1.0),  # ints are exact
        ([[0.30000000000000004, "x"]], [["x", 0.3]], 2.0),  # float column reordered
        ([["x", [0.1 + 0.2]]], [["x", [0.3]]], 2.0),  # nested
    ],
)
def test_float_tolerance(pred, gold, expected):
    assert score(tool_traj(pred), gold) == expected
