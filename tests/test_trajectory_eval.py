from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator


def test_tac_score():
    evaluator = TrajectoryEvaluator()

    # Strings are assumed valid — 2/2 = 1.0
    traj_perfect = {"tool_calls": ["search", "read"]}
    assert evaluator._calculate_tac(traj_perfect) == 1.0

    # Empty tool_calls — no evidence of error → 1.0
    assert evaluator._calculate_tac({"tool_calls": []}) == 1.0

    # Dict with unknown tool name (not in tool_schemas) → counted but not valid → 0.0
    traj_unknown = {"tool_calls": [{"name": "no_such_tool", "arguments": {}}]}
    assert evaluator._calculate_tac(traj_unknown) == 0.0

    # Dict without 'name' key (malformed call) → unknown → 0.0
    traj_malformed = {"tool_calls": [{"error_type": "nonexistent_argument"}]}
    assert evaluator._calculate_tac(traj_malformed) == 0.0

    # Mixed: one valid string + one unknown dict → 1 valid / 2 total = 0.5
    traj_mixed = {"tool_calls": ["search", {"name": "bad_tool", "arguments": {}}]}
    assert evaluator._calculate_tac(traj_mixed) == 0.5


def test_ter_score():
    evaluator = TrajectoryEvaluator()

    # Novel outputs
    traj_novel = {"tool_outputs": ["data1", "data2", "data3"]}
    assert evaluator._calculate_ter(traj_novel) == 1.0

    # Completely duplicate outputs (agent stuck in a loop calling the same tool)
    traj_stuck = {"tool_outputs": ["same_data", "same_data", "same_data"]}
    # 1 unique out of 3 total = 0.333
    assert abs(evaluator._calculate_ter(traj_stuck) - 0.333) < 0.01
