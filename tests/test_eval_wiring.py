"""Phase 6 (eval wiring) — the discriminating test the spine has to pass.

The spine's thesis is "wrap, don't rewrite": an ``EventLog`` must round-trip into
what the REAL ``TrajectoryEvaluator`` consumes and produce the SAME score as calling
that evaluator directly. If this needs hand-massaging, the abstraction is wrong.

These use only the evaluator's programmatic (no-model, no-network) metrics.
"""

from agenttune.agentic.events import EventLog
from agenttune.agentic.trajectory.dataset import Step, Trajectory
from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator


def _evaluator():
    return TrajectoryEvaluator(model_name="none", api_base=None)


# ---- Task 1: EventLog.to_eval_dict round-trips into the real evaluator ----


def test_to_eval_dict_roundtrips_through_real_evaluator():
    """dict -> EventLog -> dict must score identically to dict -> evaluator."""
    traj = {
        "tool_calls": [
            {"name": "search", "arguments": {"q": "a"}},
            {"name": "search", "arguments": {"q": "a"}},  # duplicate
            {"name": "read", "arguments": {"p": "f"}},
        ],
        "tool_outputs": ["hit", "hit", "contents"],
    }
    ev = _evaluator()
    direct_arr = ev._calculate_arr(traj)
    direct_ter = ev._calculate_ter(traj)

    log = EventLog.from_eval_dict(traj)
    round_tripped = log.to_eval_dict()

    assert ev._calculate_arr(round_tripped) == direct_arr
    assert ev._calculate_ter(round_tripped) == direct_ter


# ---- Task 2: the REAL Trajectory reaches the REAL evaluator through EventLog ----


def test_full_trajectory_projects_into_real_evaluator():
    """A real agentic Trajectory -> full EventLog -> eval dict scores the same as
    building the eval dict straight from that trajectory's steps. This is the
    load-bearing claim: the trainer-side artifact and the eval-side consumer meet
    through EventLog with no rewrite of either."""
    traj = Trajectory(
        task="t",
        steps=[
            Step(
                step_number=0,
                state="s",
                action={"name": "search", "arguments": {"q": "x"}},
                observation="doc1",
                thought="think",
            ),
            Step(
                step_number=1,
                state="s",
                action={"name": "search", "arguments": {"q": "x"}},
                observation="doc1",
                thought="again",
            ),  # duplicate call + output
            Step(
                step_number=2,
                state="s",
                action={"name": "finish", "arguments": {}},
                observation="done",
                thought="stop",
            ),
        ],
        reward=1.0,
        final_response="answer",
    )
    # reference dict built directly from the same steps
    reference = {
        "tool_calls": [s.action for s in traj.steps],
        "tool_outputs": [s.observation for s in traj.steps],
    }
    ev = _evaluator()

    log = EventLog.from_trajectory(traj)
    via_spine = log.to_eval_dict()

    assert via_spine["tool_calls"] == reference["tool_calls"]
    assert via_spine["tool_outputs"] == reference["tool_outputs"]
    assert ev._calculate_arr(via_spine) == ev._calculate_arr(reference)
    assert ev._calculate_ter(via_spine) == ev._calculate_ter(reference)


# ---- Task 3: Project.evaluate_agentic drives the real evaluator, GPU-free ----


def test_project_evaluate_agentic_uses_real_evaluator():
    """The spine's own runtime traces (Project.infer) reach the real evaluator's
    programmatic metrics with no model call — proving end-to-end wiring, not a toy."""
    from agenttune.agentic.harness import DictToolHarness
    from agenttune.agentic.project import Project, agentic_metrics
    from agenttune.agentic.strategy import ReActStrategy

    # a policy that calls search twice (a duplicate) then finishes
    calls = iter(
        [
            {"name": "search", "arguments": {"q": "x"}},
            {"name": "search", "arguments": {"q": "x"}},
            {"name": "finish", "arguments": {"answer": "done"}},
        ]
    )
    strat = ReActStrategy(policy=lambda state: next(calls))
    harness = DictToolHarness({"search": lambda q="": "hit"})
    p = Project(strategy=strat, harness=harness)

    report = p.evaluate_agentic(["find x"])
    assert report["n"] == 1
    # arr (action repetition rate) is a real metric; the duplicate search makes it > 0
    assert report["metrics"]["arr"] > 0.0
    assert set(report["metrics"]).issuperset({"tac", "ter", "arr"})
    # a lifecycle event was emitted for the agentic eval
    assert any(ev.kind == "eval_done" and ev.stage == "evaluate_agentic" for ev in p.events())

    # agentic_metrics on a single log matches what the report aggregated
    single = agentic_metrics(p.trajectories[0])
    assert single["arr"] == report["per_trajectory"][0]["arr"]
