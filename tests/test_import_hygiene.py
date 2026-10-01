"""Import-hygiene regression tests.

Guards against a circular import: importing TrajectoryEvaluator first used to
fail because eval.agentic.trajectory_eval -> decide.closed_loop.contracts
triggered closed_loop/__init__ -> full_loop -> training_example_generator ->
back to trajectory_eval (still initializing).

Each test spawns a fresh interpreter so module-import caching cannot mask the
cycle.
"""

import subprocess
import sys


def _fresh_import(statement: str):
    return subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
    )


def test_trajectory_evaluator_imports_first():
    r = _fresh_import("from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator")
    assert r.returncode == 0, f"circular import:\n{r.stderr}"


def test_training_example_generator_imports_first():
    r = _fresh_import(
        "from agenttune.decide.closed_loop.training_example_generator "
        "import TrainingExampleGenerator"
    )
    assert r.returncode == 0, f"circular import:\n{r.stderr}"
