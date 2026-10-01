"""Phase 6 (heal, full loop) — the SelfHealLoop orchestrator drives the closed loop
detect -> classify -> generate -> retrain end to end, with the litellm-dependent
stages (classify/generate) INJECTED as plain callables so the whole thing is testable
with fakes (no litellm, no GPU, no network).

Discriminating test: feed REAL `Failure` objects (from `Project.heal()` on a looping
spine trajectory, exactly as test_heal.py builds them) through the loop with a FAKE
classifier + FAKE generator + capturing FAKE trainer, and assert the pipeline THREADS
them: Failure -> ClassifiedFailure -> TrainingExample -> the trainer's dataset, with the
per-trajectory identity preserved and the summary counts closing arithmetically.
"""

import pytest
from _real_backends import real_litellm_response

from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.heal_loop import SelfHealLoop, as_sync_classifier
from agenttune.agentic.project import Project
from agenttune.decide.closed_loop.contracts import (
    ClassifiedFailure,
    Failure,
    TrainingExample,
)
from agenttune.decide.closed_loop.failure_classifier import FailureClassifier

# Loads Qwen2.5-0.5B via _real_backends (3-5GB RSS + ~1GB download);
# not for the 7.8GB CPU CI runner. Runs under -m qwen_e2e.
pytestmark = pytest.mark.qwen_e2e

# Note: unlike the other files in this folder, most of test_heal_loop.py is
# deliberately left untouched. The whole point of this file is that
# SelfHealLoop is decoupled from litellm — classifier/generator are injected
# as plain callables, and two tests explicitly assert litellm is NEVER called
# (test_heal_loop_module_does_not_import_litellm,
# test_full_loop_makes_no_litellm_call). Swapping those to a real model would
# defeat the tests they are. Instead, the threading tests below get a REAL
# classifier (FailureClassifier backed by Qwen2.5-0.5B-Instruct via
# as_sync_classifier) in place of _fake_classifier, proving the production
# adapter really works end-to-end with a real model — the generator stays the
# lightweight fake since its job here is dataset-shaping, not model behavior
# (real-model generation is already covered by
# tests/agentic_real/decide/closed_loop/test_training_example_generator.py).


async def _real_acompletion(**kwargs):
    prompt = kwargs["messages"][-1]["content"]
    return real_litellm_response(prompt, max_new_tokens=64)


def _real_classifier(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "acompletion", _real_acompletion, raising=False)
    return as_sync_classifier(FailureClassifier(model_name="qwen2.5-0.5b-instruct"))


# ---- spine fixtures (mirror test_heal.py) ----


def _looping_log(tool="search", n=4):
    log = EventLog(tier="light")
    for _ in range(n):
        log.append(Event(EventKind.TOOL_CALL, {"action": {"name": tool, "arguments": {}}}))
        log.append(Event(EventKind.TOOL_RESULT, {"output": "same"}))
    return log


# ---- fakes: plain callables, no litellm ----


def _fake_classifier(failures):
    """list[Failure] -> list[ClassifiedFailure], 1:1, identity preserved."""
    return [
        ClassifiedFailure(
            failure=f,
            root_cause="loop_collapse" if f.failure_type == "loop_collapse" else "wrong_tool",
            confidence=0.9,
            analysis=f"fake analysis for {f.trajectory_id}",
        )
        for f in failures
    ]


def _fake_generator(classified):
    """list[ClassifiedFailure] -> list[TrainingExample], 1:1, preference pair preset."""
    out = []
    for cf in classified:
        out.append(
            TrainingExample(
                trajectory_id=cf.failure.trajectory_id,
                original_failure_type=cf.failure.failure_type,
                root_cause=cf.root_cause,
                prompt=[{"role": "user", "content": f"ctx:{cf.failure.trajectory_id}"}],
                chosen=[{"role": "assistant", "content": "corrected"}],
                rejected=[{"role": "assistant", "content": "the-failed-one"}],
            )
        )
    return out


class _CapturingTrainer:
    """Mirrors the trainer contract Project.train uses: built with train_dataset=...,
    exposes .train() -> dict. Captures the dataset it was handed."""

    last_dataset = None

    def __init__(self, train_dataset=None, **kw):
        self.train_dataset = train_dataset
        _CapturingTrainer.last_dataset = train_dataset

    def train(self):
        return {"status": "trained", "n": len(self.train_dataset)}


# ---- tests ----


def test_pipeline_threads_failure_to_classified_to_example_to_dataset(monkeypatch):
    """The crux: real Failures -> the trainer's dataset, identity threaded through."""
    p = Project()
    p.add_trajectory(_looping_log(tool="search", n=4))
    failures = p.heal(max_revisits=3)
    loop_failures = [f for f in failures if f.failure_type == "loop_collapse"]
    assert loop_failures, "spine must produce a real loop_collapse Failure"

    _CapturingTrainer.last_dataset = None
    loop = SelfHealLoop(
        _real_classifier(monkeypatch), _fake_generator, trainer_factory=_CapturingTrainer
    )
    summary = loop.run(loop_failures)

    # counts close arithmetically
    assert summary["n_failures"] == len(loop_failures)
    assert summary["n_classified"] == len(loop_failures)
    assert summary["n_generated"] == len(loop_failures)
    assert summary["n_dataset_rows"] == len(loop_failures)
    assert summary["trained"] is True

    # artifacts are the real typed objects
    assert all(isinstance(c, ClassifiedFailure) for c in summary["classified"])
    assert all(isinstance(e, TrainingExample) for e in summary["generated"])

    # THREADING: the ClassifiedFailure wraps the SAME Failure object
    assert summary["classified"][0].failure is loop_failures[0]
    # the TrainingExample carries that failure's trajectory_id
    tid = loop_failures[0].trajectory_id
    assert summary["generated"][0].trajectory_id == tid

    # the trainer actually received the dataset built from the generated examples
    captured = _CapturingTrainer.last_dataset
    assert captured is not None and len(captured) == len(loop_failures)
    row = captured[0]
    assert row["prompt"] == [{"role": "user", "content": f"ctx:{tid}"}]
    assert row["chosen"] == [{"role": "assistant", "content": "corrected"}]
    assert row["rejected"] == [{"role": "assistant", "content": "the-failed-one"}]
    assert summary["train_result"] == {"status": "trained", "n": len(loop_failures)}


def test_run_without_trainer_still_classifies_and_generates(monkeypatch):
    """No trainer_factory -> loop classifies + generates + reports, no training."""
    fs = [Failure(trajectory_id="t1", failure_type="loop_collapse", failed_stage_name="search")]
    loop = SelfHealLoop(_real_classifier(monkeypatch), _fake_generator)  # no trainer
    summary = loop.run(fs)

    assert summary["n_classified"] == 1
    assert summary["n_generated"] == 1
    assert summary["n_dataset_rows"] == 1  # dataset is still built...
    assert summary["trained"] is False  # ...but not trained
    assert summary["train_result"] is None
    assert isinstance(summary["generated"][0], TrainingExample)


def test_run_on_project_drives_heal_then_loop(monkeypatch):
    """run_on(project) calls project.heal() and threads the failures through."""
    p = Project()
    p.add_trajectory(_looping_log(tool="search", n=4))
    _CapturingTrainer.last_dataset = None
    loop = SelfHealLoop(
        _real_classifier(monkeypatch), _fake_generator, trainer_factory=_CapturingTrainer
    )
    summary = loop.run_on(p, max_revisits=3)
    assert summary["n_failures"] >= 1
    assert summary["n_dataset_rows"] == summary["n_generated"]
    assert summary["trained"] is True


def test_dataset_builder_bridges_completions_to_pair():
    """A generator that emits only completions/rewards (Path A form) is bridged to a
    DPO pair via derive_preference_from_completions — no preset chosen/rejected."""

    def completions_gen(classified):
        return [
            TrainingExample(
                trajectory_id=cf.failure.trajectory_id,
                original_failure_type=cf.failure.failure_type,
                root_cause=cf.root_cause,
                prompt=[{"role": "user", "content": "p"}],
                completions=[
                    [{"role": "assistant", "content": "good"}],
                    [{"role": "assistant", "content": "bad"}],
                ],
                rewards=[1.0, -1.0],
            )
            for cf in classified
        ]

    fs = [Failure(trajectory_id="t1", failure_type="wrong_tool", failed_stage_name="fetch")]
    _CapturingTrainer.last_dataset = None
    loop = SelfHealLoop(_fake_classifier, completions_gen, trainer_factory=_CapturingTrainer)
    summary = loop.run(fs)
    assert summary["n_dataset_rows"] == 1
    row = _CapturingTrainer.last_dataset[0]
    assert row["chosen"] == [{"role": "assistant", "content": "good"}]
    assert row["rejected"] == [{"role": "assistant", "content": "bad"}]


def test_dataset_builder_skips_unpairable_and_counts_close():
    """An example with no derivable preference pair is skipped; counts still close."""

    def unpairable_gen(classified):
        return [
            TrainingExample(
                trajectory_id=cf.failure.trajectory_id,
                original_failure_type=cf.failure.failure_type,
                root_cause=cf.root_cause,
                prompt=[{"role": "user", "content": "p"}],
                is_negative_only=True,  # no chosen/rejected, no completions
            )
            for cf in classified
        ]

    fs = [Failure(trajectory_id="t1", failure_type="wrong_tool", failed_stage_name="fetch")]
    loop = SelfHealLoop(_fake_classifier, unpairable_gen, trainer_factory=_CapturingTrainer)
    summary = loop.run(fs)
    assert summary["n_generated"] == 1
    assert summary["n_dataset_rows"] == 0
    assert summary["n_skipped"] == 1
    # nothing to train on -> no training attempted
    assert summary["trained"] is False


def test_heal_loop_module_does_not_import_litellm():
    """heal_loop itself must not import litellm (the litellm-bound stages are injected,
    never imported here). NB: the closed_loop package __init__ imports litellm eagerly,
    so litellm ends up in sys.modules regardless — that's a pre-existing package fact,
    not a call. What we own: heal_loop's own namespace stays litellm-free."""
    import agenttune.agentic.heal_loop as hl

    assert "litellm" not in getattr(hl, "__dict__", {})
    assert "failure_classifier" not in getattr(hl, "__dict__", {})
    assert "training_example_generator" not in getattr(hl, "__dict__", {})


def test_full_loop_makes_no_litellm_call(monkeypatch):
    """The discriminating no-network guarantee: run the WHOLE loop with fakes and prove
    litellm is never CALLED (acompletion/completion). A real call would explode."""
    import litellm

    def _boom(*a, **k):  # any litellm call is a hard failure
        raise AssertionError("litellm was called during the self-heal loop")

    monkeypatch.setattr(litellm, "acompletion", _boom, raising=False)
    monkeypatch.setattr(litellm, "completion", _boom, raising=False)

    fs = [Failure(trajectory_id="t1", failure_type="loop_collapse", failed_stage_name="search")]
    loop = SelfHealLoop(_fake_classifier, _fake_generator, trainer_factory=_CapturingTrainer)
    summary = loop.run(fs)
    assert summary["trained"] is True and summary["n_dataset_rows"] == 1


def test_as_sync_classifier_awaits_the_real_async_stage():
    """The production adapter bridges an async `.classify_batch` coroutine into the plain
    sync callable the loop expects. Discriminating: the fake's coroutine records the exact
    failures it received and returns derived objects; the adapter must actually await it and
    return that result (not the coroutine, not None)."""
    from agenttune.agentic.heal_loop import as_sync_classifier

    seen = {}

    class _AsyncClassifier:
        async def classify_batch(self, failures):
            seen["failures"] = failures
            return [
                ClassifiedFailure(
                    failure=f, root_cause=f.failure_type, confidence=0.9, analysis="async"
                )
                for f in failures
            ]

    fs = [Failure(trajectory_id="t7", failure_type="tool_crash", failed_stage_name="run")]
    classify = as_sync_classifier(_AsyncClassifier())
    out = classify(fs)

    assert seen["failures"] == fs  # the coroutine actually ran with our input
    assert [c.failure for c in out] == fs  # and its awaited result threads back
    assert out[0].root_cause == "tool_crash"


def test_as_sync_generator_awaits_the_real_async_stage():
    """Same bridge for `.generate_batch` — adapter awaits the coroutine and returns rows."""
    from agenttune.agentic.heal_loop import as_sync_generator

    class _AsyncGenerator:
        async def generate_batch(self, classified):
            return [
                TrainingExample(
                    trajectory_id=c.failure.trajectory_id,
                    original_failure_type=c.failure.failure_type,
                    root_cause=c.root_cause,
                    prompt=[{"role": "user", "content": "recover"}],
                    chosen=[{"role": "assistant", "content": "fixed"}],
                    rejected=[{"role": "assistant", "content": "looped"}],
                )
                for c in classified
            ]

    cf = [
        ClassifiedFailure(
            failure=Failure(
                trajectory_id="t8", failure_type="loop_collapse", failed_stage_name="s"
            ),
            root_cause="loop_collapse",
            confidence=0.8,
            analysis="x",
        )
    ]
    generate = as_sync_generator(_AsyncGenerator())
    out = generate(cf)

    assert [e.trajectory_id for e in out] == ["t8"]
    assert out[0].chosen[0]["content"] == "fixed"


def test_export():
    from agenttune.agentic import SelfHealLoop as S

    assert S is SelfHealLoop
