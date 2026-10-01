"""SelfHealLoop — the full self-healing orchestrator (Phase 6, on top of detection).

The spine already does GPU/network-free failure DETECTION (``Project.heal`` runs the
real closed-loop ``FailureDetector``). This module continues the loop end to end:

    detected failures  ->  classify root cause  ->  generate corrective examples  ->  retrain

The classify and generate stages are litellm-dependent in the real code, so this
orchestrator takes them as INJECTED plain callables:

    classifier:  list[Failure]          -> list[ClassifiedFailure]
    generator:   list[ClassifiedFailure] -> list[TrainingExample]

That makes the whole loop testable with fakes (no litellm, no GPU, no network) while
the real deployment wraps the existing async ``FailureClassifier`` /
``TrainingExampleGenerator`` (see ``as_sync_classifier`` / ``as_sync_generator``).

Import hygiene: this module imports ONLY the litellm-free contracts + stdlib. It never
imports ``failure_classifier`` / ``training_example_generator`` (which pull litellm) —
the caller constructs those and hands them in. The loop emits no side effects beyond the
summary it returns (no files, no events, no global state); a ``run_on(project)`` call
does trigger the project's OWN heal events, which is the project's behavior, not the
loop's.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from agenttune.decide.closed_loop.contracts import (
    ClassifiedFailure,
    Failure,
    TrainingExample,
)

Classifier = Callable[[list[Failure]], list[ClassifiedFailure]]
Generator = Callable[[list[ClassifiedFailure]], list[TrainingExample]]


def _training_example_to_row(ex: TrainingExample) -> dict | None:
    """Turn one ``TrainingExample`` into a DPO/BCO preference row, or ``None`` if it
    has no derivable preference pair (then the loop skips it).

    Schema: ``{prompt, chosen, rejected}`` — exactly what the adapter-only retrain path
    (DPO/BCO) consumes per the ``TrainingExample`` docstring. If the example carries only
    the multi-completion form (Path A's generator), ``derive_preference_from_completions``
    bridges best-reward -> chosen / worst -> rejected in place.
    """
    if not ex.has_preference_pair():
        ex.derive_preference_from_completions()
    if not ex.has_preference_pair():
        return None
    return {"prompt": ex.prompt, "chosen": ex.chosen, "rejected": ex.rejected}


def build_dataset(examples: list[TrainingExample]) -> list[dict]:
    """Build the preference dataset the retrain step trains on, skipping unpairable
    examples. Kept public so callers/tests can preview the rows GPU-free."""
    rows = []
    for ex in examples:
        row = _training_example_to_row(ex)
        if row is not None:
            rows.append(row)
    return rows


class SelfHealLoop:
    """Drive detect -> classify -> generate -> retrain with the litellm stages injected.

    Parameters
    ----------
    classifier:
        ``list[Failure] -> list[ClassifiedFailure]``. In tests a fake; in production
        wrap the async ``FailureClassifier`` via :func:`as_sync_classifier`.
    generator:
        ``list[ClassifiedFailure] -> list[TrainingExample]``. In tests a fake; in
        production wrap the async ``TrainingExampleGenerator`` via
        :func:`as_sync_generator`.
    trainer_factory:
        Optional. Called as ``trainer_factory(train_dataset=rows, **trainer_kwargs)``
        and expected to return an object with ``.train() -> dict`` — the SAME contract
        ``Project.train`` uses (e.g. agenttune's TRL trainers), keeping this loop
        GPU/dep-free. If ``None``, the loop still classifies + generates + builds the
        dataset and reports, but does not train.
    retraining_trigger:
        Optional ``decide.closed_loop.retraining_trigger.RetrainingTrigger`` (Path B).
        If given, every generated example is added to its buffer and training is
        gated on ``.should_trigger()`` — without one, the loop trains as soon as it
        has a non-empty dataset (the original, ungated behavior; unchanged for
        backward compatibility). This closes a real gap: ``Project.heal()``/
        ``SelfHealLoop`` used to train immediately with no volume/drift check.
    evaluate_deploy:
        Optional zero-arg callable returning a ``decide.closed_loop.deployment_gate
        .GateDecision`` (or anything with an ``.approved`` bool), called once after a
        successful train. The loop does not deploy anything itself (it never has —
        see the module docstring); this only surfaces the gate's verdict on the
        summary dict (``deploy_decision``) so the caller can decide whether to keep,
        roll back, or redeploy the trained artifact.
    """

    def __init__(
        self,
        classifier: Classifier,
        generator: Generator,
        *,
        trainer_factory: Callable[..., Any] | None = None,
        retraining_trigger: Any | None = None,
        evaluate_deploy: Callable[[], Any] | None = None,
    ):
        self.classifier = classifier
        self.generator = generator
        self.trainer_factory = trainer_factory
        self.retraining_trigger = retraining_trigger
        self.evaluate_deploy = evaluate_deploy

    def run(self, failures: list[Failure], **trainer_kwargs) -> dict:
        """Run classify -> generate -> (trigger check) -> (retrain) -> (deploy gate)
        on already-detected ``failures``.

        Returns a summary dict: counts (``n_failures``, ``n_classified``,
        ``n_generated``, ``n_dataset_rows``, ``n_skipped``), the ``trained`` flag,
        the ``train_result`` (or ``None``), ``trigger_checked``/``trigger_fired``/
        ``trigger_reason`` (Path B gating — see ``retraining_trigger`` above),
        ``deploy_decision`` (Path B gate verdict, or ``None``), and the
        ``classified`` / ``generated`` / ``dataset`` artifacts.
        """
        failures = list(failures)
        classified = list(self.classifier(failures))
        generated = list(self.generator(classified))
        dataset = build_dataset(generated)

        trigger_checked = self.retraining_trigger is not None
        trigger_fired, trigger_reason = True, "no_trigger_configured"
        if self.retraining_trigger is not None:
            for ex in generated:
                self.retraining_trigger.buffer.add(ex)
            trigger_fired, trigger_reason = self.retraining_trigger.should_trigger()

        trained = False
        train_result = None
        deploy_decision = None
        if self.trainer_factory is not None and dataset and trigger_fired:
            trainer = self.trainer_factory(train_dataset=dataset, **trainer_kwargs)
            train_result = trainer.train()
            trained = True
            if self.evaluate_deploy is not None:
                deploy_decision = self.evaluate_deploy()

        return {
            "n_failures": len(failures),
            "n_classified": len(classified),
            "n_generated": len(generated),
            "n_dataset_rows": len(dataset),
            "n_skipped": len(generated) - len(dataset),
            "trained": trained,
            "train_result": train_result,
            "trigger_checked": trigger_checked,
            "trigger_fired": trigger_fired,
            "trigger_reason": trigger_reason,
            "deploy_decision": deploy_decision,
            "classified": classified,
            "generated": generated,
            "dataset": dataset,
        }

    def run_on(self, project, *, detector=None, max_revisits: int = 3, **trainer_kwargs) -> dict:
        """Detect failures on ``project`` (via the real ``Project.heal``) then run the
        full loop over them. ``project.heal`` emits the project's own lifecycle events;
        the loop adds no side effects of its own."""
        failures = project.heal(detector=detector, max_revisits=max_revisits)
        return self.run(failures, **trainer_kwargs)


# ---- production adapters: wrap the existing async litellm stages as plain callables ----
# These take an ALREADY-CONSTRUCTED instance so this module never imports the
# litellm-dependent modules; the caller constructs FailureClassifier /
# TrainingExampleGenerator (importing litellm at THEIR call site, not here).


def as_sync_classifier(classifier) -> Classifier:
    """Adapt an async ``FailureClassifier`` (``.classify_batch`` coroutine) into the
    plain ``list[Failure] -> list[ClassifiedFailure]`` callable the loop expects."""
    return lambda failures: asyncio.run(classifier.classify_batch(list(failures)))


def as_sync_generator(generator) -> Generator:
    """Adapt an async ``TrainingExampleGenerator`` (``.generate_batch`` coroutine) into
    the plain ``list[ClassifiedFailure] -> list[TrainingExample]`` callable the loop
    expects."""
    return lambda classified: asyncio.run(generator.generate_batch(list(classified)))
