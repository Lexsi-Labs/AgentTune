"""
Integration test: Path A's real TrainingExampleGenerator -> Path B retrain.

This proves the Path A -> Path B handshake with the COLLEAGUE'S ACTUAL
generator code (not synthetic fixtures). The only things stubbed are the
external boundaries: the LLM call (litellm.acompletion) and the replay
validator's subprocess — exactly what you'd mock in CI. Everything in between
(his _process_single / generate_batch building TrainingExample objects with
completions+rewards, then our bridge converting them to DPO/BCO datasets) runs
for real.

Run:
    pytest tests/decide/closed_loop/test_integration_path_a_to_b.py -v
"""

import types
from unittest.mock import AsyncMock, patch

import pytest

from agenttune.decide.closed_loop.contracts import ClassifiedFailure, Failure
from agenttune.decide.closed_loop.retrain_config import (
    examples_to_bco_dataset,
    examples_to_dpo_dataset,
)
from agenttune.decide.closed_loop.retraining_trigger import RetrainingTrigger, TriggerConfig

# ---------------------------------------------------------------------------
# Helpers — build a ClassifiedFailure exactly as Path A's classifier emits
# ---------------------------------------------------------------------------


def _classified(root_cause="wrong_tool", tid="traj_1"):
    failure = Failure(
        trajectory_id=tid,
        failure_type="tool_crash",
        failed_stage_name="calculator_tool",
        context={"prompt": "Count active subscriptions", "tool_called": "calculator"},
        error_message="SyntaxError: invalid syntax in math expression",
    )
    return ClassifiedFailure(
        failure=failure,
        root_cause=root_cause,
        confidence=0.9,
        analysis="Agent used the calculator tool for a SQL task.",
    )


def _make_generator(outcomes=None):
    """Real generator + validator, with LLM + subprocess boundaries stubbed."""
    from agenttune.decide.closed_loop.replay_validator import ReplayValidator
    from agenttune.decide.closed_loop.training_example_generator import (
        TrainingExampleGenerator,
    )

    validator = ReplayValidator()
    # Stub the subprocess replay → (is_valid, tool_outputs). We derive the
    # result deterministically FROM THE COMPLETION TEXT (unique per completion),
    # so distinct completions always score distinct rewards — independent of how
    # generate_batch's asyncio.gather interleaves validator calls. The generator
    # computes reward = TAC + TER (− penalty if invalid); we make TER vary by
    # returning a duplicate-controlled output list keyed to the text, so each
    # example gets a derivable best/worst preference pair.
    if outcomes is not None:
        validator.validate_example_with_trace = AsyncMock(side_effect=outcomes)
    else:

        def _digits(text):
            d = "".join(ch for ch in str(text) if ch.isdigit())
            return int(d) if d else len(str(text))

        async def _varied(example, completion_text):
            n = _digits(completion_text)
            # More duplicates → lower TER → distinct reward per distinct text.
            dup = n % 4
            outputs = ["row"] * dup + [f"u{n}"]  # TER = 1/(dup+1)
            return (True, outputs)

        validator.validate_example_with_trace = AsyncMock(side_effect=_varied)

    gen = TrainingExampleGenerator(validator=validator, model_name="stub/model")
    return gen


def _fake_completion(text: str):
    """Shape a litellm.acompletion return value: resp.choices[0].message.content."""
    msg = types.SimpleNamespace(content=text)
    choice = types.SimpleNamespace(message=msg)
    return types.SimpleNamespace(choices=[choice])


# ---------------------------------------------------------------------------
# The integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_generator_output_feeds_dpo_and_bco():
    """His generator emits completions/rewards only; our bridge must turn that
    into usable DPO pairs and BCO rows."""
    gen = _make_generator()

    # Each LLM call returns a distinct "corrected action". The trailing number
    # makes each completion's text unique (the stubbed validator scores reward
    # deterministically from the text, so distinct texts → distinct rewards).
    fake_texts = iter(
        [
            '{"tool": "execute_sql", "query": "SELECT COUNT(*) FROM subscriptions", "v": 1}',
            '{"tool": "execute_sql", "query": "SELECT COUNT(id) FROM subscriptions", "v": 2}',
            '{"tool": "execute_sql", "query": "SELECT COUNT(*) FROM subs", "v": 3}',
            '{"tool": "execute_sql", "query": "SELECT COUNT(1) FROM subs", "v": 6}',
        ]
    )

    async def fake_acompletion(**kwargs):
        return _fake_completion(next(fake_texts, '{"tool": "execute_sql"}'))

    with patch("litellm.acompletion", new=fake_acompletion):
        examples = await gen.generate_batch([_classified(root_cause="wrong_tool")])

    # 1. His generator produced real TrainingExample objects.
    assert len(examples) == 1
    ex = examples[0]
    assert ex.has_completions()  # completions + rewards populated
    assert ex.has_preference_pair()  # generator sets chosen/rejected when rewards differ
    assert len(ex.completions) == 2  # wrong_tool → N=2 per his _determine_num_completions
    # Generator sets chosen/rejected directly (best correction vs. real failed action
    # or lowest-reward completion). The DPO converter uses that pair as-is.
    assert ex.has_preference_pair()

    # 2. DPO converter uses the generator-set pair (or bridges if absent).
    dpo_rows = examples_to_dpo_dataset(list(examples))
    assert len(dpo_rows) == 1
    assert set(dpo_rows[0]) == {"prompt", "chosen", "rejected"}
    assert dpo_rows[0]["chosen"] and dpo_rows[0]["rejected"]

    # 3. Our BCO converter also yields labelled rows from the same output.
    #    (Re-generate: the DPO bridge mutated ex.chosen/rejected in place, so
    #     BCO now sees the preference form — still valid, one desirable + one undesirable.)
    bco_rows = examples_to_bco_dataset(list(examples))
    labels = sorted(r["label"] for r in bco_rows)
    assert True in labels and False in labels


@pytest.mark.asyncio
async def test_real_generator_output_flows_through_buffer_and_trigger():
    """Full handshake: his generator → our buffer → trigger fires → drains."""
    # 3 failures × 2 completions (wrong_tool → N=2) = 6 validator calls.
    # Alternate valid/failed so each example has one high- and one low-reward
    # completion → a derivable preference pair.
    outcomes = [(True, ["a", "b", "c"]), (False, [])] * 3
    gen = _make_generator(outcomes=outcomes)

    counter = {"n": 0}

    async def fake_acompletion(**kwargs):
        counter["n"] += 1
        return _fake_completion('{"tool": "execute_sql", "q": %d}' % counter["n"])

    trig = RetrainingTrigger(
        TriggerConfig(
            total_failures_threshold=3,
            dominance_ratio=2.0,
            min_examples_ready=999,
            max_buffer_size=100,
        )
    )

    with patch("litellm.acompletion", new=fake_acompletion):
        # Generate from several classified failures and push into Path B's buffer.
        failures = [_classified(root_cause="wrong_tool", tid=f"t{i}") for i in range(3)]
        examples = await gen.generate_batch(failures)

    for ex in examples:
        trig.buffer.add(ex)

    assert len(trig.buffer) == 3
    drained = trig.check_and_fire()  # volume trigger fires at 3
    assert drained is not None
    assert len(drained) == 3

    # And the drained (real-generator) examples convert to a DPO dataset.
    rows = examples_to_dpo_dataset(drained)
    assert len(rows) == 3
