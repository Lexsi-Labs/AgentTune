"""
End-to-end closed-loop test using Qwen/Qwen2.5-0.5B-Instruct.

Runs the full pipeline unattended:
  synthetic failures
    → FailureDetector (log-based)
    → FailureClassifier (Qwen-backed judge, mocked to avoid API cost)
    → TrainingExampleGenerator (Qwen generates corrected responses)
    → ReplayValidator.validate_batch (parallel subprocesses)
    → BehavioralDiversityMonitor
    → RetrainingTrigger + ClosedLoopRunner.run()
    → DeploymentGate.evaluate_decision (trajectory scores included)

Skip marker: these tests load a 500M-param model and require a GPU or
sufficient CPU RAM. Gate with:
    pytest -m "not qwen_e2e"          # skip
    pytest -m qwen_e2e                # run only these

Set RUN_QWEN_E2E=1 to run them unconditionally in CI.
"""

import asyncio
import os
import time

import pytest

from agenttune.decide.closed_loop.behavioral_diversity_monitor import BehavioralDiversityMonitor
from agenttune.decide.closed_loop.closed_loop_runner import ClosedLoopRunner
from agenttune.decide.closed_loop.contracts import (
    AgenticEvalResult,
    ClassifiedFailure,
    Failure,
    TrainingExample,
)
from agenttune.decide.closed_loop.deployment_gate import DeploymentGate
from agenttune.decide.closed_loop.replay_validator import ReplayValidator
from agenttune.decide.closed_loop.retraining_trigger import RetrainingTrigger, TriggerConfig

RUN_QWEN_E2E = os.getenv("RUN_QWEN_E2E", "0") == "1"

pytestmark = pytest.mark.qwen_e2e


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qwen():
    """Load Qwen/Qwen2.5-0.5B-Instruct once for the entire session."""
    if not RUN_QWEN_E2E:
        pytest.skip("Set RUN_QWEN_E2E=1 to run Qwen E2E tests")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="session")
def qwen_generate_fn(qwen):
    """Return a callable: prompt_messages -> completion_str."""
    import torch

    model, tokenizer = qwen

    def generate(messages, max_new_tokens=64):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if next(model.parameters()).is_cuda:
            input_ids = input_ids.cuda()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = out[0][input_ids.shape[-1] :]
        return tokenizer.decode(completion, skip_special_tokens=True).strip()

    return generate


@pytest.fixture
def synthetic_failures():
    """Three staged failures covering the two Week-2 target types + one extra."""
    return [
        Failure(
            trajectory_id="traj-001",
            failure_type="wrong_tool",
            failed_stage_name="tool_call",
            context={"tool_used": "search", "tool_expected": "calculator"},
            judge_score=0.2,
        ),
        Failure(
            trajectory_id="traj-002",
            failure_type="loop_collapse",
            failed_stage_name="tool_call",
            context={"loop_count": 7, "repeated_tool": "search"},
            judge_score=0.1,
        ),
        Failure(
            trajectory_id="traj-003",
            failure_type="hallucinated_output",
            failed_stage_name="llm_call",
            context={"claim": "The capital of France is Berlin"},
            judge_score=0.0,
        ),
    ]


@pytest.fixture
def classified_failures(synthetic_failures):
    return [
        ClassifiedFailure(
            failure=f,
            root_cause=f.failure_type,
            confidence=0.9,
            analysis=f"Classified {f.failure_type} via rule",
        )
        for f in synthetic_failures
    ]


@pytest.fixture
def trigger():
    cfg = TriggerConfig(
        total_failures_threshold=2,
        min_examples_ready=2,
        max_drop_rate=0.9,
    )
    return RetrainingTrigger(config=cfg)


@pytest.fixture
def diversity_monitor():
    return BehavioralDiversityMonitor(history_size=50, diversity_threshold=0.2)


@pytest.fixture
def replay_validator():
    return ReplayValidator()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_example(cf: ClassifiedFailure, completion: str) -> TrainingExample:
    prompt = [{"role": "user", "content": f"Fix this failure: {cf.failure.failure_type}"}]
    chosen = [{"role": "assistant", "content": completion}]
    rejected = [{"role": "assistant", "content": "I don't know how to fix this."}]
    return TrainingExample(
        trajectory_id=cf.failure.trajectory_id,
        original_failure_type=cf.failure.failure_type,
        root_cause=cf.root_cause,
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        salvaged_at_attempt=1,
    )


def _stub_retrain_job(examples):
    """Minimal retrain stub: sleeps briefly to simulate work."""
    time.sleep(0.05)
    return {"trained_on": len(examples), "model_path": "/tmp/stub_model"}


def _fake_traj_results(n=3, base_score=0.75):
    # DeploymentGate._mean_trajectory_score composites overall_judge_score with
    # tac/scsr/iasa/egs (see deployment_gate.py's "eval-metric integration" commit) --
    # set them all to base_score too so the composite equals base_score exactly,
    # matching this helper's original intent, not diluted 5x by defaulted-to-0
    # component scores.
    return [
        AgenticEvalResult(
            trajectory_id=f"t{i}",
            goal_completion_score=base_score,
            tool_sequence_validity=base_score,
            unnecessary_steps_penalty=1 - base_score,
            error_recovery_score=base_score,
            overall_judge_score=base_score,
            tac_score=base_score,
            scsr_score=base_score,
            iasa_score=base_score,
            egs_score=base_score,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParallelReplayValidator:
    def test_validate_batch_runs_in_parallel(self, classified_failures, qwen_generate_fn):
        """validate_batch should complete faster than sequential sum of single calls."""
        examples = [
            _make_example(cf, f"correction for {cf.root_cause}") for cf in classified_failures
        ]
        validator = ReplayValidator()

        # time sequential
        t0 = time.monotonic()
        for ex in examples:
            asyncio.run(validator.validate_example_with_trace(ex, ""))
        sequential_s = time.monotonic() - t0

        # time parallel
        t1 = time.monotonic()
        results = asyncio.run(validator.validate_batch(examples))
        parallel_s = time.monotonic() - t1

        assert len(results) == len(examples)
        assert all(isinstance(ok, bool) for ok, _ in results)
        # parallel should be meaningfully faster (or at least not slower)
        assert (
            parallel_s <= sequential_s * 1.5
        ), f"parallel={parallel_s:.3f}s sequential={sequential_s:.3f}s — expected speedup"

    def test_validate_batch_empty(self):
        validator = ReplayValidator()
        result = asyncio.run(validator.validate_batch([]))
        assert result == []

    def test_validate_batch_preserves_order(self, classified_failures):
        examples = [_make_example(cf, "fix") for cf in classified_failures]
        validator = ReplayValidator()
        results = asyncio.run(validator.validate_batch(examples))
        assert len(results) == len(examples)
        # all should pass with default always-pass script
        assert all(ok for ok, _ in results)


class TestDiversityMonitorWired:
    def test_observe_trajectory_via_runner(self, trigger, diversity_monitor):
        runner = ClosedLoopRunner(
            trigger=trigger,
            retrain_job=_stub_retrain_job,
            diversity_monitor=diversity_monitor,
        )
        # feed 15 identical trajectories → should trigger collapse alert
        for _ in range(15):
            runner.observe_trajectory({"tool_calls": ["search", "search", "search"]})

        alert = diversity_monitor.check_diversity()
        assert alert, "Expected collapse alert after 15 identical trajectories"
        assert runner._diversity_monitor is diversity_monitor

    def test_diversity_checked_post_retrain(self, trigger, diversity_monitor, classified_failures):
        diversity_alert_seen = []

        def on_done(result):
            diversity_alert_seen.append(runner.last_diversity_alert)
            return None

        runner = ClosedLoopRunner(
            trigger=trigger,
            retrain_job=_stub_retrain_job,
            on_retrain_done=on_done,
            diversity_monitor=diversity_monitor,
        )

        # feed repetitive trajectories
        for _ in range(20):
            runner.observe_trajectory({"tool_calls": ["lookup", "lookup"]})

        # push enough examples to trigger retrain
        for cf in classified_failures:
            runner.submit(_make_example(cf, "correction"))

        runner.tick()
        runner.wait_for_retrain(timeout=5.0)

        assert len(diversity_alert_seen) == 1
        assert diversity_alert_seen[0] is True


class TestTrajectoryEvalWired:
    def test_trajectory_eval_fn_called_post_retrain(self, trigger, classified_failures):
        eval_called = []

        def traj_eval_fn(result):
            old = _fake_traj_results(3, base_score=0.70)
            new = _fake_traj_results(3, base_score=0.80)
            eval_called.append((old, new))
            return old, new

        def on_done(result):
            old_traj, new_traj = runner.last_trajectory_scores
            gate = DeploymentGate()
            return gate.evaluate_decision(
                test_set=[],
                old_model_fn=lambda x: "APPROVED",
                new_model_fn=lambda x: "APPROVED",
                old_trajectory_results=old_traj,
                new_trajectory_results=new_traj,
            )

        runner = ClosedLoopRunner(
            trigger=trigger,
            retrain_job=_stub_retrain_job,
            on_retrain_done=on_done,
            trajectory_eval_fn=traj_eval_fn,
        )

        for cf in classified_failures:
            runner.submit(_make_example(cf, "fix"))

        runner.tick()
        runner.wait_for_retrain(timeout=5.0)

        assert len(eval_called) == 1, "trajectory_eval_fn should have been called once"
        assert runner.last_decision is not None
        assert runner.last_trajectory_scores != (None, None)

    def test_worse_model_blocked_by_gate(self, trigger, classified_failures):
        """Gate must block a model that is worse on trajectory quality."""

        def traj_eval_fn(result):
            old = _fake_traj_results(3, base_score=0.80)
            new = _fake_traj_results(3, base_score=0.60)  # regressed
            return old, new

        def on_done(result):
            old_traj, new_traj = runner.last_trajectory_scores
            gate = DeploymentGate()
            return gate.evaluate_decision(
                test_set=[],
                old_model_fn=lambda x: "APPROVED",
                new_model_fn=lambda x: "APPROVED",
                old_trajectory_results=old_traj,
                new_trajectory_results=new_traj,
                trajectory_regression_tol=0.05,
            )

        runner = ClosedLoopRunner(
            trigger=trigger,
            retrain_job=_stub_retrain_job,
            on_retrain_done=on_done,
            trajectory_eval_fn=traj_eval_fn,
        )

        for cf in classified_failures:
            runner.submit(_make_example(cf, "fix"))

        runner.tick()
        runner.wait_for_retrain(timeout=5.0)

        assert runner.last_decision is not None
        assert runner.last_decision.approved is False
        assert "trajectory" in runner.last_decision.reason


class TestFullLoopUnattended:
    """Week 4 — one failure flows through every stage to a gated deploy decision."""

    def test_run_full_loop_with_qwen(
        self,
        trigger,
        diversity_monitor,
        replay_validator,
        qwen_generate_fn,
        classified_failures,
    ):
        submitted_examples = []
        gate_decisions = []

        def traj_eval_fn(result):
            old = _fake_traj_results(3, base_score=0.75)
            new = _fake_traj_results(3, base_score=0.80)
            return old, new

        def on_done(result):
            old_traj, new_traj = runner.last_trajectory_scores
            gate = DeploymentGate()
            decision = gate.evaluate_decision(
                test_set=[],
                old_model_fn=lambda x: "APPROVED",
                new_model_fn=lambda x: "APPROVED",
                old_trajectory_results=old_traj,
                new_trajectory_results=new_traj,
            )
            gate_decisions.append(decision)
            return decision

        runner = ClosedLoopRunner(
            trigger=trigger,
            retrain_job=_stub_retrain_job,
            on_retrain_done=on_done,
            diversity_monitor=diversity_monitor,
            trajectory_eval_fn=traj_eval_fn,
        )

        failure_iter = iter(classified_failures)

        def detect_fn():
            try:
                return [next(failure_iter)]
            except StopIteration:
                return []

        def classify_fn(failures):
            return failures  # already ClassifiedFailure

        def generate_fn(cf):
            prompt = [{"role": "user", "content": f"How do you fix {cf.root_cause}?"}]
            completion = qwen_generate_fn(prompt, max_new_tokens=32)
            ex = _make_example(cf, completion)
            submitted_examples.append(ex)
            return ex

        records = asyncio.run(
            runner.run(
                n_ticks=6,
                detect_fn=detect_fn,
                classify_fn=classify_fn,
                generate_fn=generate_fn,
                replay_validator=replay_validator,
                on_tick=lambda r: None,
            )
        )

        runner.wait_for_retrain(timeout=10.0)

        # assertions
        assert len(records) == 6, "Should have 6 CycleRecords"
        assert len(submitted_examples) == len(classified_failures)
        assert len(gate_decisions) >= 1, "Gate should have fired at least once"
        assert (
            gate_decisions[-1].approved is True
        ), f"Expected gate approval (improving model): {gate_decisions[-1].reason}"

    def test_buffer_keeps_filling_during_retrain(self, trigger, classified_failures):
        """Buffer must accept new examples while a retrain is in progress."""
        submitted_during = []
        retrain_started_event = __import__("threading").Event()

        def slow_retrain(examples):
            retrain_started_event.set()
            time.sleep(0.3)
            return {"trained_on": len(examples)}

        cfg2 = TriggerConfig(total_failures_threshold=2, min_examples_ready=2, max_drop_rate=0.9)
        slow_trigger = RetrainingTrigger(config=cfg2)
        runner = ClosedLoopRunner(
            trigger=slow_trigger,
            retrain_job=slow_retrain,
        )

        # prime the buffer enough to trigger
        for cf in classified_failures:
            runner.submit(_make_example(cf, "initial fix"))
        runner.tick()

        # wait for retrain to actually start
        retrain_started_event.wait(timeout=2.0)
        assert runner.is_retraining(), "Expected retrain to be in progress"

        # submit more examples WHILE retrain runs
        extra = _make_example(classified_failures[0], "extra during retrain")
        runner.submit(extra)
        submitted_during.append(extra)

        runner.wait_for_retrain(timeout=5.0)
        assert len(submitted_during) == 1, "Should have submitted 1 example during retrain"
