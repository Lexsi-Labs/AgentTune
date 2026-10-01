"""The thesis test: ONE Project's trajectories flow through EVERY lifecycle stage.

Every other test validates a single stage against a real consumer with its own fixture.
This one proves the actual claim — "one trajectory schema across the lifecycle" — by running
collect_rollout -> evaluate -> distill -> heal on the SAME accumulated trajectories of a
single Project, GPU-free. If the mixed trajectory list broke any stage, it would surface here.
"""

from agenttune.agentic.project import Project, agentic_metrics
from agenttune.agentic.rollout_engines.base import RolloutEngine


class _FakeTok:
    def apply_chat_template(self, *a, **k):
        raise RuntimeError("no template")

    def encode(self, text, add_special_tokens=True):
        return []


class FakeEngine(RolloutEngine):
    def _get_tokenizer(self):
        return _FakeTok()

    def generate(self, prompts, tools, gen_cfg):
        return {
            "completions": ["reasoned answer 42"],
            "logprobs": [[-0.1, -0.2]],
            "metadata": {"backend": "fake"},
        }


class _RecordingTrainer:
    last_kwargs = None

    def __init__(self, **kwargs):
        _RecordingTrainer.last_kwargs = kwargs

    def train(self):
        return {"train_loss": 0.05}


def test_one_trajectory_set_through_the_whole_lifecycle():
    p = Project()

    # BUILD/COLLECT — real rollout machinery, full-tier trajectories
    logs = p.collect_rollout(FakeEngine(), ["q1", "q2"], max_steps=1)
    assert len(logs) == 2 and all(l.tier == "full" for l in logs)

    # EVAL — the real evaluator's metrics on those SAME trajectories
    scores = [agentic_metrics(l) for l in p.trajectories]
    assert len(scores) == 2 and all("arr" in s for s in scores)

    # DISTILL — SFT a student on those SAME trajectories
    result = p.distill("student-1.5B", trainer_factory=_RecordingTrainer)
    assert result["train_loss"] == 0.05
    assert len(_RecordingTrainer.last_kwargs["train_dataset"]) == 2  # one row per trajectory

    # HEAL — inspect those SAME trajectories via the real FailureDetector
    failures = p.heal()
    assert isinstance(failures, list)  # runs clean on the set

    # one Project, one schema: the lifecycle event stream spans every stage
    stages = {ev.stage for ev in p.events()}
    assert {"collect_rollout", "distill", "heal"} <= stages
    assert len(p.trajectories) == 2 and len(p.native_trajectories) == 2
