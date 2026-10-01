"""
Case study 2 — Agentic distillation (the headline value path).
==============================================================

Agentic distillation is *behavior cloning of a design*, NOT weight-KD: capture a strong
teacher agent's `full`-tier trajectories and SFT a small `student` on them — compressing an
agent DESIGN into a <3B model.

This rides the exact same rails as `train(fmt='sft')`: the teacher's trajectories become
SFT rows (schema: `messages`), and a `trainer_factory` receives the small student id + those
rows. The `Project` stays GPU-free — the factory is where the real `.train()` runs on a GPU.

Run:  python examples/agentic_distillation.py
"""

from agenttune.agentic import Project
from agenttune.agentic.rollout_engines.demo_engine import DemoRolloutEngine


class RecordingTrainerFactory:
    """Stand-in for a real SFT trainer. Records what the spine hands it so we can prove the
    student gets the small model id + the teacher-derived rows (real trainers run on GPU)."""

    def __init__(self):
        self.captured = {}

    def __call__(self, *, model, train_dataset, **kwargs):
        self.captured = {"model": model, "n_rows": len(train_dataset), "kwargs": kwargs}
        factory = self

        class _Trainer:
            def train(self):
                return {
                    "status": "would-train-on-gpu",
                    "student": factory.captured["model"],
                    "n_rows": factory.captured["n_rows"],
                }

        return _Trainer()


def main() -> None:
    # 1) TEACHER — a capable agent produces full-tier trajectories (GPU-free demo engine).
    teacher = Project()
    teacher_tasks = ["route the refund ticket", "classify the invoice", "answer the FAQ"]
    teacher.collect_rollout(DemoRolloutEngine(), teacher_tasks, tools=[], max_steps=2)
    print(f"[teacher] collected {len(teacher.trajectories)} full-tier trajectories")

    rows = teacher.sft_dataset()
    print(
        f"[teacher] behavior-cloning dataset -> {len(rows)} SFT rows (schema: {list(rows[0].keys())})"
    )

    # 2) DISTILL — SFT a small student on the teacher's trajectories.
    factory = RecordingTrainerFactory()
    result = teacher.distill("student-1.5B", trainer_factory=factory)
    print(
        f"[distill] trainer received model={factory.captured['model']!r}, "
        f"n_rows={factory.captured['n_rows']}"
    )
    print(f"[distill] result -> {result}")


if __name__ == "__main__":
    main()
