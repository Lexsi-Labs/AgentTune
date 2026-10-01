"""
Healthcare case study — distil a medical-coding agent into an on-prem model.
============================================================================

Clinical text can't leave the hospital's network, and coding volume makes hosted inference
expensive. Agentic distillation fits: a strong teacher agent assigns diagnosis codes to
encounters, and its trajectories are behavior-cloned into a small (<3B) student that runs
inside the hospital's own environment. GPU-free; the injected trainer is where the real
fine-tune would run on the hospital's GPU.

Illustrative only — not a certified coding system.

Run:  python examples/medical_coding_distillation.py
"""

from agenttune.agentic import Project
from agenttune.agentic.rollout_engines.demo_engine import DemoRolloutEngine


class OnPremTrainerFactory:
    def __init__(self):
        self.captured = {}

    def __call__(self, *, model, train_dataset, **kwargs):
        self.captured = {"model": model, "n_rows": len(train_dataset)}

        class _Trainer:
            def train(_self):
                return {
                    "status": "would-train-in-hospital-vpc",
                    "student": self.captured["model"],
                    "n_rows": self.captured["n_rows"],
                }

        return _Trainer()


def main():
    teacher = Project()
    encounters = [
        "encounter: type 2 diabetes with neuropathy, routine follow-up",
        "encounter: community-acquired pneumonia, admitted",
        "encounter: essential hypertension, medication review",
    ]
    teacher.collect_rollout(DemoRolloutEngine(), encounters, tools=[], max_steps=2)
    print(f"[teacher]  {len(teacher.trajectories)} coded-encounter trajectories collected")

    rows = teacher.sft_dataset()
    print(
        f"[teacher]  behavior-cloning dataset -> {len(rows)} SFT rows (schema {list(rows[0].keys())})"
    )

    factory = OnPremTrainerFactory()
    result = teacher.distill("med-coder-1.3B", trainer_factory=factory)
    print(
        f"[distill]  student='{factory.captured['model']}' fed {factory.captured['n_rows']} teacher rows"
    )
    print(f"[distill]  result -> {result}")


if __name__ == "__main__":
    main()
