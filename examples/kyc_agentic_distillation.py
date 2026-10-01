"""
BFSI case study — Distil a KYC-triage agent into a small on-prem model.
======================================================================

Banks often cannot send customer PII to a large hosted model, and want inference cheap and
on-premises. Agentic distillation fits: run a strong teacher agent to produce KYC-triage
trajectories, then behavior-clone them into a small (<3B) student that can run in the bank's
own VPC. The teacher's *design* is compressed into the student — not its weights. GPU-free
here; the injected trainer is where the real fine-tune would run on the bank's GPU.

Run:  python examples/kyc_agentic_distillation.py
"""

from agenttune.agentic import Project
from agenttune.agentic.rollout_engines.demo_engine import DemoRolloutEngine


class OnPremTrainerFactory:
    """Stand-in for the bank's on-prem SFT trainer. Records what the spine hands it."""

    def __init__(self):
        self.captured = {}

    def __call__(self, *, model, train_dataset, **kwargs):
        self.captured = {"model": model, "n_rows": len(train_dataset)}

        class _Trainer:
            def train(_self):
                return {
                    "status": "would-train-in-bank-vpc",
                    "student": self.captured["model"],
                    "n_rows": self.captured["n_rows"],
                }

        return _Trainer()


def main():
    # 1) TEACHER — a capable KYC analyst agent triages onboarding cases (GPU-free engine).
    teacher = Project()
    cases = [
        "onboard: sole trader, cash-intensive business, PEP=no",
        "onboard: non-resident, complex ownership, source-of-funds unclear",
        "onboard: salaried applicant, clean sanctions screen",
    ]
    teacher.collect_rollout(DemoRolloutEngine(), cases, tools=[], max_steps=2)
    print(f"[teacher]  {len(teacher.trajectories)} KYC-triage trajectories collected")

    rows = teacher.sft_dataset()
    print(
        f"[teacher]  behavior-cloning dataset -> {len(rows)} SFT rows (schema {list(rows[0].keys())})"
    )

    # 2) DISTILL — compress the analyst design into a small on-prem student.
    factory = OnPremTrainerFactory()
    result = teacher.distill("kyc-analyst-1.3B", trainer_factory=factory)
    print(
        f"[distill]  student='{factory.captured['model']}' fed {factory.captured['n_rows']} teacher rows"
    )
    print(f"[distill]  result -> {result}")


if __name__ == "__main__":
    main()
