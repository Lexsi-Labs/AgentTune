"""
Retail case study — distil a catalog-enrichment agent into a small model.
=========================================================================

Catalog enrichment is high-volume and low-margin: for every SKU an agent normalises the title,
fills attributes, and writes a description. Running a frontier model per SKU across millions of
listings is expensive. Agentic distillation fits: a strong teacher agent enriches a sample, and
its trajectories are behavior-cloned into a small (<3B) student that runs cheaply at catalog
scale. GPU-free; the injected trainer is where the real fine-tune would run on a GPU.

Run:  python examples/catalog_enrichment_distillation.py
"""

from agenttune.agentic import Project
from agenttune.agentic.rollout_engines.demo_engine import DemoRolloutEngine


class CatalogTrainerFactory:
    def __init__(self):
        self.captured = {}

    def __call__(self, *, model, train_dataset, **kwargs):
        self.captured = {"model": model, "n_rows": len(train_dataset)}

        class _Trainer:
            def train(_self):
                return {
                    "status": "would-train-on-gpu",
                    "student": self.captured["model"],
                    "n_rows": self.captured["n_rows"],
                }

        return _Trainer()


def main():
    teacher = Project()
    skus = [
        "sku: mens running shoe, mesh, size 10, blue — raw supplier title",
        "sku: stainless water bottle 750ml — raw supplier title",
        "sku: wireless earbuds, ANC, usb-c — raw supplier title",
    ]
    teacher.collect_rollout(DemoRolloutEngine(), skus, tools=[], max_steps=2)
    print(f"[teacher]  {len(teacher.trajectories)} enriched-SKU trajectories collected")

    rows = teacher.sft_dataset()
    print(
        f"[teacher]  behavior-cloning dataset -> {len(rows)} SFT rows (schema {list(rows[0].keys())})"
    )

    factory = CatalogTrainerFactory()
    result = teacher.distill("catalog-enricher-1.3B", trainer_factory=factory)
    print(
        f"[distill]  student='{factory.captured['model']}' fed {factory.captured['n_rows']} teacher rows"
    )
    print(f"[distill]  result -> {result}")


if __name__ == "__main__":
    main()
