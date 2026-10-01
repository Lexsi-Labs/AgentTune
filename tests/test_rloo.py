import logging

import pytest

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def _free_gpu_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info(0)
        return free / (1024**3)
    except Exception:
        return 0.0


pytestmark = pytest.mark.gpu

_MIN_FREE_GB = 16.0

skip_if_low_vram = pytest.mark.skipif(
    _free_gpu_gb() < _MIN_FREE_GB,
    reason=f"RLOO training needs >{_MIN_FREE_GB} GB free GPU VRAM (found {_free_gpu_gb():.1f} GB)",
)


@skip_if_low_vram
def test_rloo_pipeline():
    """End-to-end RLOO training pipeline sanity check."""
    from datasets import Dataset

    from agenttune.core.backend_factory import create_agentic_trainer

    train_data = [
        {"prompt": "What is the capital of France?"},
        {"prompt": "What is 2 plus 2?"},
        {"prompt": "Name one planet in our solar system."},
        {"prompt": "What colour is the sky on a clear day?"},
    ]
    train_dataset = Dataset.from_list(train_data)
    assert len(train_dataset) == 4

    def length_reward(prompts=None, completions=None, **kwargs):
        responses = completions or []
        if not responses:
            return []
        max_len = max(len(r) for r in responses) or 1
        return [len(r) / max_len for r in responses]

    trainer_obj = create_agentic_trainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="rloo",
        reward_funcs=length_reward,
        train_dataset=train_dataset,
        num_generations=2,
        max_completion_length=32,
        temperature=0.9,
        beta=0.05,
        max_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        logging_steps=1,
        save_steps=9999,
        output_dir="/tmp/rloo_test_run",
    )

    trainer_obj.setup_data()
    assert trainer_obj.train_dataset is not None

    trainer_obj.setup_trainer()
    assert trainer_obj.trainer is not None

    trainer_obj.trainer.train()
