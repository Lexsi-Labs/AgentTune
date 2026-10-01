"""RLOO training pipeline sanity check (Unsloth backend).

Mirrors tests/test_rloo.py but pins backend="unsloth" explicitly and adds
LoRA + 4-bit so the Unsloth code path is actually exercised.

Note: unlike tests/test_rloo.py, this calls .train() rather than calling
setup_data()/setup_trainer() directly. The Unsloth RLOO wrapper needs
setup_model() to run first (it swaps the model kwarg from a string to the
loaded object) and only .train() calls it -- setup_trainer() alone still
expects a string-to-model resolution that only the TRL wrapper does
internally, so calling it standalone raises `'str' object has no attribute
'config'` on the Unsloth backend. That's a real, currently-unfixed gap in
the Unsloth backend (not something this test works around).

Run in isolation (see tests/unsloth/conftest.py for why):
    RUN_UNSLOTH_TESTS=1 pytest tests/unsloth/test_rloo_unsloth.py -q --no-cov
"""

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
def test_rloo_pipeline_unsloth():
    """End-to-end RLOO training pipeline sanity check (Unsloth backend)."""
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
        backend="unsloth",  # pin explicitly -- this is what we're testing
        reward_funcs=length_reward,
        train_dataset=train_dataset,
        num_generations=2,
        max_completion_length=32,
        temperature=0.9,
        beta=0.05,
        max_steps=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        logging_steps=1,
        save_steps=9999,
        output_dir="/tmp/rloo_unsloth_test_run",
        lora_r=8,
        lora_alpha=8,
        load_in_4bit=True,
        max_seq_length=512,
    )

    results = trainer_obj.train()
    assert results is not None
