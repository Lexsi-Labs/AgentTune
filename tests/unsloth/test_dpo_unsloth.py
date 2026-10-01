"""DPO reward-ranked-rollout training pipeline sanity check (Unsloth backend).

Mirrors tests/test_dpo.py but pins backend="unsloth" explicitly and adds
LoRA + 4-bit so the Unsloth code path is actually exercised.

Run in isolation (see tests/unsloth/conftest.py for why):
    RUN_UNSLOTH_TESTS=1 pytest tests/unsloth/test_dpo_unsloth.py -q --no-cov
"""

import logging

import pytest

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

pytestmark = pytest.mark.gpu

train_data = [
    {"prompt": "What is the capital of France?"},
    {"prompt": "What is 2 plus 2?"},
    {"prompt": "Name one planet in our solar system."},
    {"prompt": "What colour is the sky on a clear day?"},
]


def length_reward(responses, prompts=None, **kwargs):
    responses = responses or []
    if not responses:
        return []
    max_len = max(len(r) for r in responses) or 1
    return [len(r) / max_len for r in responses]


def test_dpo_pipeline_unsloth():
    """End-to-end DPO reward-ranked-rollout training pipeline sanity check (Unsloth backend)."""
    from datasets import Dataset

    from agenttune.core.backend_factory import create_agentic_trainer

    train_dataset = Dataset.from_list(train_data)
    assert len(train_dataset) == 4

    trainer_obj = create_agentic_trainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="dpo",
        backend="unsloth",  # pin explicitly -- this is what we're testing
        reward_funcs=length_reward,
        train_dataset=train_dataset,
        num_generations=2,  # generate 2 responses per prompt -> 1 chosen, 1 rejected
        max_new_tokens=32,  # keep short for fast testing
        temperature=0.9,  # diversity between generations
        beta=0.1,
        loss_type="sigmoid",
        max_steps=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        logging_steps=1,
        save_steps=9999,
        output_dir="/tmp/dpo_unsloth_test_run",
        lora_r=8,
        lora_alpha=8,
        load_in_4bit=True,
        max_seq_length=512,
    )

    results = trainer_obj.train()
    assert results is not None
