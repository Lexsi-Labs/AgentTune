import logging

import pytest

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from datasets import Dataset

from agenttune.core.backend_factory import create_agentic_trainer

# Trains Qwen2.5-0.5B with a full TRL trainer — same class of test as
# test_rloo/test_grpo (gpu-marked); the 7GB CPU CI runner OOMs on it.
pytestmark = pytest.mark.gpu

train_data = [
    {"prompt": "What is the capital of France?"},
    {"prompt": "What is 2 plus 2?"},
    {"prompt": "Name one planet in our solar system."},
    {"prompt": "What colour is the sky on a clear day?"},
]


def length_reward(responses, prompts=None, **kwargs):
    """
    Simple reward: longer responses score higher.
    In production replace with an LLM judge or exact-match checker.
    """
    if not responses:
        return []
    max_len = max(len(r) for r in responses) or 1
    rewards = [len(r) / max_len for r in responses]
    for i, (r, rw) in enumerate(zip(responses, rewards, strict=False)):
        print(f"  [reward] response[{i}]={r[:60]!r}  score={rw:.3f}")
    return rewards


def test_dpo_pipeline():
    """End-to-end DPO reward-ranked-rollout training pipeline sanity check."""
    train_dataset = Dataset.from_list(train_data)
    assert len(train_dataset) == 4

    trainer_obj = create_agentic_trainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        algorithm="dpo",
        reward_funcs=length_reward,
        train_dataset=train_dataset,
        num_generations=2,  # generate 2 responses per prompt → 1 chosen, 1 rejected
        max_new_tokens=32,  # keep short for fast testing
        temperature=0.9,  # diversity between generations
        beta=0.1,
        loss_type="sigmoid",
        max_steps=1,  # just 1 step to verify the pipeline
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        logging_steps=1,
        save_steps=1,
        output_dir="/tmp/dpo_test_run",
    )

    results = trainer_obj.train()
    assert results is not None
