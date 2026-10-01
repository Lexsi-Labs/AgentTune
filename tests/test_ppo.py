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


def length_reward(prompts=None, completions=None, **kwargs):
    responses = completions or []
    if not responses:
        return []
    max_len = max(len(r) for r in responses) or 1
    rewards = [len(r) / max_len for r in responses]
    for i, (r, rw) in enumerate(zip(responses, rewards, strict=False)):
        print(f"  [reward] completion[{i}]={r[:60]!r}  score={rw:.3f}")
    return rewards


def test_ppo_pipeline():
    """End-to-end PPO agentic training pipeline sanity check."""
    train_dataset = Dataset.from_list(train_data)
    assert len(train_dataset) == 4

    trainer_obj = create_agentic_trainer(
        algorithm="ppo",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=length_reward,
        train_dataset=train_dataset,
        max_prompt_length=64,
        total_episodes=4,
        response_length=32,
        per_device_train_batch_size=2,
        num_ppo_epochs=1,
        num_mini_batches=1,
        local_rollout_forward_batch_size=2,
        kl_coef=0.05,
        learning_rate=1e-6,
        temperature=0.9,
        logging_steps=1,
        save_steps=9999,
        num_sample_generations=0,
        output_dir="/tmp/ppo_test_run",
    )

    results = trainer_obj.train()
    assert results is not None
