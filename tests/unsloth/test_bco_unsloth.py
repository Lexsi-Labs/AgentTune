"""BCO agentic (tool-calling) training pipeline sanity check (Unsloth backend).

Mirrors tests/test_bco.py but pins backend="unsloth" explicitly and adds
LoRA + 4-bit so the Unsloth code path is actually exercised.

Run in isolation (see tests/unsloth/conftest.py for why):
    RUN_UNSLOTH_TESTS=1 pytest tests/unsloth/test_bco_unsloth.py -q --no-cov
"""

import logging

import pytest

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

pytestmark = pytest.mark.gpu

train_data = [
    {"prompt": "What is 12 plus 45?"},
    {"prompt": "What is 100 divided by 4?"},
    {"prompt": "What is 7 times 8?"},
    {"prompt": "What is 256 minus 99?"},
]


def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression and return the result as a string."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"


def correctness_reward(prompts=None, responses=None, **kwargs):
    _OP_MAP = {
        "plus": "+",
        "minus": "-",
        "times": "*",
        "multiplied by": "*",
        "divided by": "/",
        "to the power of": "**",
    }
    responses = responses or []
    rewards = []
    for prompt, response in zip(prompts or [], responses, strict=False):
        expr = prompt.lower().replace("what is", "").replace("?", "").strip()
        for word, sym in _OP_MAP.items():
            expr = expr.replace(word, sym)
        try:
            expected = str(eval(expr, {"__builtins__": {}})).rstrip("0").rstrip(".")
            score = 1.0 if expected in response else 0.0
        except Exception:
            score = 0.0
        rewards.append(score)
    return rewards


def test_bco_pipeline_unsloth():
    """End-to-end BCO agentic (tool-calling) training pipeline sanity check (Unsloth backend)."""
    from datasets import Dataset

    from agenttune.core.backend_factory import create_agentic_trainer

    train_dataset = Dataset.from_list(train_data)
    assert len(train_dataset) == 4

    trainer_obj = create_agentic_trainer(
        algorithm="bco",
        backend="unsloth",  # pin explicitly -- this is what we're testing
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=correctness_reward,
        tools=[calculator],
        max_steps_per_turn=2,
        system_prompt=(
            "You are a helpful assistant. "
            "Always use the calculator tool to solve arithmetic problems. "
            "Never guess — call the tool and use its result in your answer."
        ),
        train_dataset=train_dataset,
        use_rollouts=True,
        score_threshold=0.5,
        num_generations=2,
        prompts_per_epoch=4,
        beta=0.1,
        max_length=128,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=9999,
        output_dir="/tmp/bco_unsloth_test_run",
        eval_strategy="no",
        lora_r=8,
        lora_alpha=8,
        load_in_4bit=True,
        max_seq_length=512,
    )

    results = trainer_obj.train()
    assert results is not None
