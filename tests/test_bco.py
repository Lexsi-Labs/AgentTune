import logging

import pytest

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from datasets import Dataset

from agenttune.core.backend_factory import create_agentic_trainer

# Trains Qwen2.5-0.5B with a full TRL trainer — same class of test as
# test_rloo/test_grpo (gpu-marked); the 7GB CPU CI runner OOMs on it.
pytestmark = pytest.mark.gpu

train_data = [
    {"prompt": "What is 12 plus 45?"},
    {"prompt": "What is 100 divided by 4?"},
    {"prompt": "What is 7 times 8?"},
    {"prompt": "What is 256 minus 99?"},
]


def calculator(expression: str) -> str:
    """
    Evaluate a simple arithmetic expression and return the result as a string.

    Args:
        expression: A valid Python arithmetic expression to evaluate.
            Examples: "12 + 45", "100 / 4", "3 ** 5", "7 * 8".
            Supports +, -, *, /, **, //, % operators.
            Does NOT support function calls or variable names.

    Returns:
        str: The numeric result of the expression, or an error message if
             the expression is invalid or cannot be evaluated safely.
    """
    try:
        result = str(eval(expression, {"__builtins__": {}}))
        print(f"  [tool] calculator({expression!r}) = {result}")
        return result
    except Exception as e:
        return f"Error: {e}"


def correctness_reward(prompts=None, responses=None, **kwargs):
    """
    Score each response based on whether it contains the correct numeric answer.

    Args:
        prompts   (list[str]): The input prompts, e.g. "What is 7 times 8?".
        responses (list[str]): The model-generated responses to score.
        **kwargs              : Unused extra keyword arguments (ignored).

    Returns:
        list[float]: 1.0 if the response contains the correct answer, else 0.0.
    """

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
        print(f"  [reward] response={response[:60]!r}  score={score:.1f}")

    return rewards


def test_bco_pipeline():
    """End-to-end BCO agentic (tool-calling) training pipeline sanity check."""
    train_dataset = Dataset.from_list(train_data)
    assert len(train_dataset) == 4

    trainer_obj = create_agentic_trainer(
        algorithm="bco",
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
        output_dir="/tmp/bco_agentic_run",
        eval_strategy="no",
    )

    results = trainer_obj.train()
    assert results is not None
