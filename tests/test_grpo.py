import re

import pytest


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
    reason=f"GRPO training needs >{_MIN_FREE_GB} GB free GPU VRAM (found {_free_gpu_gb():.1f} GB)",
)


def calculator(expression: str) -> dict:
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e), "expression": expression}


def unit_converter(value: float, from_unit: str, to_unit: str) -> dict:
    CONVERSIONS = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
        ("c", "f"): lambda v: v * 9 / 5 + 32,
        ("f", "c"): lambda v: (v - 32) * 5 / 9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key not in CONVERSIONS:
        return {"error": f"Conversion {from_unit} → {to_unit} not supported."}
    factor = CONVERSIONS[key]
    converted = factor(value) if callable(factor) else value * factor
    return {
        "result": round(converted, 4),
        "from": f"{value} {from_unit}",
        "to": f"{converted:.4f} {to_unit}",
    }


GRPO_EXAMPLES = [
    {
        "prompt": [
            {
                "role": "user",
                "content": "A store sells apples for $1.50 each. If you buy 24 apples and get a 15% discount, how much do you pay in total?",
            }
        ],
        "answer": "30.6",
    },
    {
        "prompt": [
            {
                "role": "user",
                "content": "A train travels at 120 km/h. How long (in hours) does it take to cover 450 km?",
            }
        ],
        "answer": "3.75",
    },
    {
        "prompt": [
            {
                "role": "user",
                "content": "You invest $5000 at 8% annual interest. What is the total after 3 years with simple interest?",
            }
        ],
        "answer": "6200",
    },
    {
        "prompt": [
            {
                "role": "user",
                "content": "A rectangle has a length of 14.5 cm and a width of 8.3 cm. What is its area?",
            }
        ],
        "answer": "120.35",
    },
    {
        "prompt": [
            {
                "role": "user",
                "content": "If 7 workers can finish a job in 12 days, how many days will 4 workers take?",
            }
        ],
        "answer": "21",
    },
    {
        "prompt": [
            {"role": "user", "content": "Convert 250 km to miles and round to 2 decimal places."}
        ],
        "answer": "155.34",
    },
    {
        "prompt": [
            {
                "role": "user",
                "content": "A car uses 6.5 litres per 100 km. How many litres are needed for a 320 km trip?",
            }
        ],
        "answer": "20.8",
    },
    {
        "prompt": [
            {
                "role": "user",
                "content": "You score 78, 85, 91, and 74 on four tests. What is your average?",
            }
        ],
        "answer": "82",
    },
]


def reward_correct_answer(completions, prompts=None, answer=None, **kwargs):
    scores = []
    for comp, gt in zip(completions, answer or [], strict=False):
        final_text = ""
        if isinstance(comp, list):
            for msg in reversed(comp):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    final_text = str(msg.get("content", ""))
                    break
        else:
            final_text = str(comp)
        numbers = re.findall(r"-?\d+\.?\d*", final_text)
        try:
            gt_val = float(gt)
        except (ValueError, TypeError):
            scores.append(0.0)
            continue
        score = 0.0
        for n in numbers:
            try:
                pred = float(n)
                if abs(pred - gt_val) < 1e-3:
                    score = 1.5
                    break
                if gt_val != 0 and abs(pred - gt_val) / abs(gt_val) < 0.01:
                    score = max(score, 0.5)
            except ValueError:
                continue
        scores.append(score)
    return scores


def reward_tool_used(completions, prompts=None, **kwargs):
    scores = []
    for comp in completions:
        used = any(
            isinstance(msg, dict) and msg.get("role") == "tool"
            for msg in (comp if isinstance(comp, list) else [])
        )
        scores.append(1.0 if used else 0.0)
    return scores


def reward_concise_answer(completions, prompts=None, **kwargs):
    scores = []
    for comp in completions:
        final_text = ""
        if isinstance(comp, list):
            for msg in reversed(comp):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    final_text = str(msg.get("content", ""))
                    break
        else:
            final_text = str(comp)
        scores.append(0.3 if len(final_text.split()) <= 80 else 0.0)
    return scores


@skip_if_low_vram
def test_grpo_pipeline():
    """End-to-end GRPO agentic training pipeline sanity check."""
    from datasets import Dataset

    from agenttune.core.backend_factory import create_agentic_trainer

    train_dataset = Dataset.from_list(GRPO_EXAMPLES)
    assert len(train_dataset) == 8

    grpo_trainer = create_agentic_trainer(
        algorithm="grpo",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=[reward_correct_answer, reward_tool_used, reward_concise_answer],
        tools=[calculator, unit_converter],
        train_dataset=train_dataset,
        output_dir="./output/agentic_grpo_math",
        max_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        warmup_steps=0,
        num_generations=2,
        max_completion_length=32,
        max_steps_per_turn=2,
        temperature=0.7,
        beta=0.04,
        seed=42,
        logging_steps=1,
        save_steps=1,
        report_to="none",
        log_completions=True,
        chat_template_kwargs={"enable_thinking": False},
    )

    grpo_results = grpo_trainer.train()
    assert grpo_results is not None
