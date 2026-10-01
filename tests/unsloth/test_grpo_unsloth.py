"""GRPO agentic training pipeline sanity check (Unsloth backend).

Mirrors tests/test_grpo.py but pins backend="unsloth" explicitly (instead of
relying on backend="auto") and adds LoRA + 4-bit so the Unsloth code path is
actually exercised end to end, not silently skipped.

Run in isolation (see tests/unsloth/conftest.py for why):
    RUN_UNSLOTH_TESTS=1 pytest tests/unsloth/test_grpo_unsloth.py -q --no-cov
"""

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


GRPO_EXAMPLES = [
    {"prompt": [{"role": "user", "content": "What is 12 + 30?"}], "answer": "42"},
    {"prompt": [{"role": "user", "content": "What is 9 * 6?"}], "answer": "54"},
    {"prompt": [{"role": "user", "content": "What is 100 - 37?"}], "answer": "63"},
    {"prompt": [{"role": "user", "content": "What is 144 / 12?"}], "answer": "12"},
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
                if abs(float(n) - gt_val) < 1e-3:
                    score = 1.0
                    break
            except ValueError:
                continue
        scores.append(score)
    return scores


@skip_if_low_vram
def test_grpo_pipeline_unsloth():
    """End-to-end GRPO agentic training pipeline sanity check (Unsloth backend)."""
    from datasets import Dataset

    from agenttune.core.backend_factory import create_agentic_trainer

    train_dataset = Dataset.from_list(GRPO_EXAMPLES)
    assert len(train_dataset) == 4

    grpo_trainer = create_agentic_trainer(
        algorithm="grpo",
        backend="unsloth",  # pin explicitly -- this is what we're testing
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=[reward_correct_answer],
        train_dataset=train_dataset,
        output_dir="/tmp/grpo_unsloth_test_run",
        max_steps=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=24,
        max_seq_length=512,
        learning_rate=5e-5,
        warmup_steps=0,
        temperature=0.7,
        beta=0.04,
        seed=42,
        logging_steps=1,
        save_steps=1000,
        report_to="none",
        lora_r=8,
        lora_alpha=8,
        load_in_4bit=True,
    )

    grpo_results = grpo_trainer.train()
    assert grpo_results is not None
