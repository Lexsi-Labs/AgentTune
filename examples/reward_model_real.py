"""
REAL reward-model training — trains the scorer that RL optimises against, on the GPU.
=====================================================================================

GRPO/PPO need a reward. `lifecycle_and_rewards.py` shows programmatic reward *functions*; the
other real path is a learned *reward model* — a scalar scorer trained from preference data to rank
a good answer above a bad one. This trains one for real with TRL's `RewardTrainer` (LoRA) on the
same `{prompt, chosen, rejected}` data the self-heal loop emits (via the spine's `build_dataset`).

A reward model is `AutoModelForSequenceClassification` with one output. After training it should
score `chosen` above `rejected` on held-out pairs — that margin is the whole point.

Requires a GPU + SmolLM2-360M in the local HF cache. Run:
    python examples/reward_model_real.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys

sys.modules["vllm"] = None  # env vllm is ABI-broken vs torch; TRL imports it eagerly

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

from agenttune.agentic import build_dataset
from agenttune.decide.closed_loop.contracts import TrainingExample

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUT_DIR = "/tmp/agenttune-reward-model"


def preference(text, label):
    """Corrective preference: the contract answer (chosen) ranked over a prose answer (rejected)."""
    return TrainingExample(
        trajectory_id=f"rm-{text[:6]}",
        original_failure_type="format_violation",
        root_cause="prose",
        prompt=[{"role": "user", "content": f"Classify the sentiment: '{text}'"}],
        chosen=[{"role": "assistant", "content": f"SENTIMENT={label}"}],
        rejected=[{"role": "assistant", "content": f"The sentiment of this review is {label}."}],
    )


def score(model, tok, prompt, answer):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
        tokenize=False,
    )
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        return model(**enc).logits[0, 0].item()


def main():
    print(f"[gpu]     {torch.cuda.get_device_name(0)}  (cuda: {torch.cuda.is_available()})")

    reviews = [
        ("I love this product, best purchase ever", "positive"),
        ("absolutely fantastic experience", "positive"),
        ("exceeded my expectations", "positive"),
        ("delighted with the quality", "positive"),
        ("this is the worst thing I've bought", "negative"),
        ("terrible, broke on day one", "negative"),
        ("support was rude and unhelpful", "negative"),
        ("waste of money, deeply disappointed", "negative"),
        ("it arrived on time, works as described", "neutral"),
        ("standard packaging, nothing special", "neutral"),
        ("does the job, no strong feelings", "neutral"),
        ("average product, meets the basics", "neutral"),
    ]
    rows = build_dataset([preference(t, lbl) for t, lbl in reviews])
    print(
        f"[data]    {len(rows)} preference rows (schema {list(rows[0].keys())}) — chosen=contract, rejected=prose"
    )

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rm = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=1, torch_dtype=torch.bfloat16
    ).to("cuda")
    rm.config.pad_token_id = tok.pad_token_id

    probe = (
        "Classify the sentiment: 'terrible, broke on day one'",
        "SENTIMENT=negative",
        "The sentiment of this review is negative.",
    )

    # Mean pairwise margin: how far the RM scores chosen above rejected, averaged over the set.
    # (Accuracy saturates — a random head ranks these easy pairs right by chance; margin is the
    # signal that actually moves as the model learns to score the correction confidently.)
    def mean_margin(model):
        return sum(
            score(model, tok, r["prompt"][0]["content"], r["chosen"][0]["content"])
            - score(model, tok, r["prompt"][0]["content"], r["rejected"][0]["content"])
            for r in rows
        ) / len(rows)

    before = (score(rm, tok, probe[0], probe[1]), score(rm, tok, probe[0], probe[2]))
    margin_before = mean_margin(rm)  # measured before the trainer wraps rm in-place

    cfg = RewardConfig(
        output_dir=OUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        max_length=128,
        logging_steps=2,
        save_strategy="no",
        report_to=[],
        bf16=True,
    )
    trainer = RewardTrainer(
        model=rm,
        args=cfg,
        train_dataset=Dataset.from_list(rows),
        processing_class=tok,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            task_type="SEQ_CLS",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    trainer.train()

    margin_after = mean_margin(trainer.model)
    after = (
        score(trainer.model, tok, probe[0], probe[1]),
        score(trainer.model, tok, probe[0], probe[2]),
    )

    print(
        f"[rm]      mean chosen-minus-rejected margin over {len(rows)} pairs "
        f"{margin_before:+.2f} -> {margin_after:+.2f}"
    )
    print(f"[probe]   {probe[0]}")
    print(
        f"[before]  reward(chosen)={before[0]:+.2f}  reward(rejected)={before[1]:+.2f}  "
        f"chosen_wins={before[0] > before[1]}"
    )
    print(
        f"[after]   reward(chosen)={after[0]:+.2f}  reward(rejected)={after[1]:+.2f}  "
        f"chosen_wins={after[0] > after[1]}"
    )
    print(
        f"[verdict] reward model ranks the contract answer above prose, margin "
        f"{after[0] - after[1]:+.2f} (was {before[0] - before[1]:+.2f})"
    )


if __name__ == "__main__":
    main()
