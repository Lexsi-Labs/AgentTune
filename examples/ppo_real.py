"""
REAL PPO — actor-critic RL against a learned reward model, on the GPU.
=====================================================================

PPO is the classic actor-critic RLHF algorithm: a policy generates, a reward model scores, and a
value model estimates the baseline. Unlike GRPO/RLOO (which take a Python reward function), PPO
needs an actual reward *model* — so this example first trains one (TRL `RewardTrainer`) on the
contract-vs-prose preference data, then runs real `trl.experimental.ppo.PPOTrainer` (LoRA) to push
the policy toward higher reward.

(PPO and BCO live under `trl.experimental` in TRL 1.6 — the old top-level import raises
AttributeError; the experimental path is the working one.)

Requires a GPU + SmolLM2-360M in the local HF cache. Run:
    python examples/ppo_real.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import sys

sys.modules["vllm"] = None  # env vllm is ABI-broken vs torch; TRL imports it eagerly

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer
from trl.experimental.ppo import PPOConfig, PPOTrainer

from agenttune.agentic import build_dataset
from agenttune.decide.closed_loop.contracts import TrainingExample

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUT_DIR = "/tmp/agenttune-ppo-student"
PROBE = "Classify the sentiment: 'terrible, broke on day one'"

REVIEWS = [
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


def pref(text, label):
    return TrainingExample(
        trajectory_id=f"ppo-{text[:6]}",
        original_failure_type="format_violation",
        root_cause="prose",
        prompt=[{"role": "user", "content": f"Classify the sentiment: '{text}'"}],
        chosen=[{"role": "assistant", "content": f"SENTIMENT={label}"}],
        rejected=[{"role": "assistant", "content": f"The sentiment of this review is {label}."}],
    )


def generate(model, tok, prompt):
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        o = model.generate(
            **enc,
            max_new_tokens=14,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(o[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def train_reward_model(tok, rows):
    rm = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=1, torch_dtype=torch.bfloat16
    )
    rm.config.pad_token_id = tok.pad_token_id
    cfg = RewardConfig(
        output_dir=OUT_DIR + "-rm",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        max_length=128,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        bf16=True,
    )
    RewardTrainer(
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
    ).train()
    return rm.merge_and_unload() if hasattr(rm, "merge_and_unload") else rm


def main():
    print(f"[gpu]     {torch.cuda.get_device_name(0)}  (cuda: {torch.cuda.is_available()})")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 1) Train the reward model PPO will optimise against.
    rows = build_dataset([pref(t, lbl) for t, lbl in REVIEWS])
    reward_model = train_reward_model(tok, rows).to("cuda")
    print(f"[rm]      reward model trained on {len(rows)} contract-vs-prose pairs")

    # 2) Prompt dataset (tokenized) for on-policy generation.
    def to_ids(r):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": f"Classify the sentiment: '{r}'"}],
            add_generation_prompt=True,
        )
        if not isinstance(
            ids, list
        ):  # transformers>=5.x returns a BatchEncoding (not a dict subclass)
            ids = ids["input_ids"]
        return {"input_ids": ids}

    ds = Dataset.from_list([to_ids(r) for r, _ in REVIEWS] * 6)  # repeat for enough episodes
    print(f"[data]    {len(ds)} prompt episodes")

    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    before = generate(base, tok, PROBE)
    del base
    torch.cuda.empty_cache()

    # 3) PPO: policy (LoRA) + value model + the trained reward model.
    policy = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=1, torch_dtype=torch.bfloat16
    )
    value_model.config.pad_token_id = tok.pad_token_id
    cfg = PPOConfig(
        # PPO is unstable — a hotter LR / lower KL drove the reward negative and the policy adrift.
        # These conservative settings improve the RLHF reward modestly without collapse.
        output_dir=OUT_DIR,
        total_episodes=192,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        num_mini_batches=1,
        num_ppo_epochs=2,
        learning_rate=1e-5,
        response_length=12,
        temperature=1.0,
        kl_coef=0.05,
        missing_eos_penalty=1.0,
        stop_token="eos",
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
    )
    trainer = PPOTrainer(
        args=cfg,
        processing_class=tok,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=ds,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    trainer.train()

    scores = [h["objective/scores"] for h in trainer.state.log_history if "objective/scores" in h]
    after = generate(
        trainer.model.policy if hasattr(trainer.model, "policy") else trainer.model, tok, PROBE
    )

    # PPO is the most unstable / sample-hungry of these algorithms. This run verifies the full
    # actor-critic pipeline executes on the GPU (reward-model training -> policy/value/reward-model
    # PPO loop). The reward-model score over a short run is NOISY — it moves but does not reliably
    # climb; a stable gain needs far more episodes and careful tuning. Reported honestly.
    print(f"[ppo]     {len(scores)} updates completed on GPU (RM-trained, actor-critic loop ran)")
    print(
        f"[reward]  reward-model score over updates: start {scores[0]:+.2f}, "
        f"peak {max(scores):+.2f}, end {scores[-1]:+.2f}  (noisy — PPO is unstable at this scale)"
    )
    print(f"[probe]   {PROBE}")
    print(f"[before]  base policy -> {before!r}")
    print(f"[after]   PPO policy  -> {after!r}")
    print(
        f"[verdict] PPO pipeline ran end-to-end on GPU; reward reached {max(scores):+.2f} "
        f"(from {scores[0]:+.2f}) but a reliable gain needs many more episodes"
    )


if __name__ == "__main__":
    main()
