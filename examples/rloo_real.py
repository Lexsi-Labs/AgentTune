"""
REAL RLOO — on-policy RL with a leave-one-out baseline, on the GPU.
==================================================================

RLOO (REINFORCE Leave-One-Out) is the other on-policy RL family: like GRPO it samples several
completions per prompt and reinforces the higher-reward ones, but it uses a leave-one-out baseline
instead of the group-normalised advantage. This trains a real TRL `RLOOTrainer` (LoRA) on
SmolLM2-360M, same task and reward as the GRPO example so the two are directly comparable.

The reward pays out only for the correct label in the exact `SENTIMENT=<label>` contract, so a
constant-label shortcut caps at ~1/3 and the policy has to actually classify.

Requires a GPU + SmolLM2-360M in the local HF cache. Run:
    python examples/rloo_real.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys

sys.modules["vllm"] = None  # env vllm is ABI-broken vs torch; TRL imports it eagerly

import re

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RLOOConfig, RLOOTrainer

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUT_DIR = "/tmp/agenttune-rloo-student"
INSTRUCT = (
    "Classify the sentiment as positive, negative, or neutral. "
    "Reply with exactly SENTIMENT=<label> and nothing else. Review: {r!r}"
)
PROBE = INSTRUCT.format(r="the delivery was late and support ignored me")


def contract_reward(completions, label=None, **kwargs):
    def as_text(comp):
        if isinstance(comp, list):
            for m in reversed(comp):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    return str(m.get("content", ""))
            return ""
        return str(comp)

    labels = label if isinstance(label, list) else [label] * len(completions)
    # Reward ONLY the correct label in the exact contract; a constant-label shortcut caps at ~1/3.
    return [
        1.0 if re.match(rf"^sentiment={re.escape(str(g))}\b", as_text(c).strip().lower()) else 0.0
        for c, g in zip(completions, labels, strict=False)
    ]


def sample(model, tok, prompt, n=4, seed=0):
    torch.manual_seed(seed)
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    outs = []
    with torch.no_grad():
        for _ in range(n):
            o = model.generate(
                **enc,
                max_new_tokens=12,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            outs.append(
                tok.decode(o[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            )
    return outs


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
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [[{"role": "user", "content": INSTRUCT.format(r=r)}] for r, _ in reviews]
    ds = Dataset.from_dict({"prompt": prompts, "label": [lbl for _, lbl in reviews]})
    print(f"[data]    {len(ds)} prompts for on-policy sampling")

    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    before = sample(base, tok, PROBE, n=4)
    del base
    torch.cuda.empty_cache()

    cfg = RLOOConfig(
        output_dir=OUT_DIR,
        num_generations=16,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=30,
        learning_rate=1e-5,
        max_completion_length=12,
        max_grad_norm=0.5,
        temperature=1.1,
        logging_steps=4,
        save_steps=10000,
        beta=0.02,
        bf16=True,
        report_to=[],
        use_vllm=False,
    )
    trainer = RLOOTrainer(
        model=MODEL,
        reward_funcs=contract_reward,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    trainer.train()
    trainer.save_model(OUT_DIR)

    hist = [h for h in trainer.state.log_history if "reward" in h]
    r0, r1 = (hist[0]["reward"], hist[-1]["reward"]) if hist else (float("nan"), float("nan"))

    del trainer
    torch.cuda.empty_cache()
    policy = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda"), OUT_DIR
    ).merge_and_unload()
    after = sample(policy, tok, PROBE, n=4)

    def rate(s, p):
        return sum(p(x.strip().lower()) for x in s) / len(s)

    def is_contract(s):
        return s.startswith("sentiment=")

    def is_correct(s):
        return s.startswith("sentiment=negative")

    print(f"[rloo]    {len(hist)} logged steps  |  mean reward {r0:.3f} -> {r1:.3f}")
    print(f"[probe]   {PROBE}")
    print(f"[before]  base policy samples -> {before}")
    print(f"[after]   RL policy samples   -> {after}")
    print(
        f"[verdict] reward improved: {r1 > r0}  |  contract rate {rate(before, is_contract):.0%} "
        f"-> {rate(after, is_contract):.0%}  |  correct-label rate "
        f"{rate(before, is_correct):.0%} -> {rate(after, is_correct):.0%}"
    )


if __name__ == "__main__":
    main()
