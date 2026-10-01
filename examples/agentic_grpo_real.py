"""
REAL GRPO — on-policy RL that actually trains a model on the GPU.
=================================================================

The GPU-free case studies show reward *shaping* (`lifecycle_and_rewards.py`) but never run the
RL optimiser. This one does: it drives agenttune's real `TrlAgenticGrpo` trainer (TRL's
GRPOTrainer underneath, LoRA) to fine-tune SmolLM2-360M with a reward function, on the GPU.

The reward pays out ONLY for the correct label in the exact contract (`SENTIMENT=<label>`), so a
constant-label shortcut caps at ~1/3 and the policy has to actually classify. GRPO samples
several completions per prompt, scores them, and raises the probability of the higher-reward
ones. We generate before and after to show the policy moved — on both format and correctness.

Two dead ends worth calling out (they're in the git history): evaluating the trainer's in-memory
model gives garbage (it's left in a generation-hostile state — reload the saved adapter instead),
and a reward that gave partial credit for format-with-wrong-label got reward-hacked into always
answering "positive". The pure correct-label reward below is what actually made it classify.

Standard-mode GRPO (reward over text completions). The same `TrlAgenticGrpo` is what
`Project.train(fmt='grpo', rollout_engine=...)` wraps for the on-policy agentic-tool case.

Requires a GPU + SmolLM2-360M in the local HF cache. Run:
    python examples/agentic_grpo_real.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# This env's vllm wheel is ABI-incompatible with the installed torch, and TRL imports vllm even
# when use_vllm=False. Mark it unavailable so TRL takes the plain-transformers generation path.
# (importlib.util.find_spec returns None when sys.modules[name] is None.)
import sys

sys.modules["vllm"] = None

import re

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from agenttune.backends.trl.agentic.grpo.agentic_grpo import TrlAgenticGrpo

OUT_DIR = "/tmp/agenttune-grpo-student"

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
INSTRUCT = (
    "Classify the sentiment as positive, negative, or neutral. "
    "Reply with exactly SENTIMENT=<label> and nothing else. Review: {r!r}"
)
PROBE = INSTRUCT.format(r="the delivery was late and support ignored me")
LABELS = ("positive", "negative", "neutral")


def contract_reward(completions, label=None, **kwargs):
    """Reward the SENTIMENT=<label> contract, with partial credit for the right label word.

    Standard-mode GRPO passes completions as list[str] and forwards dataset columns (here
    `label`) as aligned lists. Reward variance across a prompt's samples is what GRPO learns from.
    """

    def as_text(comp):
        if isinstance(comp, list):  # conversational: last assistant message
            for m in reversed(comp):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    return str(m.get("content", ""))
            return ""
        return str(comp)

    labels = label if isinstance(label, list) else [label] * len(completions)
    out = []
    for comp, gold in zip(completions, labels, strict=False):
        t = as_text(comp).strip().lower()
        # Pure objective: reward ONLY the correct label in the exact contract. A constant-label
        # shortcut then caps at ~1/3 (only its own class), so the policy must actually classify.
        out.append(1.0 if re.match(rf"^sentiment={re.escape(str(gold))}\b", t) else 0.0)
    return out


def sample(model, tok, prompt, n=4, seed=0):
    """Sample n completions (the way GRPO scores them). RL policies are peaky, so greedy decode
    is unrepresentative — the reward is defined over samples, so we evaluate over samples too."""
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
        ("exceeded my expectations, five stars", "positive"),
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
    # Conversational prompts — let TRL apply the chat template once (passing a pre-templated
    # string would double-template during training and mismatch eval).
    prompts = [[{"role": "user", "content": INSTRUCT.format(r=r)}] for r, _ in reviews]
    ds = Dataset.from_dict({"prompt": prompts, "label": [lbl for _, lbl in reviews]})
    print(f"[data]    {len(ds)} prompts for on-policy sampling")

    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    before = sample(base, tok, PROBE, n=4)
    del base
    torch.cuda.empty_cache()

    grpo = TrlAgenticGrpo(
        model=MODEL,
        processing_class=tok,
        reward_funcs=contract_reward,
        train_dataset=ds,
        output_dir=OUT_DIR,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
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
    grpo.train()  # writes the trained LoRA adapter to OUT_DIR

    hist = [h for h in grpo.trainer.state.log_history if "reward" in h]
    r0, r1 = (hist[0]["reward"], hist[-1]["reward"]) if hist else (float("nan"), float("nan"))

    # Evaluate the SAVED adapter, loaded fresh — the trainer's in-memory model is left in a
    # generation-hostile state (gradient checkpointing / wrappers), so reloading is the honest
    # way to see the policy the run produced.
    del grpo
    torch.cuda.empty_cache()
    policy = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda"), OUT_DIR
    ).merge_and_unload()
    after = sample(policy, tok, PROBE, n=4)

    def rate(samples, pred):
        return sum(pred(s.strip().lower()) for s in samples) / len(samples)

    def is_contract(s):
        return s.startswith("sentiment=")

    def is_correct(s):
        return s.startswith("sentiment=negative")  # the probe review is negative

    print(f"[grpo]    {len(hist)} logged steps  |  mean reward {r0:.3f} -> {r1:.3f}")
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
