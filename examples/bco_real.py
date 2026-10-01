"""
REAL BCO — Binary Classifier Optimization from unpaired feedback, on the GPU.
============================================================================

BCO (Binary Classifier Optimization), like KTO, learns from unpaired thumbs-up / thumbs-down
labels rather than {chosen, rejected} pairs — but it optimises a binary classifier over the
policy's log-ratios rather than KTO's Kahneman-Tversky value. This trains a real
`trl.experimental.bco.BCOTrainer` (LoRA) on SmolLM2-360M: contract answers desirable, prose
answers undesirable, unpaired.

(PPO and BCO live under `trl.experimental` in TRL 1.6 — importing from the old top-level path
raises AttributeError; the experimental path is the working one.)

Requires a GPU + SmolLM2-360M in the local HF cache. Run:
    python examples/bco_real.py
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
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.bco.bco_config import BCOConfig
from trl.experimental.bco.bco_trainer import BCOTrainer

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUT_DIR = "/tmp/agenttune-bco-student"
PROBE = "Classify the sentiment: 'terrible, broke on day one'"


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
    # Unpaired binary feedback: contract answer = desirable, prose answer = undesirable.
    rows = []
    for text, lbl in reviews:
        prompt = [{"role": "user", "content": f"Classify the sentiment: '{text}'"}]
        rows.append(
            {
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": f"SENTIMENT={lbl}"}],
                "label": True,
            }
        )
        rows.append(
            {
                "prompt": prompt,
                "completion": [
                    {"role": "assistant", "content": f"The sentiment of this review is {lbl}."}
                ],
                "label": False,
            }
        )
    print(
        f"[data]    {len(rows)} unpaired rows ({sum(r['label'] for r in rows)} desirable / "
        f"{sum(not r['label'] for r in rows)} undesirable)"
    )

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    before = generate(base, tok, PROBE)
    del base
    torch.cuda.empty_cache()

    cfg = BCOConfig(
        output_dir=OUT_DIR,
        num_train_epochs=12,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        beta=0.1,
        max_length=128,
        logging_steps=2,
        save_strategy="no",
        report_to=[],
        bf16=True,
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    trainer = BCOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=Dataset.from_list(rows),
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

    # Snapshot before `del trainer` below — a closure over `trainer` itself
    # would NameError if `last()` were ever called after the del.
    log_history = trainer.state.log_history

    def last(metric):
        vals = [h[metric] for h in log_history if metric in h]
        return vals[-1] if vals else float("nan")

    r_chosen, r_rejected = last("rewards/chosen"), last("rewards/rejected")

    del trainer, model
    torch.cuda.empty_cache()
    policy = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda"), OUT_DIR
    ).merge_and_unload()
    after = generate(policy, tok, PROBE)

    print(
        f"[bco]     implicit reward — desirable {r_chosen:+.2f} vs undesirable {r_rejected:+.2f} "
        f"(margin {r_chosen - r_rejected:+.2f})"
    )
    print(f"[probe]   {PROBE}")
    print(f"[before]  base policy -> {before!r}")
    print(f"[after]   BCO policy  -> {after!r}")
    print(
        f"[verdict] BCO scores desirable above undesirable: {r_chosen > r_rejected}; "
        f"greedy output moved off prose: {before.strip() != after.strip()}"
    )


if __name__ == "__main__":
    main()
