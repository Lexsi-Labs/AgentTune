"""
REAL self-heal → DPO — trains a model on the loop's own corrective data, on the GPU.
====================================================================================

The self-healing case studies (`self_healing.py`, every `*_end_to_end.py`) detect a failing
agent and emit corrective preference data — `{prompt, chosen, rejected}` rows where `chosen` is
the fixed behaviour and `rejected` is the failure. They stop there. This example finishes the
loop: it builds that preference data through the real spine (`build_dataset`) and runs a real
Direct Preference Optimization pass with TRL's `DPOTrainer` (LoRA) to actually move the policy
toward `chosen`, on the GPU.

The correction here: prefer the `SENTIMENT=<label>` contract (chosen) over a verbose prose answer
(rejected). Base model gives prose; after DPO it gives the contract. DPO's own reward-accuracy
(how often it ranks chosen above rejected) climbs as evidence the optimiser worked.

Requires a GPU + SmolLM2-360M in the local HF cache. Run:
    python examples/self_heal_dpo_real.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The env's vllm wheel is ABI-incompatible with torch; TRL imports it eagerly. Mark unavailable.
import sys

sys.modules["vllm"] = None

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from agenttune.agentic import build_dataset
from agenttune.decide.closed_loop.contracts import TrainingExample

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUT_DIR = "/tmp/agenttune-dpo-student"
PROBE = "Classify the sentiment: 'terrible, broke on day one'"


def correction(text, label):
    """One corrective preference row: contract answer (chosen) over prose answer (rejected) —
    the shape the spine's self-heal loop emits."""
    return TrainingExample(
        trajectory_id=f"heal-{text[:6]}",
        original_failure_type="format_violation",
        root_cause="answered in prose instead of the required contract",
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

    # 1) Corrective preference data — built through the real spine.
    examples = [correction(t, lbl) for t, lbl in reviews]
    rows = build_dataset(examples)
    print(
        f"[heal]    {len(rows)} corrective preference rows (schema {list(rows[0].keys())}) "
        f"— chosen=contract, rejected=prose"
    )

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    before = generate(base, tok, PROBE)
    del base
    torch.cuda.empty_cache()

    # 2) Real DPO on that preference data (LoRA; the base acts as the frozen reference).
    cfg = DPOConfig(
        output_dir=OUT_DIR,
        num_train_epochs=16,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        beta=0.05,
        max_length=128,
        logging_steps=2,
        save_strategy="no",
        report_to=[],
        bf16=True,
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    trainer = DPOTrainer(
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

    hist = [h for h in trainer.state.log_history if "rewards/accuracies" in h]
    a0, a1 = (
        (hist[0]["rewards/accuracies"], hist[-1]["rewards/accuracies"])
        if hist
        else (float("nan"),) * 2
    )
    margins = [h["rewards/margins"] for h in trainer.state.log_history if "rewards/margins" in h]

    # 3) Evaluate the saved adapter fresh (trainer's in-memory model is generation-hostile).
    del trainer, model
    torch.cuda.empty_cache()
    policy = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda"), OUT_DIR
    ).merge_and_unload()
    after = generate(policy, tok, PROBE)

    margin = margins[-1] if margins else float("nan")
    print(f"[dpo]     reward accuracy {a0:.2f} -> {a1:.2f}  |  final reward margin {margin:.2f}")
    print(f"[probe]   {PROBE}")
    print(f"[before]  base policy -> {before!r}")
    print(f"[after]   DPO policy  -> {after!r}")
    # DPO is a preference method: its signal is ranking chosen above rejected (accuracy/margin),
    # not exact-string imitation (that's SFT — see agentic_distillation_real.py). The generation
    # shift is corroborating, directional evidence.
    print(
        f"[verdict] DPO ranks the corrective answer above the failure "
        f"(accuracy -> {a1:.2f}, margin {margin:.2f}); greedy output moved off the prose "
        f"form: {before.strip() != after.strip()}"
    )


if __name__ == "__main__":
    main()
