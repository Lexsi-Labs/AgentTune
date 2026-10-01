"""
REAL agentic distillation — actually trains a model on the GPU.
===============================================================

Unlike the GPU-free case studies (which inject a stand-in trainer), this one runs the whole path
for real: a teacher's demonstrations become full-tier trajectories, `Project.distill` assembles
the SFT dataset from them, and a REAL `trainer_factory` loads a small student
(HuggingFaceTB/SmolLM2-360M-Instruct) and fine-tunes it with TRL's SFTTrainer on the GPU.

To make the learning visible, the teacher demonstrates an unusual output contract:
answer every sentiment request with exactly `SENTIMENT=<label>`. The base model does not do this;
after distillation it does. We generate before and after to show the weights actually changed.

Requires a GPU + the model in the local HF cache. Run:
    python examples/agentic_distillation_real.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from agenttune.agentic import Event, EventKind, EventLog, Project

STUDENT = "HuggingFaceTB/SmolLM2-360M-Instruct"
PROBE = "Classify sentiment: 'the delivery was late and support ignored me'"


def teacher_demo(text: str, label: str) -> EventLog:
    """One full-tier teacher trajectory: a request (user) and the demonstrated answer (assistant)."""
    log = EventLog(tier="full")
    log.append(Event(EventKind.OBSERVATION, {"text": f"Classify sentiment: '{text}'"}))
    log.append(Event(EventKind.TEXT, {"text": f"SENTIMENT={label}"}))
    return log


def generate(model, tok, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()


class RealSFTTrainerFactory:
    """A real trainer_factory: loads the student and fine-tunes it with TRL on the GPU."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.model = None
        self.tok = None
        self.before = None

    def __call__(self, *, model, train_dataset, **kwargs):
        self.tok = AutoTokenizer.from_pretrained(model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to(
            "cuda"
        )
        self.before = generate(self.model, self.tok, PROBE)

        cfg = SFTConfig(
            output_dir=self.out_dir,
            num_train_epochs=25,
            per_device_train_batch_size=4,
            learning_rate=3e-4,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
            bf16=True,
            max_length=256,
            completion_only_loss=True,
            gradient_checkpointing=False,
        )
        # LoRA keeps the base model's language intact and only learns the new contract.
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        return SFTTrainer(
            model=self.model,
            args=cfg,
            train_dataset=Dataset.from_list(train_dataset),
            processing_class=self.tok,
            peft_config=lora,
        )


def main():
    print(
        f"[gpu]      {torch.cuda.get_device_name(0)}  (cuda available: {torch.cuda.is_available()})"
    )

    # 1) Teacher demonstrations -> full-tier trajectories -> the real spine holds them.
    demos = [
        ("I love this product, best purchase ever", "positive"),
        ("absolutely fantastic experience", "positive"),
        ("the team was helpful and kind", "positive"),
        ("exceeded my expectations, five stars", "positive"),
        ("smooth setup and great value", "positive"),
        ("delighted with the quality", "positive"),
        ("this is the worst thing I've bought", "negative"),
        ("terrible, broke on day one", "negative"),
        ("support was rude and unhelpful", "negative"),
        ("waste of money, deeply disappointed", "negative"),
        ("shipping was slow and item was damaged", "negative"),
        ("frustrating and buggy, would not recommend", "negative"),
        ("it arrived on time, works as described", "neutral"),
        ("standard packaging, nothing special", "neutral"),
        ("does the job, no strong feelings", "neutral"),
        ("average product, meets the basics", "neutral"),
    ]
    teacher = Project()
    for text, label in demos:
        teacher.add_trajectory(teacher_demo(text, label))
    rows = teacher.sft_dataset()
    print(
        f"[teacher]  {len(teacher.trajectories)} full-tier trajectories -> {len(rows)} SFT rows "
        f"(schema {list(rows[0].keys())})"
    )

    # 2) REAL distillation: Project.distill assembles the dataset and calls the real trainer.
    factory = RealSFTTrainerFactory(out_dir="/tmp/agenttune-real-student")
    result = teacher.distill(STUDENT, trainer_factory=factory)

    # 3) Show the training actually happened and the student learned the contract.
    print(
        f"[distill]  TRL SFTTrainer finished: {result.metrics.get('train_runtime', '?'):.1f}s, "
        f"final train_loss={result.training_loss:.3f}"
    )
    after = generate(factory.model, factory.tok, PROBE)
    print(f"[probe]    prompt: {PROBE}")
    print(f"[before]   base model  -> {factory.before!r}")
    print(f"[after]    distilled   -> {after!r}")
    print(f"[verdict]  learned the SENTIMENT= contract: {after.startswith('SENTIMENT=')}")


if __name__ == "__main__":
    main()
