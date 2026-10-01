"""
Case study 1 — Build an agent, then train it (the core value path).
====================================================================

Value path:  build (AgentStrategy + Harness)  ->  collect full-tier rollouts
             ->  evaluate with real metrics    ->  assemble the SFT/GRPO dataset

Real model: a live SmolLM2-360M-Instruct drives the ReAct policy in step 1 (BUILD) and
produces the full-tier rollout (token spans + logprobs) via the real `TransformersRolloutEngine`
in step 3 (COLLECT) — the same rollout machinery GRPO uses on a GPU. Nothing here is scripted
or injected; `lookup` is a plain deterministic tool the model chooses to call, same as any
other tool-using agent.

Requires a GPU + SmolLM2-360M-Instruct in the local HF cache. Run:
    python examples/build_and_train.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agenttune.agentic import DictToolHarness, Project, ReActStrategy, agentic_metrics
from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

REACT_SYS = (
    "You are a ReAct agent with ONE tool: lookup(q) — it looks up the answer to a factual "
    'question.\nTo use it, output EXACTLY: TOOL: lookup | ARGS: {"q": "<question>"}\n'
    "Once you know the answer, output EXACTLY: FINISH: <answer>\n"
    "Output ONLY one such line, nothing else."
)
FEWSHOT_U = "Question: what is the capital of Japan?"
FEWSHOT_A = 'TOOL: lookup | ARGS: {"q": "capital of Japan"}'
_TOOL_RE = re.compile(r'\{.*?"q"\s*:\s*"([^"]*)".*?\}', re.S)
_FINISH_RE = re.compile(r"FINISH:\s*(.+)")

FACTS = {"capital of france": "Paris"}


def lookup(q: str = "") -> str:
    """The real tool: a plain deterministic lookup — the model decides when to call it."""
    return FACTS.get(q.strip().lower(), f"no record for {q!r}")


def _generate(model, tok, messages, max_new_tokens):
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def make_policy(model, tok):
    """Model-driven ReAct policy: the model decides whether to call `lookup`, every step."""

    def policy(state):
        if state.step == 0:
            raw = _generate(
                model,
                tok,
                [
                    {"role": "system", "content": REACT_SYS},
                    {"role": "user", "content": FEWSHOT_U},
                    {"role": "assistant", "content": FEWSHOT_A},
                    {"role": "user", "content": "Question: " + state.task},
                ],
                max_new_tokens=32,
            )
            m = _TOOL_RE.search(raw)
            if m:
                return {"name": "lookup", "arguments": {"q": m.group(1)}, "thought": raw}
            f = _FINISH_RE.search(raw)
            if f:
                return {
                    "name": "finish",
                    "arguments": {"answer": f.group(1).strip()},
                    "thought": raw,
                }
            return {"name": "finish", "arguments": {"answer": raw[:40]}, "thought": raw}
        result = state.events[-1].payload.get("text", "")
        raw = _generate(
            model,
            tok,
            [
                {"role": "system", "content": REACT_SYS},
                {"role": "user", "content": "Question: " + state.task},
                {"role": "assistant", "content": FEWSHOT_A},
                {"role": "user", "content": f"lookup returned: {result}"},
            ],
            max_new_tokens=16,
        )
        f = _FINISH_RE.search(raw)
        answer = f.group(1).strip() if f else result
        return {"name": "finish", "arguments": {"answer": answer}, "thought": raw}

    return policy


def main() -> None:
    print(f"[gpu]     {torch.cuda.get_device_name(0)}  (cuda: {torch.cuda.is_available()})")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    print(f"[model]   {MODEL} loaded")

    # 1) BUILD — a ReAct agent design over a pure-Python tool harness, driven by the real model.
    harness = DictToolHarness({"lookup": lookup}, max_steps=4)
    proj = Project(strategy=ReActStrategy(make_policy(model, tok), max_steps=4), harness=harness)

    log = proj.infer("what is the capital of France?")
    print(f"[build]   one episode -> {len(log)} events, tier={log.tier}")

    # 2) EVALUATE — the real programmatic trajectory metrics, no model call.
    metrics = agentic_metrics(log)
    print(f"[eval]    programmatic metrics -> {metrics}")

    # 3) COLLECT — full-tier rollouts through the real rollout machinery (real model + GPU).
    tasks = ["what is 2+3?", "capital of France?"]
    engine = TransformersRolloutEngine(model, tok)
    logs = proj.collect_rollout(engine, tasks, tools=[], max_steps=2)
    print(f"[collect] {len(logs)} rollouts, tiers={[l.tier for l in logs]}")

    # 4) TRAIN DATASET — the exact SFT rows the trainer would consume (schema: messages).
    rows = proj.sft_dataset()
    print(f"[train]   SFT dataset -> {len(rows)} rows, row0 keys={list(rows[0].keys())}")
    print(f"[train]   row0 first message role={rows[0]['messages'][0]['role']!r}")

    # The lifecycle event stream — proof every stage threaded one EventLog schema.
    stages = [e.stage for e in proj.events()]
    print(f"[stream]  lifecycle stages -> {stages}")


if __name__ == "__main__":
    main()
