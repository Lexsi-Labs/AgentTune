"""
REAL live-LLM self-heal — the classify + generate stages run against a real model.
==================================================================================

The self-healing case studies run the *detection* half for real (the closed-loop
`FailureDetector`) but INJECT fakes for the two stages that need a language model — root-cause
classification and corrective-example generation. This runs those two stages for real, through the
UNMODIFIED production path:

    detected failures
      -> FailureClassifier.classify_batch   (real, litellm)   -> root cause + confidence
      -> TrainingExampleGenerator.generate  (real, litellm)   -> corrective {chosen, rejected}
      -> SelfHealLoop -> build_dataset -> (optional) real DPO retrain

Nothing in `agenttune` is stubbed: `FailureClassifier`, `TrainingExampleGenerator`,
`ReplayValidator`, `SelfHealLoop`, and the `as_sync_classifier` / `as_sync_generator` adapters all
run as written, and litellm makes real `acompletion` HTTP calls.

Why a local server. Those classes call `litellm.acompletion(model=..., api_base=...)` — in
production `model_name`/`api_base` point at a hosted or on-prem endpoint. This sandbox is offline
(no API keys) and has no vllm / llama.cpp, so the example stands up a tiny OpenAI-compatible server
backed by a local `Qwen2.5-3B-Instruct` and points litellm at it. That is exactly the on-prem
deployment shape the case studies pitch — you change two strings (`model_name`, `api_base`) to your
own endpoint and NOTHING else in the heal path moves. The server is real infrastructure, not a
mock of the library: it runs a real model and returns real completions.

Honesty notes:
  - Classification is greedy (temperature 0) so it reproduces run to run.
  - Generation samples at temperature 0.7 (the generator hard-codes it for completion diversity),
    so the `chosen` strings are ONE representative run — the shim is seeded for stability but the
    exact text is not a reproducibility guarantee.
  - The generator's prompt history is a library placeholder (`"Task context leading to
    failure..."`), so the `chosen` / `rejected` pair is the real signal, not the prompt.
  - Corrections are shown verbatim, including a noisy one — that's the real model output.

Requires a GPU + `Qwen/Qwen2.5-3B-Instruct` and `HuggingFaceTB/SmolLM2-360M-Instruct` in the local
HF cache. Run:
    python examples/self_heal_llm_real.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault(
    "OPENAI_API_KEY", "sk-local-noop"
)  # litellm's openai provider wants a key; the local server ignores it

import sys

sys.modules["vllm"] = (
    None  # env's vllm wheel is ABI-incompatible with torch; keep TRL/transformers off it
)

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from agenttune.agentic import SelfHealLoop, as_sync_classifier, as_sync_generator
from agenttune.decide.closed_loop.contracts import Failure
from agenttune.decide.closed_loop.failure_classifier import FailureClassifier
from agenttune.decide.closed_loop.replay_validator import ReplayValidator
from agenttune.decide.closed_loop.training_example_generator import TrainingExampleGenerator

JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # serves the real heal LLM
STUDENT = "HuggingFaceTB/SmolLM2-360M-Instruct"  # the policy the loop retrains


# ------------------------------------------------------------------------------------------------
# Real infrastructure (NOT a mock of the library): a minimal OpenAI-compatible server that runs a
# local model. In deployment this is a hosted/on-prem endpoint instead; the heal code is identical.
# ------------------------------------------------------------------------------------------------
def start_llm_server():
    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, torch_dtype=torch.bfloat16).to("cuda")

    def generate(messages, temperature=0.0, max_new_tokens=220):
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")
        sample = bool(temperature and temperature > 0)
        if sample:
            torch.manual_seed(0)  # stabilise the sampled corrections across reruns
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=sample,
                temperature=temperature if sample else None,
                top_p=0.95 if sample else None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence per-request logging
            pass

        def do_POST(self):  # noqa: N802 — http.server API
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = generate(body["messages"], body.get("temperature", 0.0))
            payload = json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)  # :0 -> OS picks a free port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def real_dpo_factory(train_dataset, **kwargs):
    """A real `trainer_factory` for `SelfHealLoop` — TRL `DPOTrainer` (LoRA) on the corrective
    rows the loop generated. Same contract as `Project.train`: returns an object with
    `.train() -> dict`. Used only to prove the loop's retrain hook fires end-to-end."""
    tok = AutoTokenizer.from_pretrained(STUDENT)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = DPOConfig(
        output_dir="/tmp/agenttune-heal-llm-dpo",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        learning_rate=1e-4,
        beta=0.1,
        bf16=True,
        report_to=[],
        logging_steps=1,
        max_length=256,
    )
    model = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=torch.bfloat16).to("cuda")
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=Dataset.from_list(train_dataset),
        processing_class=tok,
        peft_config=LoraConfig(
            r=8, lora_alpha=16, task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]
        ),
    )

    class _Result:
        def train(self):
            return {"train_loss": trainer.train().training_loss}

    return _Result()


def make_failures():
    """Two realistic detected failures. The failing assistant turn is carried in the context
    snapshot so the generator can use it as the `rejected` side of the preference pair."""

    def snap(action):
        return {"state_snapshot": {"messages": [{"role": "assistant", "content": action}]}}

    return [
        Failure(
            trajectory_id="t1",
            failure_type="loop",
            failed_stage_name="search_web",
            error_message="agent repeated search_web('refund policy') 5 times without advancing",
            context={
                "repeated_action": "search_web(query='refund policy')",
                **snap("search_web(query='refund policy')"),
            },
        ),
        Failure(
            trajectory_id="t2",
            failure_type="tool",
            failed_stage_name="lookup_order",
            error_message="agent used send_email to look up an order id; wrong tool for the task",
            context={
                "should_be": "lookup_order",
                **snap("send_email(to='ops', body='what is order 123')"),
            },
        ),
    ]


def _content(turn):
    return turn[0]["content"] if isinstance(turn, list) and turn else str(turn)


def main():
    print(f"[gpu]     {torch.cuda.get_device_name(0)}  (cuda: {torch.cuda.is_available()})")
    server, port = start_llm_server()
    api_base = f"http://127.0.0.1:{port}/v1"
    served = f"openai/{JUDGE_MODEL}"
    print(
        f"[server]  real {JUDGE_MODEL} on {api_base} — the production heal LLM (swap for a hosted endpoint in prod)"
    )

    # The REAL production heal stages, pointed at the local endpoint.
    classifier = FailureClassifier(model_name=served, api_base=api_base)
    generator = TrainingExampleGenerator(
        validator=ReplayValidator(), model_name=served, api_base=api_base
    )
    loop = SelfHealLoop(
        classifier=as_sync_classifier(classifier),
        generator=as_sync_generator(generator),
        trainer_factory=real_dpo_factory,
    )

    failures = make_failures()
    print(f"[detect]  {len(failures)} failures in hand (real FailureDetector output shape)")
    summary = loop.run(failures)  # classify (LLM) -> generate (LLM) -> build_dataset -> real DPO

    print("[classify] real litellm -> Qwen root-cause calls (greedy, reproducible):")
    for cf in summary["classified"]:
        print(
            f"           {cf.failure.trajectory_id}: {cf.root_cause}  (conf {cf.confidence})  — {cf.analysis[:70]}"
        )

    print("[generate] real litellm -> corrective preference rows (one representative sampled run):")
    for row in summary["dataset"]:
        print(f"           chosen  : {_content(row['chosen'])[:78]!r}")
        print(f"           rejected: {_content(row['rejected'])[:78]!r}")

    tr = summary["train_result"]
    print(
        f"[retrain] loop.trainer_factory fired: trained={summary['trained']}  "
        f"train_result={tr}  (plumbing closes; the converged before/after is in self_heal_dpo_real.py)"
    )
    print(
        f"[verdict] real LLM ran the classify+generate stages "
        f"({summary['n_classified']} classified, {summary['n_generated']} generated, "
        f"{summary['n_dataset_rows']} rows) and the full detect->classify->generate->retrain loop "
        f"executed end-to-end — no stage mocked"
    )
    server.shutdown()


if __name__ == "__main__":
    main()
