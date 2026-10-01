"""
REAL self-heal reward — the TAC/TER ranking that picks chosen vs rejected.
=========================================================================

The self-heal generator (`decide.closed_loop.TrainingExampleGenerator`) turns a
classified failure into a preference pair by *scoring* several synthesized
corrections and taking best-reward → `chosen`, worst → `rejected`. The score is
`TAC (Tool Argument Correctness) + TER (Tool Efficacy Reward)`.

That reward was never run for real. `_process_single` built a **mock trajectory**
that shoved the raw completion string in as `tool_calls` ("Passing raw string to
test TAC fallback" / "In a real app, parse comp_text as JSON"). `_calculate_tac`'s
string branch then returns **1.0 for every completion** regardless of whether the
corrected tool call is actually valid — so TAC could not tell a good correction
from a malformed one, and the preference was driven by TER alone.

Two library fixes make the reward real:
  1. `_process_single` now parses the correction as JSON into a real
     `{name, arguments}` tool call, so TAC validates the corrected arguments.
  2. `TrainingExampleGenerator(tool_schemas=...)` threads schemas to the evaluator
     (without them TAC has nothing to validate against — so the parsed path only
     engages when schemas are supplied; the schema-less legacy path is unchanged).

Part 1 proves the reward now *discriminates* (deterministic, no model). Part 2
runs the real generator end-to-end against a live local LLM with real schemas, so
the fix is exercised in the actual heal wiring.

HONEST SCOPE — read before quoting
----------------------------------
The claim is that the self-heal reward is **real and discriminating**: a
well-formed corrected tool call scores strictly higher TAC than a malformed one,
so `chosen`/`rejected` is meaningful — which the old raw-string path (TAC≡1.0)
could not deliver. Part 2 shows the real generator producing real, schema-scored
rewards over a live LLM's corrections; it is not a training-convergence claim (the
end-to-end retrain-from-heal is `self_heal_dpo_real.py` / `self_heal_llm_real.py`).

Part 2 needs a GPU + `Qwen/Qwen2.5-3B-Instruct` cached. Run:
    python examples/self_heal_reward_real.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENAI_API_KEY", "sk-local-noop")

import sys

sys.modules["vllm"] = None

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agenttune.decide.closed_loop.contracts import ClassifiedFailure, Failure
from agenttune.decide.closed_loop.replay_validator import ReplayValidator
from agenttune.decide.closed_loop.training_example_generator import (
    TrainingExampleGenerator,
    _parse_tool_call,
)
from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator

MODEL = "Qwen/Qwen2.5-3B-Instruct"

# A real tool schema: `lookup_order` requires an integer `order_id`.
LOOKUP_SCHEMA = {
    "type": "object",
    "properties": {"order_id": {"type": "integer"}},
    "required": ["order_id"],
    "additionalProperties": False,
}
TOOL_SCHEMAS = {"lookup_order": LOOKUP_SCHEMA}


def part1_reward_discriminates():
    """Deterministic: the fixed TAC reward scores a valid correction above a malformed one."""
    ev = TrajectoryEvaluator(tool_schemas=TOOL_SCHEMAS)

    def tac_parsed(text):  # the FIXED heal path: parse → validate args against schema
        call = _parse_tool_call(text)
        return ev._calculate_tac({"tool_calls": [call if call is not None else text]})

    def tac_raw(text):  # the OLD heal path: raw string → _calculate_tac string fallback
        return ev._calculate_tac({"tool_calls": [text]})

    valid = '{"name": "lookup_order", "arguments": {"order_id": 123}}'
    malformed = '{"name": "lookup_order", "arguments": {"order_id": "not-a-number"}}'

    print("\n── Part 1: does the self-heal reward discriminate? (deterministic) ──")
    print(f"  correction A (valid, order_id=123)          : {valid}")
    print(f"  correction B (malformed, order_id='...')    : {malformed}")
    print(
        f"  OLD raw-string TAC  → A={tac_raw(valid):.2f}  B={tac_raw(malformed):.2f}   "
        f"(both 1.0 — vacuous, can't rank)"
    )
    print(
        f"  NEW parsed TAC      → A={tac_parsed(valid):.2f}  B={tac_parsed(malformed):.2f}   "
        f"(validates args against the schema)"
    )

    assert (
        tac_raw(valid) == tac_raw(malformed) == 1.0
    ), "old path was expected to be vacuous (both 1.0)"
    assert tac_parsed(valid) > tac_parsed(malformed), "fixed reward must rank valid above malformed"
    assert tac_parsed(malformed) < 1.0, "malformed args must not score a perfect TAC"
    print("  ✓ the fixed reward ranks a well-formed correction above a malformed one")


def start_llm_server():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")

    def generate(messages, temperature=0.0, max_new_tokens=96):
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to("cuda")
        sample = bool(temperature and temperature > 0)
        if sample:
            torch.manual_seed(0)
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
        def log_message(self, *a):
            pass

        def do_POST(self):  # noqa: N802
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

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def part2_real_generator():
    """The real generator scores a live LLM's corrections through the fixed reward path."""
    print("\n── Part 2: real generator + live LLM + real schemas (end-to-end) ──")
    server, port = start_llm_server()
    api_base = f"http://127.0.0.1:{port}/v1"
    served = f"openai/{MODEL}"
    print(f"  local {MODEL} served on {api_base} (swap for a hosted endpoint in prod)")

    # A real classified failure: the agent used the wrong tool; it should have called
    # lookup_order with an integer order_id.
    failure = Failure(
        trajectory_id="order-42",
        failure_type="tool",
        failed_stage_name="lookup_order",
        error_message="agent called send_email to look up an order instead of lookup_order",
        context={
            "should_be": "lookup_order",
            "state_snapshot": {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "send_email(to='ops', body='what is order 42?')",
                    }
                ]
            },
        },
    )
    classified = ClassifiedFailure(
        failure=failure,
        root_cause="wrong_tool",
        confidence=0.9,
        analysis="Used send_email; should have called lookup_order(order_id=<int>).",
    )

    generator = TrainingExampleGenerator(
        validator=ReplayValidator(),
        model_name=served,
        api_base=api_base,
        tool_schemas=TOOL_SCHEMAS,  # ← makes TAC validate the corrected arguments
    )
    examples = asyncio.run(generator.generate_batch([classified]))
    server.shutdown()

    assert examples, "generator produced no training example from the failure"
    ex = examples[0]
    print(f"  synthesized {len(ex.completions)} corrections; real schema-scored rewards:")
    for comp, r in zip(ex.completions, ex.rewards, strict=False):
        text = comp[0]["content"]
        parsed = _parse_tool_call(text)
        tag = "parsed-tool-call" if parsed is not None else "raw-string"
        print(f"    reward={r:+.2f}  [{tag}]  {text[:72]!r}")
    print(f"  chosen  : {(ex.chosen or [{'content': None}])[0]['content']!r}")
    print(f"  rejected: {(ex.rejected or [{'content': None}])[0]['content']!r}")

    assert ex.rewards, "no rewards computed"
    assert ex.completions and len(ex.completions) == len(
        ex.rewards
    ), "completions/rewards misaligned"
    # At least one correction parsed as a real tool call and was schema-scored (not the
    # raw-string fallback) — i.e. the fixed reward path genuinely engaged on real output.
    parsed_any = any(_parse_tool_call(c[0]["content"]) is not None for c in ex.completions)
    print(f"  a correction parsed as a real tool call (schema-scored): {parsed_any}")
    assert parsed_any, (
        "no live correction parsed as a tool call — the fixed reward path "
        "never engaged; broaden _parse_tool_call for the shapes this LLM emits"
    )
    return parsed_any


def main():
    part1_reward_discriminates()
    parsed_any = part2_real_generator()

    print("\n── Summary ──")
    print("  reward discriminates valid vs malformed (Part 1)        : True")
    print("  real generator scored a live LLM's corrections (Part 2) : True")
    print(f"  a real tool call was schema-scored, not raw-string      : {parsed_any}")
    print(
        "  scope: the self-heal reward is real & discriminating; retrain-convergence is elsewhere"
    )
    print("\n✓ Self-heal TAC/TER reward ran for real and ranks corrections meaningfully.")


if __name__ == "__main__":
    main()
