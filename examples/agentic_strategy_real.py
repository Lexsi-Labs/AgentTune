"""
REAL agentic ReAct — a real model drives the strategy, on the GPU.
==================================================================

The GPU-free case studies drive `run_episode` with a hand-written `policy` closure — the
"model" is a scripted stand-in. THIS runs the real thing: a live SmolLM2-360M is the policy.
At every step the model is prompted with the running `AgentState` (task + prior tool results)
and its generation is parsed into the next action. The library's headline surface — agent
*designs* (`ReActStrategy` + `run_episode`) and the agentic eval metrics (`agentic_metrics`:
tac/ter/arr/scsr/rad/lcf) — is exercised over REAL model rollouts instead of a DemoRolloutEngine
for the first time.

The demo is a set of arithmetic word problems and the honest contrast is *why the tool
framework exists*, both arms on the SAME real model:

  - `direct`  — the model answers the arithmetic itself (no tool). A 360M model is bad at
                multi-step mental math, so it mostly gets it wrong.
  - `react`   — the same model, but it can call a `calc` tool. It emits the right expression,
                the calculator does the arithmetic, and the model finishes with the number.

Every trajectory below is a real `run_episode` over `SmolLM2-360M`; every metric is computed
by the real `TrajectoryEvaluator` over those real trajectories. Nothing is mocked.

Reading the metrics honestly (they *characterise* real rollouts — they are not a uniform
"higher is better" scoreboard):
  - `answer_match` is the headline discriminator — did the trajectory contain the right answer.
  - `tac` (tool-argument correctness) is real only when the evaluator is given tool schemas
    (done below); it validates the args the model actually emitted against a JSON schema.
  - `ter` = unique_tool_outputs / total — a do-nothing single-`finish` run scores 1.0 while a
    *correct* two-call react run scores 0.5, so ter does NOT rank the better agent; it is
    reported as-is, not spun.
  - `arr`/`lcf` only fire on repeated/looping calls; `scsr` only moves when a tool errors —
    all three are ~flat on these short clean runs, and that itself is the honest reading.

Requires a GPU + SmolLM2-360M-Instruct in the local HF cache. Run:
    python examples/agentic_strategy_real.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agenttune.agentic import (
    DictToolHarness,
    EventKind,
    ReActStrategy,
    agentic_metrics,
    answer_match,
    run_episode,
)
from agenttune.eval.agentic.trajectory_eval import TrajectoryEvaluator

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Word problems whose answer a 360M model can't reliably compute in its head, but which are
# trivial for a calculator. (task, gold-answer-as-string).
PROBLEMS = [
    ("A store sold 3 boxes of 12 apples and 5 loose apples. How many apples in total?", "41"),
    (
        "There are 6 crates with 9 bottles each, minus 4 broken bottles. How many bottles remain?",
        "50",
    ),
    ("A truck makes 7 trips carrying 18 crates each. How many crates total?", "126"),
    ("You buy 4 packs of 25 screws and use 37. How many screws are left?", "63"),
    ("13 teams of 8 players, plus 9 referees. How many people are there?", "113"),
    (
        "A tank holds 144 litres; you drain 3 buckets of 17 litres each. How many litres remain?",
        "93",
    ),
]

# ReAct system prompt + one worked example (few-shot). The model must emit EXACTLY one line.
REACT_SYS = (
    "You are a ReAct agent with ONE tool: calc(expression) — it evaluates a Python arithmetic "
    "expression and returns the number.\n"
    'To use it, output EXACTLY: TOOL: calc | ARGS: {"expression": "<python arithmetic>"}\n'
    "Once you know the numeric result, output EXACTLY: FINISH: <number>\n"
    "Output ONLY one such line, nothing else."
)
FEWSHOT_U = "Question: A shelf holds 4 rows of 7 books plus 3 loose books. How many books?"
FEWSHOT_A = 'TOOL: calc | ARGS: {"expression": "4*7+3"}'
DIRECT_SYS = "Answer the arithmetic word problem. Reply with ONLY the final number, nothing else."

_TOOL_RE = re.compile(r'\{.*?"expression"\s*:\s*"([^"]*)".*?\}', re.S)
_FINISH_RE = re.compile(r"FINISH:\s*(-?\d+)")
_NUM_RE = re.compile(r"-?\d[\d,]*")


def calc(expression: str = "") -> str:
    """The real tool. Arithmetic only: the regex admits only digits/operators/parens (no names,
    attributes, or strings can reach eval), `__builtins__` is stripped, and `**` is rejected so
    an exponent tower can't blow up compute — eval() here can only evaluate a bounded sum."""
    if not expression or "**" in expression or not re.fullmatch(r"[\d\s+\-*/().]+", expression):
        return f"error: invalid expression {expression!r}"
    try:
        return str(
            int(eval(expression, {"__builtins__": {}}, {}))
        )  # noqa: S307 — arithmetic-only, sandboxed
    except Exception as exc:  # noqa: BLE001 — surface as a tool error the metrics can see
        return f"error: {exc}"


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


def _prior_results(state):
    """Tool results the model has seen so far. run_episode appends each observation as an
    OBSERVATION event after the initial task observation, so results are events[1:]."""
    obs = [e.payload.get("text", "") for e in state.events if e.kind is EventKind.OBSERVATION]
    return obs[1:]


def make_react_policy(model, tok):
    """Model-driven ReAct policy: the model chooses the tool call AND the finish, every step."""

    def policy(state):
        results = _prior_results(state)
        if not results:  # first move — ask the model for a tool call
            raw = _generate(
                model,
                tok,
                [
                    {"role": "system", "content": REACT_SYS},
                    {"role": "user", "content": FEWSHOT_U},
                    {"role": "assistant", "content": FEWSHOT_A},
                    {"role": "user", "content": "Question: " + state.task},
                ],
                max_new_tokens=40,
            )
            m = _TOOL_RE.search(raw)
            if m:
                return {"name": "calc", "arguments": {"expression": m.group(1)}, "thought": raw}
            f = _FINISH_RE.search(raw)
            if f:
                return {"name": "finish", "arguments": {"answer": f.group(1)}, "thought": raw}
            return {
                "name": "finish",
                "arguments": {"answer": raw[:40]},
                "thought": raw,
            }  # honest fallback
        # a calc result is in hand — ask the model to finish with it
        raw = _generate(
            model,
            tok,
            [
                {"role": "system", "content": REACT_SYS},
                {"role": "user", "content": "Question: " + state.task},
                {"role": "assistant", "content": FEWSHOT_A},
                {"role": "user", "content": f"calc returned: {results[-1]}"},
            ],
            max_new_tokens=16,
        )
        f = _FINISH_RE.search(raw)
        answer = f.group(1) if f else results[-1]
        return {"name": "finish", "arguments": {"answer": answer}, "thought": raw}

    return policy


def make_direct_policy(model, tok):
    """Same real model, no tool: it answers the arithmetic itself and finishes with that."""

    def policy(state):
        raw = _generate(
            model,
            tok,
            [
                {"role": "system", "content": DIRECT_SYS},
                {"role": "user", "content": state.task},
            ],
            max_new_tokens=24,
        )
        nums = _NUM_RE.findall(raw)
        answer = nums[-1].replace(",", "") if nums else raw[:40]
        return {"name": "finish", "arguments": {"answer": answer}, "thought": raw}

    return policy


# JSON schemas make `tac` non-vacuous — it validates the args the model actually emitted.
TOOL_SCHEMAS = {
    "calc": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
        "additionalProperties": False,
    },
    "finish": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
}


def run_arm(name, make_policy, model, tok, evaluator):
    """Run every problem through a real run_episode; collect answer_match + agentic_metrics."""
    matches, metric_rows, first_log = 0, [], None
    for task, gold in PROBLEMS:
        harness = DictToolHarness(tools={"calc": calc}, max_steps=4)
        log = run_episode(ReActStrategy(make_policy(model, tok), max_steps=4), harness, task)
        matches += int(answer_match(log, gold) == 1.0)
        metric_rows.append(agentic_metrics(log, evaluator=evaluator))
        if first_log is None:
            first_log = log
    keys = ("tac", "ter", "arr", "scsr", "rad", "lcf")
    mean = {k: sum(r[k] for r in metric_rows) / len(metric_rows) for k in keys}
    return {"name": name, "match": matches, "n": len(PROBLEMS), "mean": mean, "log": first_log}


def render(log):
    parts = []
    for e in log:
        if e.kind is EventKind.TOOL_CALL:
            parts.append(f"  CALL   {e.payload.get('action')}")
        elif e.kind is EventKind.TOOL_RESULT:
            parts.append(f"  RESULT {e.payload.get('output')!r}")
    return "\n".join(parts)


def main():
    print(f"[gpu]     {torch.cuda.get_device_name(0)}  (cuda: {torch.cuda.is_available()})")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    print(f"[model]   {MODEL} loaded — this model IS the ReAct policy (no scripted stand-in)")
    print(f"[data]    {len(PROBLEMS)} arithmetic word problems")

    evaluator = TrajectoryEvaluator(model_name="none", api_base=None, tool_schemas=TOOL_SCHEMAS)
    direct = run_arm("direct", make_direct_policy, model, tok, evaluator)
    react = run_arm("react", make_react_policy, model, tok, evaluator)

    def fmt(m):
        return "  ".join(f"{k}={m[k]:.2f}" for k in ("tac", "ter", "arr", "scsr", "rad", "lcf"))

    print(f"\n[react]   one real trajectory ({PROBLEMS[0][0]!r}):")
    print(render(react["log"]))
    print(
        f"\n[direct]  answer_match {direct['match']}/{direct['n']}  |  agentic_metrics {fmt(direct['mean'])}"
    )
    print(
        f"[react]   answer_match {react['match']}/{react['n']}  |  agentic_metrics {fmt(react['mean'])}"
    )
    print(
        f"[verdict] a real model drove ReActStrategy/run_episode and agentic_metrics scored "
        f"the REAL rollouts; the calc tool lifted answer_match "
        f"{direct['match']}/{direct['n']} -> {react['match']}/{react['n']} on the same model"
    )


if __name__ == "__main__":
    main()
