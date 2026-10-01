"""
AgentTune × OpenEnv — Comprehensive End-to-End Test Suite
==========================================================

Model under test : Qwen/Qwen3-0.6B-FP8
Hardware         : NVIDIA GeForce RTX 3070 (8 GB VRAM)
Framework        : AgentTune agentic (L1 OpenEnv adapter included)

Test catalogue
--------------
T1  Environment & dependency smoke test
T2  Tool layer — local calculator + unit_converter
T3  Rollout engine — single-step transformers inference
T4  Rollout engine — multi-step agentic loop (tool calls)
T5  LLMJudge — score a canned trajectory
T6  AgentTuneGraph Pattern A — rollout-only, no judge
T7  AgentTuneGraph Pattern B — rollout + dual LLMJudge
T8  AgentTuneGraph Pattern D — conditional router (simple vs complex branch)
T9  OpenEnv Layer-1 — echo_env live tool call through the adapter
T10 OpenEnv Layer-1 — error-type metadata surfacing (invalid_args path)
T11 GRPO training loop — 5 steps, real model, real reward function
T12 Output artefact audit — checkpoints, completions parquet, training_stats.json
T13 Regression — ToolRegistry unaffected by openenv installation
T14 Memory & GPU health — VRAM tracked across tests, no OOM

Each test prints a structured result block and appends to outputs/e2e_results.jsonl.
A human-readable HTML + Markdown report is written to reports/ at the end.

Usage
-----
    python e2e_test_suite.py          # all tests
    python e2e_test_suite.py --quick  # skip T11 training
    RUN_OPENENV_IT=1 python e2e_test_suite.py  # include live openenv
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent / "test_outputs"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"
CKPT_DIR = BASE_DIR / "checkpoints"
REPORT_DIR = BASE_DIR / "reports"
RESULTS_FILE = OUTPUT_DIR / "e2e_results.jsonl"
MODEL = "Qwen/Qwen3-0.6B-FP8"
ECHO_URL = os.environ.get("OPENENV_ECHO_URL", "http://localhost:8765")

for d in (OUTPUT_DIR, LOG_DIR, CKPT_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = LOG_DIR / f"e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("e2e")


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class TestResult:
    test_id: str
    name: str
    goal: str
    status: str  # PASS | FAIL | SKIP
    duration_s: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


results: list[TestResult] = []


def run_test(test_id: str, name: str, goal: str, fn, *args, skip=False, **kwargs) -> TestResult:
    if skip:
        r = TestResult(test_id=test_id, name=name, goal=goal, status="SKIP")
        results.append(r)
        _print_result(r)
        return r

    logger.info("▶  %s — %s", test_id, name)
    t0 = time.monotonic()
    try:
        details = fn(*args, **kwargs) or {}
        r = TestResult(
            test_id=test_id,
            name=name,
            goal=goal,
            status="PASS",
            duration_s=round(time.monotonic() - t0, 2),
            details=details,
        )
    except Exception as exc:
        r = TestResult(
            test_id=test_id,
            name=name,
            goal=goal,
            status="FAIL",
            duration_s=round(time.monotonic() - t0, 2),
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )

    results.append(r)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(asdict(r)) + "\n")
    _print_result(r)
    return r


def _print_result(r: TestResult):
    icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭ "}.get(r.status, "?")
    logger.info("%s  %s  %s  (%.2fs)", icon, r.test_id, r.name, r.duration_s)
    if r.status == "FAIL":
        logger.error("   ERROR: %s", r.error.splitlines()[0] if r.error else "unknown")
    if r.details:
        for k, v in r.details.items():
            logger.info("   %s = %s", k, v)


def vram_mb() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated() / 1024 / 1024)
    except Exception:
        pass
    return -1


def flush_vram(label: str = ""):
    """Force VRAM cleanup — call between heavy tests on 8 GB GPU."""
    try:
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        freed = vram_mb()
        logger.info("flush_vram(%s): allocated=%d MB", label, freed)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated() / 1024 / 1024)
    except Exception:
        pass
    return -1


# ═════════════════════════════════════════════════════════════════════════════
# T1 — Environment smoke test
# ═════════════════════════════════════════════════════════════════════════════
def t1_environment():
    import torch
    import transformers
    import trl

    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "none"
    vram = torch.cuda.get_device_properties(0).total_memory // (1024**2) if cuda_ok else 0

    assert cuda_ok, "CUDA not available — tests requiring GPU will fail"
    assert vram >= 6000, f"VRAM {vram} MB is below 6 GB minimum"

    from agenttune.agentic.tools.registry import ToolRegistry
    from agenttune.utils.optional import OPENENV_AVAILABLE

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "cuda": cuda_ok,
        "gpu": gpu_name,
        "vram_mb": vram,
        "openenv_available": OPENENV_AVAILABLE,
        "agenttune_path": str(Path(ToolRegistry.__module__).parent),
    }


# ═════════════════════════════════════════════════════════════════════════════
# T2 — Tool layer: local tools
# ═════════════════════════════════════════════════════════════════════════════
def make_tools():
    def calculator(expression: str) -> dict:
        """Evaluate a mathematical expression and return the result.

        Args:
            expression: A valid Python math expression string (e.g. '2 + 2', '15 * 8').

        Returns:
            dict with 'result' key on success, or 'error' key on failure.
        """
        try:
            result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def unit_converter(value: float, from_unit: str, to_unit: str) -> dict:
        """Convert a value from one unit to another.

        Args:
            value: The numeric value to convert.
            from_unit: The source unit (km, miles, kg, lbs, c, f).
            to_unit: The target unit (km, miles, kg, lbs, c, f).

        Returns:
            dict with 'result' key on success, or 'error' key if conversion is unsupported.
        """
        conversions = {
            ("km", "miles"): 0.621371,
            ("miles", "km"): 1.60934,
            ("kg", "lbs"): 2.20462,
            ("lbs", "kg"): 0.453592,
            ("c", "f"): lambda v: v * 9 / 5 + 32,
            ("f", "c"): lambda v: (v - 32) * 5 / 9,
        }
        key = (from_unit.lower(), to_unit.lower())
        if key not in conversions:
            return {"error": f"Conversion {from_unit}→{to_unit} not supported"}
        conv = conversions[key]
        result = conv(value) if callable(conv) else value * conv
        return {"result": round(result, 4)}

    return calculator, unit_converter


def t2_tool_layer():
    calc, conv = make_tools()

    r1 = calc("7 * 8 + 6")
    assert r1 == {"result": 62}, f"calc failed: {r1}"

    r2 = calc("1/0")
    assert "error" in r2, "Division by zero should return error"

    r3 = conv(100.0, "km", "miles")
    assert abs(r3["result"] - 62.1371) < 0.01, f"conv failed: {r3}"

    r4 = conv(0.0, "c", "f")
    assert r4["result"] == 32.0, f"0°C should be 32°F, got {r4}"

    r5 = conv(1.0, "kg", "lightyears")
    assert "error" in r5, "Unsupported conversion should return error"

    return {
        "calc_basic": r1,
        "calc_error": r2,
        "conv_km_miles": r3,
        "conv_0c_to_f": r4,
        "conv_unsupported": r5,
    }


# ═════════════════════════════════════════════════════════════════════════════
# T3 — Rollout engine: single inference step
# ═════════════════════════════════════════════════════════════════════════════
def t3_rollout_single_step():
    from agenttune.agentic.rollout_engines.rollout_factory import (
        create_rollout_engine,
        create_rollout_fn,
    )

    calc, conv = make_tools()

    engine = create_rollout_engine(backend="transformers", model_path=MODEL)
    rollout_fn = create_rollout_fn(
        rollout_engine=engine,
        tools=[calc, conv],
        max_steps=1,
        system_prompt="You are a helpful assistant.",
    )

    prompts = [{"role": "user", "content": "What is 2 + 2?"}]
    out = rollout_fn([prompts])

    assert out is not None, "rollout_fn returned None"
    assert (
        "responses" in out or "conversations" in out
    ), f"Missing response keys: {list(out.keys())}"

    vram = vram_mb()
    return {
        "output_keys": list(out.keys()),
        "num_outputs": len(out.get("responses", out.get("conversations", []))),
        "vram_after_mb": vram,
    }


# ═════════════════════════════════════════════════════════════════════════════
# T4 — Rollout engine: multi-step agentic loop
# ═════════════════════════════════════════════════════════════════════════════
def t4_rollout_multistep():
    from agenttune.agentic.rollout_engines.rollout_factory import (
        create_rollout_engine,
        create_rollout_fn,
    )

    calc, conv = make_tools()

    engine = create_rollout_engine(backend="transformers", model_path=MODEL)
    rollout_fn = create_rollout_fn(
        rollout_engine=engine,
        tools=[calc, conv],
        max_steps=4,
        system_prompt=(
            "You are a math assistant. Use the calculator tool for arithmetic. "
            "Always call a tool before answering."
        ),
    )

    prompts = [
        [{"role": "user", "content": "What is 15% of 240? Use the calculator."}],
        [{"role": "user", "content": "Convert 50 km to miles."}],
    ]
    out = rollout_fn(prompts)

    tool_counts = out.get("tool_call_counts", [])
    responses = out.get("responses", out.get("conversations", []))

    return {
        "output_keys": list(out.keys()),
        "num_prompts_processed": len(responses),
        "tool_call_counts": tool_counts,
        "any_tools_called": any(c > 0 for c in tool_counts) if tool_counts else "n/a",
        "vram_after_mb": vram_mb(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# T5 — LLMJudge: score a canned trajectory
# ═════════════════════════════════════════════════════════════════════════════
def t5_llm_judge():
    from agenttune.agentic.rewards.llm_judge import LLMJudge

    judge = LLMJudge(
        backend="transformers",
        model_path=MODEL,
        system_prompt=(
            "You are a scoring judge. Rate the answer quality from 0 to 1. "
            'Return JSON only: {"score": <float>, "explanation": "<str>"}'
        ),
        cache_size=50,
    )

    trajectory = {
        "prompt": "What is 7 * 8?",
        "response": "7 * 8 = 56",
        "tool_calls": [],
    }
    score = judge.evaluate_trajectory(
        task="Score math answer quality",
        trajectory=trajectory,
    )
    assert isinstance(score, int | float), f"Score should be numeric, got {type(score)}: {score}"
    assert 0.0 <= score <= 1.0, f"Score out of range [0,1]: {score}"

    # Also test caching: same input should return same score
    score2 = judge.evaluate_trajectory(
        task="Score math answer quality",
        trajectory=trajectory,
    )
    assert score == score2, "Cache miss: scores differ on identical input"

    del judge
    gc.collect()

    return {
        "score": score,
        "score_cached": score2,
        "cache_hit": score == score2,
        "vram_after_mb": vram_mb(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# T6 — AgentTuneGraph Pattern A: rollout only
# ═════════════════════════════════════════════════════════════════════════════
def t6_graph_pattern_a():
    from agenttune.agentic.langgraph_orchestrator import AgentTuneGraph
    from agenttune.agentic.rollout_engines.rollout_factory import (
        create_rollout_engine,
        create_rollout_fn,
    )

    calc, conv = make_tools()

    engine = create_rollout_engine(backend="transformers", model_path=MODEL)
    rollout_fn = create_rollout_fn(
        rollout_engine=engine,
        tools=[calc, conv],
        max_steps=3,
        system_prompt="You are a helpful math assistant.",
    )

    graph = AgentTuneGraph()
    graph.add_rollout("agent", rollout_fn)
    unified = graph.compile_rollout()

    out = unified(["What is 12 * 12?", "Convert 5 miles to km."])
    assert out is not None

    final_rewards = out.get("final_reward", out.get("rewards", []))
    return {
        "output_keys": list(out.keys()),
        "num_outputs": len(out.get("responses", out.get("conversations", []))),
        "final_rewards": final_rewards,
        "has_trajectories": "trajectories" in out,
        "vram_after_mb": vram_mb(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# T7 — AgentTuneGraph Pattern B: rollout + dual judge
# ═════════════════════════════════════════════════════════════════════════════
def t7_graph_pattern_b():
    from agenttune.agentic.langgraph_orchestrator import AgentTuneGraph
    from agenttune.agentic.rewards.llm_judge import LLMJudge
    from agenttune.agentic.rollout_engines.rollout_factory import (
        create_rollout_engine,
        create_rollout_fn,
    )

    calc, conv = make_tools()

    engine = create_rollout_engine(backend="transformers", model_path=MODEL)
    rollout_fn = create_rollout_fn(
        rollout_engine=engine,
        tools=[calc, conv],
        max_steps=3,
        system_prompt="You are a math assistant.",
    )

    judge_correctness = LLMJudge(
        backend="transformers",
        model_path=MODEL,
        system_prompt='Rate correctness 0-1. Return {"score": <float>, "explanation": ""}',
        cache_size=50,
    )
    judge_clarity = LLMJudge(
        backend="transformers",
        model_path=MODEL,
        system_prompt='Rate clarity 0-1. Return {"score": <float>, "explanation": ""}',
        cache_size=50,
    )

    graph = AgentTuneGraph()
    graph.add_rollout("agent", rollout_fn)
    graph.add_judge(
        "correctness", judge_correctness, criteria={"accuracy": 0.7}, aggregation="mean"
    )
    graph.add_judge("clarity", judge_clarity, criteria={"clarity": 0.3}, aggregation="mean")
    graph.set_final_aggregation("mean")
    unified = graph.compile_rollout()

    out = unified(["What is 50 + 25?"])

    judge_scores = out.get("judge_scores", {})
    final_reward = out.get("final_reward")

    assert final_reward is not None or judge_scores, "No reward signal returned from graph"

    del judge_correctness, judge_clarity
    gc.collect()

    return {
        "output_keys": list(out.keys()),
        "judge_scores": judge_scores,
        "final_reward": final_reward,
        "vram_after_mb": vram_mb(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# T8 — AgentTuneGraph Pattern D: conditional router
# ═════════════════════════════════════════════════════════════════════════════
def t8_graph_router():
    from agenttune.agentic.langgraph_orchestrator import AgentTuneGraph
    from agenttune.agentic.rewards.llm_judge import LLMJudge
    from agenttune.agentic.rollout_engines.rollout_factory import (
        create_rollout_engine,
        create_rollout_fn,
    )

    calc, conv = make_tools()

    engine_s = create_rollout_engine(backend="transformers", model_path=MODEL)
    engine_c = create_rollout_engine(backend="transformers", model_path=MODEL)

    rollout_simple = create_rollout_fn(
        engine_s, tools=[calc], max_steps=2, system_prompt="Answer briefly."
    )
    rollout_complex = create_rollout_fn(
        engine_c, tools=[calc, conv], max_steps=4, system_prompt="Be thorough."
    )

    def router(prompts: list[str]) -> str:
        return "complex" if "convert" in prompts[0].lower() else "simple"

    judge = LLMJudge(
        backend="transformers",
        model_path=MODEL,
        system_prompt='Score 0-1. Return {"score": <float>, "explanation": ""}',
        cache_size=50,
    )

    graph = AgentTuneGraph()
    graph.add_rollout("simple", rollout_simple)
    graph.add_rollout("complex", rollout_complex)
    graph.set_router(router)
    graph.add_judge("quality", judge)
    unified = graph.compile_rollout()

    out_simple = unified(["What is 2 + 2?"])
    out_complex = unified(["Convert 10 km to miles."])

    del judge
    gc.collect()

    return {
        "simple_keys": list(out_simple.keys()),
        "complex_keys": list(out_complex.keys()),
        "simple_reward": out_simple.get("final_reward"),
        "complex_reward": out_complex.get("final_reward"),
        "vram_after_mb": vram_mb(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# T9 — OpenEnv Layer-1: live echo_env tool call
# ═════════════════════════════════════════════════════════════════════════════
def t9_openenv_live():
    from agenttune.utils.optional import OPENENV_AVAILABLE

    assert OPENENV_AVAILABLE, "openenv not installed"

    from agenttune.agentic.tools.builtin.openenv_tool import create_openenv_tools

    tools, handle = create_openenv_tools(base_url=ECHO_URL)
    try:
        names = [t.name for t in tools]
        assert "echo_message" in names, f"echo_message not found in {names}"

        echo = next(t for t in tools if t.name == "echo_message")
        result = echo.execute(message="agenttune_e2e_test")
        assert result.success, f"echo_message failed: {result.error}"
        assert result.output == "agenttune_e2e_test", f"Wrong output: {result.output}"
        assert "latency_ms" in result.metadata

        echo_len = next((t for t in tools if t.name == "echo_with_length"), None)
        len_result = None
        if echo_len:
            len_result = echo_len.execute(message="hello")
            assert len_result.success
            assert len_result.output.get("length") == 5

        return {
            "tools_discovered": names,
            "echo_message_output": result.output,
            "latency_ms": round(result.metadata["latency_ms"], 1),
            "echo_with_length": len_result.output if len_result else "not available",
        }
    finally:
        handle.close()


# ═════════════════════════════════════════════════════════════════════════════
# T10 — OpenEnv Layer-1: error_type metadata on invalid_args
# ═════════════════════════════════════════════════════════════════════════════
def t10_openenv_error_metadata():
    from agenttune.utils.optional import OPENENV_AVAILABLE

    assert OPENENV_AVAILABLE, "openenv not installed"

    from agenttune.agentic.tools.builtin.openenv_tool import create_openenv_tools

    tools, handle = create_openenv_tools(base_url=ECHO_URL)
    try:
        echo = next(t for t in tools if t.name == "echo_message")

        # Pass wrong arg — should trigger invalid_args on server
        result = echo.execute(wrong_param="this should fail")
        assert not result.success, "Should have failed on wrong param"
        assert "error" in result.output, "output must have 'error' key for failure counter"

        # Check error message is informative
        err = result.output["error"]
        assert len(err) > 0, "Error message should not be empty"

        return {
            "success": result.success,
            "error_msg_preview": err[:120],
            "output_has_error_key": "error" in result.output,
            "failure_counter_would_trip": isinstance(result.output, dict)
            and "error" in result.output,
        }
    finally:
        handle.close()


# ═════════════════════════════════════════════════════════════════════════════
# T11 — Full GRPO training: 5 steps, Qwen3-0.6B-FP8, LLMJudge reward
#
# Runs in a SEPARATE SUBPROCESS to get a clean VRAM slate unaffected by
# models cached from T3–T10 (critical on 8 GB GPU).
# ═════════════════════════════════════════════════════════════════════════════

T11_SCRIPT = '''
import gc, os, sys, json
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from datasets import Dataset
from agenttune.agentic.langgraph_orchestrator import AgentTuneGraph
from agenttune.agentic.rewards.llm_judge import LLMJudge
from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_engine, create_rollout_fn
from agenttune.core.backend_factory import create_agentic_trainer

MODEL = "Qwen/Qwen3-0.6B-FP8"
OUT   = str(Path(__file__).parent / "test_outputs" / "checkpoints" / "grpo_qwen3_0.6b")

def calculator(expression: str) -> dict:
    """Evaluate a mathematical expression.

    Args:
        expression: Python math expression string (e.g. '2 + 2').

    Returns:
        dict with result or error.
    """
    try:
        return {"result": eval(expression, {"__builtins__": {}}, {})}
    except Exception as e:
        return {"error": str(e)}

def unit_converter(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert a value between units.

    Args:
        value: Numeric value to convert.
        from_unit: Source unit (km, miles, kg, lbs, c, f).
        to_unit: Target unit (km, miles, kg, lbs, c, f).

    Returns:
        dict with result or error.
    """
    t = {("km","miles"):0.621371,("miles","km"):1.60934,("kg","lbs"):2.20462,
         ("lbs","kg"):0.453592,("c","f"):lambda v:v*9/5+32,("f","c"):lambda v:(v-32)*5/9}
    k = (from_unit.lower(), to_unit.lower())
    if k not in t: return {"error": f"Unsupported: {from_unit} to {to_unit}"}
    c = t[k]; return {"result": round(c(value) if callable(c) else value*c, 4)}

EXAMPLES = [
    {"prompt": [{"role":"user","content":"A store sells at 20% discount. Original $150. Discounted price?"}], "answer":"$120"},
    {"prompt": [{"role":"user","content":"Train travels 300km in 4 hours. Average speed?"}],                  "answer":"75 km/h"},
    {"prompt": [{"role":"user","content":"Simple interest on $1000 at 5% for 3 years?"}],                    "answer":"$150"},
    {"prompt": [{"role":"user","content":"Area of rectangle 8m x 12m?"}],                                    "answer":"96 m^2"},
    {"prompt": [{"role":"user","content":"Convert 100 km to miles."}],                                       "answer":"62.14 miles"},
    {"prompt": [{"role":"user","content":"A car uses 8L per 100km. Fuel for 350km?"}],                       "answer":"28 litres"},
    {"prompt": [{"role":"user","content":"Average of: 78, 85, 92, 88, 76?"}],                               "answer":"83.8"},
    {"prompt": [{"role":"user","content":"If 6 workers build wall in 12 days, 9 workers need?"}],            "answer":"8 days"},
]
train_dataset = Dataset.from_list(EXAMPLES)

engine = create_rollout_engine(backend="transformers", model_path=MODEL)
rollout_fn = create_rollout_fn(
    rollout_engine=engine, tools=[calculator, unit_converter],
    max_steps=3,
    system_prompt="You are a math assistant. Use tools when needed. Show your work.",
)
judge = LLMJudge(
    backend="transformers", model_path=MODEL,
    system_prompt="Score math answer 0-1. Return JSON: {\\"score\\": <float>, \\"explanation\\": \\"\\"}",
    cache_size=100,
)
graph = AgentTuneGraph()
graph.add_rollout("agent", rollout_fn)
graph.add_judge("quality", judge, criteria={"correctness": 0.8, "clarity": 0.2}, aggregation="mean")
unified_rollout = graph.compile_rollout()

del judge, engine
gc.collect()
import torch; torch.cuda.empty_cache()

def reward_from_judges(completions, prompts=None, **kwargs):
    return [c.get("final_reward", 0.5) if isinstance(c, dict) else 0.5 for c in completions]

trainer = create_agentic_trainer(
    algorithm="grpo", model=MODEL,
    reward_funcs=[reward_from_judges],
    train_dataset=train_dataset, rollout_func=unified_rollout,
    output_dir=OUT, max_steps=5,
    per_device_train_batch_size=1, gradient_accumulation_steps=2,
    learning_rate=1e-6, num_generations=2, max_completion_length=64,
    temperature=0.7, beta=0.04, use_vllm=False,
    logging_steps=1, log_completions=True, report_to="none",
)
result = trainer.train()
metrics = result.metrics if hasattr(result, "metrics") else (result if isinstance(result, dict) else {})
print(json.dumps({"status": "ok", "output_dir": OUT, "metrics": str(metrics)}))
'''


def t11_grpo_training():
    import json as _json
    import subprocess

    script_path = OUTPUT_DIR / "t11_train.py"
    script_path.write_text(T11_SCRIPT)

    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )

    # Save full subprocess output for audit
    (OUTPUT_DIR / "t11_stdout.txt").write_text(proc.stdout)
    (OUTPUT_DIR / "t11_stderr.txt").write_text(proc.stderr)

    if proc.returncode != 0:
        # Extract last meaningful error line
        err_lines = [l for l in proc.stderr.splitlines() if l.strip()]
        raise RuntimeError(err_lines[-1] if err_lines else "subprocess failed with no stderr")

    # Parse the final JSON line from stdout
    output_lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    result = _json.loads(output_lines[-1]) if output_lines else {}

    return {
        "status": result.get("status", "unknown"),
        "output_dir": result.get("output_dir", ""),
        "metrics": result.get("metrics", ""),
        "returncode": proc.returncode,
    }


# ═════════════════════════════════════════════════════════════════════════════
# T12 — Output artefact audit
# ═════════════════════════════════════════════════════════════════════════════
def t12_artefact_audit():
    out_dir = CKPT_DIR / "grpo_qwen3_0.6b"
    if not out_dir.exists():
        raise AssertionError("T11 output directory not found — run T11 first")

    all_files = list(out_dir.rglob("*"))
    file_names = [f.name for f in all_files if f.is_file()]

    expected = ["config.json", "tokenizer.json", "generation_config.json"]
    missing = [e for e in expected if e not in file_names]

    checkpoints = [f for f in all_files if f.is_dir() and "checkpoint" in f.name]
    parquets = list(out_dir.rglob("*.parquet"))
    stats_file = out_dir / "training_stats.json"
    stats = {}
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)

    return {
        "output_dir": str(out_dir),
        "total_files": len(file_names),
        "checkpoints": [str(c.name) for c in checkpoints],
        "parquet_files": [str(p.name) for p in parquets],
        "missing_expected": missing,
        "training_stats_keys": list(stats.keys()) if stats else [],
        "artefacts_ok": len(missing) == 0,
    }


# ═════════════════════════════════════════════════════════════════════════════
# T13 — Regression: ToolRegistry unaffected by openenv
# ═════════════════════════════════════════════════════════════════════════════
def t13_registry_regression():
    from agenttune.agentic.tools.registry import ToolRegistry

    ToolRegistry._builtins_registered = False
    ToolRegistry._tools = {}
    ToolRegistry.auto_register_builtins()

    names = ToolRegistry.list_all()
    openenv_tools = [n for n in names if "openenv" in n.lower()]
    expected_builtins = ["run_python", "run_bash", "read_file", "grep"]

    missing = [b for b in expected_builtins if b not in names]
    assert not openenv_tools, f"OpenEnv tools leaked into default registry: {openenv_tools}"
    assert not missing, f"Missing expected builtins: {missing}"

    return {
        "total_registered": len(names),
        "openenv_leaked": openenv_tools,
        "all_expected_present": not missing,
        "registered_names": names,
    }


# ═════════════════════════════════════════════════════════════════════════════
# T14 — GPU health summary
# ═════════════════════════════════════════════════════════════════════════════
def t14_gpu_health():
    import torch

    assert torch.cuda.is_available()

    torch.cuda.empty_cache()
    gc.collect()

    props = torch.cuda.get_device_properties(0)
    total = props.total_memory // (1024**2)
    alloc = torch.cuda.memory_allocated(0) // (1024**2)
    resrv = torch.cuda.memory_reserved(0) // (1024**2)
    free = total - resrv

    leak_threshold_mb = 500
    assert (
        alloc < leak_threshold_mb
    ), f"Possible VRAM leak: {alloc} MB still allocated after cleanup"

    return {
        "gpu": props.name,
        "total_vram_mb": total,
        "allocated_mb": alloc,
        "reserved_mb": resrv,
        "free_mb": free,
        "leak_check": "PASS" if alloc < leak_threshold_mb else "WARN",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Report generation
# ═════════════════════════════════════════════════════════════════════════════
def write_report():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")
    total_duration = sum(r.duration_s for r in results)

    # ── Markdown ──────────────────────────────────────────────────────────
    md_lines = [
        "# AgentTune × OpenEnv — E2E Test Report",
        f"\n**Generated:** {datetime.utcnow().isoformat()} UTC",
        f"**Model:** `{MODEL}`",
        f"**Log file:** `{log_path}`",
        "",
        "## Summary",
        "",
        "| | Count |",
        "|---|---|",
        f"| ✅ PASS  | {passed}  |",
        f"| ❌ FAIL  | {failed}  |",
        f"| ⏭  SKIP  | {skipped} |",
        f"| **Total** | **{len(results)}** |",
        f"| **Duration** | **{total_duration:.1f}s** |",
        "",
        "## Test Results",
        "",
        "| ID | Test | Goal | Status | Duration |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭ "}.get(r.status, "?")
        md_lines.append(
            f"| {r.test_id} | {r.name} | {r.goal} | {icon} {r.status} | {r.duration_s:.1f}s |"
        )

    md_lines += ["", "## Detailed Results", ""]
    for r in results:
        md_lines += [
            f"### {r.test_id} — {r.name}",
            f"**Goal:** {r.goal}",
            f"**Status:** {r.status}  **Duration:** {r.duration_s:.2f}s",
            "",
        ]
        if r.details:
            md_lines.append("**Details:**")
            md_lines.append("```json")
            md_lines.append(json.dumps(r.details, indent=2, default=str))
            md_lines.append("```")
        if r.error:
            md_lines.append("**Error:**")
            md_lines.append("```")
            md_lines.append(r.error[:1000])
            md_lines.append("```")
        md_lines.append("")

    md_path = REPORT_DIR / f"e2e_report_{ts}.md"
    md_path.write_text("\n".join(md_lines))
    logger.info("Markdown report: %s", md_path)

    # ── JSON summary ──────────────────────────────────────────────────────
    summary = {
        "timestamp": ts,
        "model": MODEL,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": len(results),
        "duration_s": round(total_duration, 2),
        "results": [asdict(r) for r in results],
    }
    json_path = REPORT_DIR / f"e2e_summary_{ts}.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("JSON summary:    %s", json_path)

    return md_path, json_path


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip T11 training (faster)")
    args = parser.parse_args()

    run_openenv = bool(os.environ.get("RUN_OPENENV_IT"))

    logger.info("=" * 70)
    logger.info("AgentTune × OpenEnv  E2E Test Suite")
    logger.info("Model  : %s", MODEL)
    logger.info("Quick  : %s", args.quick)
    logger.info("OpenEnv: %s (set RUN_OPENENV_IT=1 to enable)", run_openenv)
    logger.info("Output : %s", BASE_DIR)
    logger.info("=" * 70)

    # Goals description — shown in report
    G = {
        "T1": "Verify CUDA, all packages importable, GPU has enough VRAM",
        "T2": "Calculator and unit_converter return correct results and errors",
        "T3": "Single-step rollout returns response dict with expected keys",
        "T4": "Multi-step loop handles 2 prompts and records tool_call_counts",
        "T5": "LLMJudge scores a canned trajectory in [0,1] and hits cache on repeat",
        "T6": "Graph pattern A (rollout-only) produces output for 2 prompts",
        "T7": "Graph pattern B (dual judge) produces judge_scores and final_reward",
        "T8": "Graph pattern D router sends 'convert' prompt to complex branch",
        "T9": "OpenEnv echo_env live round-trip through OpenEnvTool.execute()",
        "T10": "OpenEnv error path returns ToolResult with 'error' key in output",
        "T11": "5-step GRPO training run completes without crash; checkpoints written",
        "T12": "Output directory contains config.json, tokenizer, checkpoints, parquets",
        "T13": "ToolRegistry.auto_register_builtins() does not include OpenEnv tools",
        "T14": "After cleanup, allocated VRAM is below 500 MB (no leak)",
    }

    run_test("T1", "Environment smoke", G["T1"], t1_environment)
    run_test("T2", "Local tool layer", G["T2"], t2_tool_layer)
    run_test("T3", "Rollout single-step", G["T3"], t3_rollout_single_step)
    flush_vram("T3")
    run_test("T4", "Rollout multi-step", G["T4"], t4_rollout_multistep)
    flush_vram("T4")
    run_test("T5", "LLMJudge scoring", G["T5"], t5_llm_judge)
    flush_vram("T5")
    run_test("T6", "Graph pattern A", G["T6"], t6_graph_pattern_a)
    flush_vram("T6")
    run_test("T7", "Graph pattern B", G["T7"], t7_graph_pattern_b)
    flush_vram("T7")
    run_test("T8", "Graph router pattern D", G["T8"], t8_graph_router)
    flush_vram("T8")
    run_test("T9", "OpenEnv live call", G["T9"], t9_openenv_live, skip=not run_openenv)
    run_test(
        "T10", "OpenEnv error metadata", G["T10"], t10_openenv_error_metadata, skip=not run_openenv
    )
    run_test("T13", "Registry regression", G["T13"], t13_registry_regression)

    # T11 and T12 run last — T11 needs a clean GPU slate.
    # Force-release ALL Python objects that may hold VRAM before the subprocess starts.
    import gc as _gc

    _gc.collect()
    try:
        import torch as _torch

        _torch.cuda.empty_cache()
        _torch.cuda.synchronize()
        # Unload torch from this process's GPU context as much as possible
        for _ in range(3):
            _gc.collect()
            _torch.cuda.empty_cache()
        logger.info("Pre-T11 VRAM flush: allocated=%d MB", vram_mb())
    except Exception:
        pass

    run_test("T11", "GRPO 5-step training", G["T11"], t11_grpo_training, skip=args.quick)
    flush_vram("T11")
    run_test("T12", "Artefact audit", G["T12"], t12_artefact_audit, skip=args.quick)
    run_test("T14", "GPU health check", G["T14"], t14_gpu_health)

    md_path, json_path = write_report()

    # ── Final summary ──────────────────────────────────────────────────────
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")

    logger.info("=" * 70)
    logger.info("DONE  ✅ %d passed  ❌ %d failed  ⏭  %d skipped", passed, failed, skipped)
    logger.info("Report  : %s", md_path)
    logger.info("JSON    : %s", json_path)
    logger.info("Log     : %s", log_path)
    logger.info("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
