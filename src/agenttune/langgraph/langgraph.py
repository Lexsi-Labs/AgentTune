"""
agenttune.agentic.langgraph_orchestrator
=========================================
LangGraph-based orchestrator that:

  1. Composes multiple ``create_rollout_fn`` outputs as graph nodes.
  2. Composes multiple ``LLMJudge`` instances as graph nodes.
  3. Auto-wires outputs between nodes based on the *parameter names* of
     each rollout function and judge's ``evaluate_trajectory`` / reward
     callable — no manual plumbing needed.
  4. Compiles to drop-in callables for each consumer:
       - ``compile_rollout()``     → ``unified_rollout_fn`` for ``GRPOTrainer(rollout_func=...)``.
         When judges are registered its output carries **per-completion** judge
         scores under ``judge_rewards`` (a length-N vector, not just the batch
         aggregate) so a reward_func can read one score per completion.
       - ``compile_grpo_reward()`` → ``reward_func(completions, **kwargs) -> list[float]``
         for ``GRPOTrainer(reward_funcs=[...])``; reads the rollout's ``judge_rewards``.
       - ``compile_reward()``      → ``reward_fn(trajectory) -> float`` for
         ``create_rollout_fn(reward_fn=...)`` (a single trajectory, not a batch).

────────────────────────────────────────────────────────────────────────
Quick-start
────────────────────────────────────────────────────────────────────────

    from agenttune.agentic.langgraph_orchestrator import AgentTuneGraph
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
    from agenttune.agentic.rewards.llm_judge import LLMJudge

    engine1 = VLLMRolloutEngine("Qwen/Qwen3-1.7B", gpu_memory_utilization=0.3)
    engine2 = VLLMRolloutEngine("Qwen/Qwen3-0.6B", gpu_memory_utilization=0.2)

    rollout_a = create_rollout_fn(rollout_engine=engine1, tools=[...], max_steps=5)
    rollout_b = create_rollout_fn(rollout_engine=engine2, tools=[...], max_steps=3)

    judge_a = LLMJudge(backend="vllm",         model_path="Qwen/Qwen3-0.6B")
    judge_b = LLMJudge(backend="transformers",  model_path="Qwen/Qwen3-0.6B",
                        system_prompt="You are a strict ...")

    graph = AgentTuneGraph()
    graph.add_rollout("tool_agent",   rollout_a)
    graph.add_rollout("code_agent",   rollout_b)
    graph.add_judge  ("quality",      judge_a)
    graph.add_judge  ("strict",       judge_b,   aggregation="min")

    # Optional: route prompts to ONE rollout instead of running all
    graph.set_router(lambda prompts: "code_agent" if "```" in prompts[0] else "tool_agent")

    unified_rollout_fn = graph.compile_rollout()      # → GRPOTrainer(rollout_func=...)
    grpo_reward_func   = graph.compile_grpo_reward()  # → GRPOTrainer(reward_funcs=[...])
    unified_reward_fn  = graph.compile_reward()       # → create_rollout_fn(reward_fn=...)

────────────────────────────────────────────────────────────────────────
Auto-wiring rules
────────────────────────────────────────────────────────────────────────

Rollout nodes
~~~~~~~~~~~~~
Each rollout node receives state["shared"] — a shared dict that grows
as nodes complete.  Before calling a rollout_fn the orchestrator
inspects its signature and injects any matching keys from shared:

  - ``prompts``      → state["prompts"]          (always injected)
  - ``trajectories`` → previous node's output trajectories
  - ``responses``    → previous node's output responses
  - ``rewards``      → previous node's rewards list
  - any other key    → looked up in state["shared"]

Judge nodes
~~~~~~~~~~~
Each judge node's ``evaluate_trajectory`` is called with:
  - ``task``       → trajectory.task
  - ``trajectory`` → the Trajectory object
  - ``criteria``   → judge's registered criteria (default {})

The judge's reward is stored back into state["shared"]["rewards"]
and also into state["judge_scores"][judge_name].
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from statistics import mean
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

# ─────────────────────────────────────────────────────────────────────────────
# Graph state
# ─────────────────────────────────────────────────────────────────────────────


class _GraphState(TypedDict):
    # Inputs
    prompts: list[str]
    gen_kwargs: dict[str, Any]

    # Accumulated across nodes — auto-wiring reads from here
    shared: dict[str, Any]

    # Per-node rollout outputs (node_name → raw dict from rollout_fn)
    rollout_outputs: dict[str, dict[str, Any]]

    # Per-judge scores (judge_name → float)
    judge_scores: dict[str, float]

    # Final aggregated outputs
    final_rollout: dict[str, Any]
    final_reward: float


# ─────────────────────────────────────────────────────────────────────────────
# Signature-based kwargs builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_kwargs_from_sig(fn: Callable, available: dict[str, Any]) -> dict[str, Any]:
    """
    Inspect ``fn``'s parameter names and pull matching values from
    ``available``.  Parameters with no match are left for the caller to
    supply (or use defaults).  *args/**kwargs params are ignored.
    """
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
    except (ValueError, TypeError):
        return {}

    kwargs: dict[str, Any] = {}
    for name, param in params.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in available:
            kwargs[name] = available[name]
    return kwargs


# ─────────────────────────────────────────────────────────────────────────────
# Output merge helper
# ─────────────────────────────────────────────────────────────────────────────


def _merge_dicts(base: dict, update: dict) -> dict:
    """
    Merge ``update`` into ``base``.
    List fields are concatenated; scalar fields are overwritten.
    """
    merged = dict(base)
    for k, v in update.items():
        if k in merged and isinstance(v, list) and isinstance(merged[k], list):
            merged[k] = merged[k] + v
        else:
            merged[k] = v
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Judge aggregation strategies
# ─────────────────────────────────────────────────────────────────────────────

_AGGS: dict[str, Callable[[list[float]], float]] = {
    "mean": mean,
    "min": min,
    "max": max,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────


class AgentTuneGraph:
    """
    Builds a single LangGraph StateGraph that orchestrates multiple
    rollout functions and LLM judges.

    Nodes are added in registration order and wired:

        rollout_1 → rollout_2 → ... → judge_1 → judge_2 → _aggregate → END

    With a router set:

        _route → (conditional) → selected_rollout → judge_1 → ... → END
    """

    def __init__(self):
        # Ordered registrations
        self._rollout_nodes: list[tuple] = []  # (name, fn)
        self._judge_nodes: list[tuple] = []  # (name, judge, criteria, agg_fn)

        self._router: Callable[[list[str]], str] | None = None
        self._final_agg: Callable[[list[float]], float] = mean

    # ── Registration ─────────────────────────────────────────────────────────

    def add_rollout(
        self,
        name: str,
        rollout_fn: Callable,
    ) -> AgentTuneGraph:
        """
        Register a rollout_fn (output of ``create_rollout_fn()``) as a node.

        The fn is called as:
            rollout_fn(prompts, **auto_wired_kwargs_from_previous_outputs)

        Any parameter in its signature that matches a key already in the
        shared state is injected automatically.
        """
        if any(n == name for n, _ in self._rollout_nodes):
            warnings.warn(f"Rollout node '{name}' already registered — overwriting.")
            self._rollout_nodes = [(n, f) for n, f in self._rollout_nodes if n != name]
        self._rollout_nodes.append((name, rollout_fn))
        return self

    def add_judge(
        self,
        name: str,
        judge: Any,  # LLMJudge instance
        criteria: dict[str, float] | None = None,
        aggregation: Literal["mean", "min", "max"] | Callable = "mean",
    ) -> AgentTuneGraph:
        """
        Register an LLMJudge as a reward node.

        The judge's ``evaluate_trajectory`` is called for every trajectory
        in state["shared"]["trajectories"].  Scores are aggregated with
        ``aggregation`` across trajectories and stored per-judge.

        ``aggregation`` applies *per-judge* (across its trajectory scores).
        The final reward across all judges uses ``set_final_aggregation()``.
        """
        if any(n == name for n, *_ in self._judge_nodes):
            warnings.warn(f"Judge node '{name}' already registered — overwriting.")
            self._judge_nodes = [t for t in self._judge_nodes if t[0] != name]
        agg_fn = aggregation if callable(aggregation) else _AGGS[aggregation]
        self._judge_nodes.append((name, judge, criteria or {}, agg_fn))
        return self

    def set_router(
        self,
        router_fn: Callable[[list[str]], str],
    ) -> AgentTuneGraph:
        """
        Optional.  ``router_fn(prompts) -> node_name`` selects ONE rollout
        node to run instead of all nodes sequentially.
        """
        self._router = router_fn
        return self

    def set_final_aggregation(
        self,
        strategy: Literal["mean", "min", "max"] | Callable = "mean",
    ) -> AgentTuneGraph:
        """How to combine scores across multiple judges into a single float."""
        self._final_agg = strategy if callable(strategy) else _AGGS[strategy]
        return self

    # ── Internal graph build ──────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        if not self._rollout_nodes and not self._judge_nodes:
            raise ValueError("Register at least one rollout or judge node first.")

        g = StateGraph(_GraphState)

        rollout_names = [n for n, _ in self._rollout_nodes]
        judge_names = [n for n, *_ in self._judge_nodes]

        # ── Rollout nodes ─────────────────────────────────────────────────────
        for node_name, fn in self._rollout_nodes:
            _fn = fn
            _name = node_name

            def _make_rollout_node(rollout_fn, name):
                def _node(state: _GraphState) -> _GraphState:
                    # Build kwargs from shared state + gen_kwargs, respecting fn signature
                    available = {
                        "prompts": state["prompts"],
                        **state["gen_kwargs"],
                        **state["shared"],
                    }
                    extra_kwargs = _build_kwargs_from_sig(rollout_fn, available)

                    # prompts is always positional-first; remove from kwargs if present
                    extra_kwargs.pop("prompts", None)

                    output: dict[str, Any] = rollout_fn(state["prompts"], **extra_kwargs)

                    # Store full output keyed by node name
                    state["rollout_outputs"][name] = output

                    # Merge into shared so downstream nodes can pick up outputs
                    # Key aliasing: expose output under canonical names
                    state["shared"] = _merge_dicts(
                        state["shared"],
                        {
                            "trajectories": output.get("trajectories", []),
                            "responses": output.get("responses", []),
                            "rewards": output.get("rewards", []),
                            "conversations": output.get("conversations", []),
                            "queries": output.get("queries", state["prompts"]),
                            "logprobs": output.get("logprobs", []),
                            "prompt_ids": output.get("prompt_ids", []),
                            "completion_ids": output.get("completion_ids", []),
                            "tool_call_counts": output.get("tool_call_counts", []),
                            f"{name}_output": output,  # namespaced copy
                        },
                    )
                    return state

                return _node

            g.add_node(node_name, _make_rollout_node(_fn, _name))

        # ── Judge nodes ───────────────────────────────────────────────────────
        for node_name, judge, criteria, agg_fn in self._judge_nodes:
            _judge = judge
            _name = node_name
            _criteria = criteria
            _agg = agg_fn

            def _make_judge_node(j, name, crit, agg):
                def _node(state: _GraphState) -> _GraphState:
                    trajectories = state["shared"].get("trajectories", [])

                    if not trajectories:
                        # No trajectories yet — score 0.5 as neutral fallback
                        warnings.warn(
                            f"Judge '{name}': no trajectories in shared state. "
                            "Ensure at least one rollout node runs before this judge.",
                            UserWarning,
                        )
                        state["judge_scores"][name] = 0.5
                        return state

                    # Evaluate each trajectory, then aggregate
                    scores: list[float] = []
                    for traj in trajectories:
                        task = getattr(traj, "task", "")
                        try:
                            s = j.evaluate_trajectory(
                                task=task,
                                trajectory=traj,
                                criteria=crit,
                            )
                        except Exception as e:
                            warnings.warn(f"Judge '{name}' error on trajectory: {e}")
                            s = 0.5
                        scores.append(s)

                    judge_score = agg(scores) if scores else 0.5
                    state["judge_scores"][name] = judge_score

                    # Write back into shared so downstream nodes/judges can see it
                    state["shared"][f"{name}_score"] = judge_score
                    # Keep the *per-trajectory* scores (not just the aggregate) so a
                    # per-completion reward vector can be built at aggregation time.
                    # GRPOTrainer's reward_funcs need one score per completion; the
                    # aggregate scalar alone can't supply that — which is exactly why
                    # a reward reading a single ``final_reward`` off each (string)
                    # completion silently fell back to its default.
                    state["shared"].setdefault("_judge_score_matrix", {})[name] = scores
                    return state

                return _node

            g.add_node(node_name, _make_judge_node(_judge, _name, _criteria, _agg))

        # ── Final aggregation node ────────────────────────────────────────────
        _final_agg = self._final_agg

        def _aggregate(state: _GraphState) -> _GraphState:
            scores = list(state["judge_scores"].values())
            state["final_reward"] = _final_agg(scores) if scores else 0.5

            # Per-completion reward vector: aggregate each completion's scores
            # across all judges (element-wise), so a GRPOTrainer reward_func gets
            # exactly one value per completion. This is the key the rollout output
            # was missing — ``judge_scores``/``final_reward`` are batch-level, but
            # training needs a length-N vector aligned to the N completions.
            matrix = state["shared"].get("_judge_score_matrix", {})
            judge_rewards: list[float] = []
            if matrix:
                n = max((len(v) for v in matrix.values()), default=0)
                for i in range(n):
                    per = [v[i] for v in matrix.values() if i < len(v)]
                    judge_rewards.append(_final_agg(per) if per else 0.5)

            # Build final_rollout: last rollout node output + judge scores
            if state["rollout_outputs"]:
                last_name = list(state["rollout_outputs"].keys())[-1]
                last_output = state["rollout_outputs"][last_name]
                state["final_rollout"] = _merge_dicts(
                    last_output,
                    {
                        "judge_scores": state["judge_scores"],
                        "final_reward": state["final_reward"],
                        "judge_rewards": judge_rewards,  # per-completion, length N
                    },
                )
            else:
                state["final_rollout"] = {
                    "judge_scores": state["judge_scores"],
                    "final_reward": state["final_reward"],
                    "judge_rewards": judge_rewards,
                }
            return state

        g.add_node("_aggregate", _aggregate)

        # ── Wire edges ────────────────────────────────────────────────────────
        all_nodes_ordered = rollout_names + judge_names

        if self._router and rollout_names:
            # Router → conditional dispatch to ONE rollout → judges → aggregate
            _router_fn = self._router

            def _route_node(state: _GraphState) -> _GraphState:
                chosen = _router_fn(state["prompts"])
                state["shared"]["_routed_to"] = chosen
                return state

            g.add_node("_route", _route_node)
            g.set_entry_point("_route")
            g.add_conditional_edges(
                "_route",
                lambda s: s["shared"].get("_routed_to", rollout_names[0]),
                {n: n for n in rollout_names},
            )
            # After the chosen rollout, run all judges sequentially
            for rn in rollout_names:
                if judge_names:
                    g.add_edge(rn, judge_names[0])
                else:
                    g.add_edge(rn, "_aggregate")
        else:
            # Linear: all rollouts → all judges → aggregate
            if all_nodes_ordered:
                g.set_entry_point(all_nodes_ordered[0])
                for i in range(len(all_nodes_ordered) - 1):
                    g.add_edge(all_nodes_ordered[i], all_nodes_ordered[i + 1])
                g.add_edge(all_nodes_ordered[-1], "_aggregate")
            else:
                g.set_entry_point("_aggregate")

        # Chain judges together (router mode)
        if self._router and judge_names:
            for i in range(len(judge_names) - 1):
                g.add_edge(judge_names[i], judge_names[i + 1])
            g.add_edge(judge_names[-1], "_aggregate")

        g.add_edge("_aggregate", END)
        return g.compile()

    # ── Compile: rollout_fn ───────────────────────────────────────────────────

    def compile_rollout(self) -> Callable:
        """
        Returns a unified rollout_fn compatible with:
          - ``GRPOTrainer(rollout_func=unified_rollout_fn)``
          - Standalone: ``unified_rollout_fn(prompts, temperature=0.7, ...)``

        Output dict includes all keys GRPOTrainer expects plus judge scores
        if any judges are registered.
        """
        compiled = self._build_graph()

        def unified_rollout_fn(prompts: list[str], *args, **gen_kwargs) -> dict[str, Any]:
            # Strip GRPOTrainer instance if passed as positional arg
            init: _GraphState = {
                "prompts": prompts,
                "gen_kwargs": gen_kwargs,
                "shared": {},
                "rollout_outputs": {},
                "judge_scores": {},
                "final_rollout": {},
                "final_reward": 0.5,
            }
            final = compiled.invoke(init)
            return final["final_rollout"]

        unified_rollout_fn.__doc__ = (
            f"AgentTuneGraph rollout: rollouts={[n for n,_ in self._rollout_nodes]}, "
            f"judges={[n for n,*_ in self._judge_nodes]}"
        )
        return unified_rollout_fn

    # ── Compile: reward_fn ────────────────────────────────────────────────────

    def compile_reward(self) -> Callable:
        """
        Returns a ``reward_fn(trajectory) -> float`` for use with:
            ``create_rollout_fn(..., reward_fn=graph.compile_reward())``

        Runs all registered judges against the single trajectory and
        aggregates using ``set_final_aggregation()`` strategy.
        """
        if not self._judge_nodes:
            warnings.warn(
                "No judges registered — compile_reward() returns a constant 0.5 fn.",
                UserWarning,
            )
            return lambda t: 0.5

        judges_snapshot = list(self._judge_nodes)  # (name, judge, criteria, agg_fn)
        _final_agg = self._final_agg

        def unified_reward_fn(trajectory) -> float:
            task = getattr(trajectory, "task", "")
            judge_scores: list[float] = []

            for name, judge, criteria, _agg_fn in judges_snapshot:
                try:
                    score = judge.evaluate_trajectory(
                        task=task,
                        trajectory=trajectory,
                        criteria=criteria,
                    )
                except Exception as e:
                    warnings.warn(f"Judge '{name}' error: {e}")
                    score = 0.5
                judge_scores.append(score)

            return _final_agg(judge_scores) if judge_scores else 0.5

        unified_reward_fn.__doc__ = (
            f"AgentTuneGraph reward_fn: judges={[n for n,*_ in judges_snapshot]}"
        )
        return unified_reward_fn

    # ── Compile: GRPOTrainer-shaped reward_func ───────────────────────────────

    def compile_grpo_reward(self) -> Callable:
        """
        Returns a ``reward_func(completions, **kwargs) -> list[float]`` in the exact
        shape ``GRPOTrainer(reward_funcs=[...])`` expects — one score per completion.

        Pair it with ``compile_rollout()``: the rollout runs the judges once and
        exposes their **per-completion** scores as ``judge_rewards`` in its output,
        which GRPOTrainer forwards to the reward_func. If that key is absent (e.g. a
        rollout without judges), it falls back to running ``compile_reward()`` over
        the ``trajectories`` — so it is correct either way, just not free.

            graph = AgentTuneGraph()
            graph.add_rollout("agent", rollout_fn)
            graph.add_judge("quality", judge)
            trainer = create_agentic_trainer(
                "grpo", model=..., train_dataset=ds,
                rollout_func=graph.compile_rollout(),
                reward_funcs=[graph.compile_grpo_reward()])
        """
        fallback_reward_fn = self.compile_reward() if self._judge_nodes else None

        def grpo_reward_func(
            completions=None, judge_rewards=None, trajectories=None, **kwargs
        ) -> list[float]:
            n = (
                len(completions)
                if completions is not None
                else (len(judge_rewards or []) or len(trajectories or []))
            )
            # Fast path: per-completion judge scores computed during the rollout.
            if judge_rewards is not None and len(judge_rewards) == n:
                return [float(x) for x in judge_rewards]
            # Fallback: re-run the judges over the trajectories (judges run twice).
            if fallback_reward_fn is not None and trajectories:
                return [float(fallback_reward_fn(t)) for t in trajectories]
            return [0.5] * n

        judge_names = [nm for nm, *_ in self._judge_nodes]
        grpo_reward_func.__name__ = "agenttune_graph_reward"
        grpo_reward_func.__doc__ = f"AgentTuneGraph GRPO reward_func: judges={judge_names}"
        return grpo_reward_func

    def __repr__(self) -> str:
        return (
            f"AgentTuneGraph("
            f"rollouts={[n for n,_ in self._rollout_nodes]}, "
            f"judges={[n for n,*_ in self._judge_nodes]}, "
            f"router={'set' if self._router else 'none'})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: grpo_rollout_func wrapper
# ─────────────────────────────────────────────────────────────────────────────


def make_grpo_rollout_func(unified_rollout_fn: Callable) -> Callable:
    """
    Wraps a compiled ``unified_rollout_fn`` into the exact signature that
    GRPOTrainer expects:

        def grpo_rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict

    The trainer's tokenizer, temperature, and max_completion_length are
    extracted and forwarded automatically via the auto-wiring mechanism.

    Usage:
        graph = AgentTuneGraph()
        ...
        grpo_fn = make_grpo_rollout_func(graph.compile_rollout())
        trainer = GRPOTrainer(..., rollout_func=grpo_fn)
    """

    def grpo_rollout_func(prompts: list[str], trainer) -> dict[str, Any]:
        gen_kwargs: dict[str, Any] = {}

        # Pull standard attrs from trainer if present
        for attr in ("temperature", "max_completion_length", "processing_class"):
            val = getattr(trainer, attr, None)
            if val is not None:
                gen_kwargs[attr] = val

        # Alias processing_class → tokenizer so rollout fns that accept
        # `tokenizer=` get it injected automatically
        if "processing_class" in gen_kwargs and "tokenizer" not in gen_kwargs:
            gen_kwargs["tokenizer"] = gen_kwargs["processing_class"]

        return unified_rollout_fn(prompts, **gen_kwargs)

    return grpo_rollout_func
