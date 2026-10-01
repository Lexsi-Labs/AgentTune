"""
YAML-driven Trainer Configuration Bridge for AgentTune Decide.

Allows every capability in README.md to be driven step-by-step from
a YAML config or from the decide YAML pipeline, mirroring the DECIDE_PLAN
build phases:

    Phase 1  — load config + choose algorithm
    Phase 2  — build rollout engine  (transformers / vllm / api)
    Phase 3  — attach tools
    Phase 4  — attach reward functions / LLM judge
    Phase 5  — (optional) build multi-agent AgentTuneGraph
    Phase 6  — apply PEFT / LoRA
    Phase 7  — create trainer and return

Usage — from Python:
    from agenttune.decide.trainer_config_bridge import TrainerConfigBridge

    bridge = TrainerConfigBridge("trainer_config.yaml")
    trainer = bridge.build_trainer(train_dataset=my_dataset)
    results = trainer.train()

Usage — from the decide pipeline (tool_call stage):
    - id: launch_training
      type: tool_call
      tool: trainer_config
      args:
        config_path: ./trainer_config.yaml
        audit_path: "{pipeline.audit_path}"
        stage_id: income_agent
        algorithm: dpo

See trainer_config.example.yaml for full configuration reference.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from agenttune.core.backend_factory import create_agentic_trainer

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _resolve_env(value: str) -> str:
    """Replace ${VAR} references with environment variable values."""

    def replacer(match):
        var = match.group(1)
        return os.environ.get(var, "")

    return re.sub(r"\$\{([^}]+)\}", replacer, value)


def _resolve_envs(obj: Any) -> Any:
    """Recursively resolve env vars in a config dict."""
    if isinstance(obj, str):
        return _resolve_env(obj)
    if isinstance(obj, dict):
        return {k: _resolve_envs(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_envs(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# TrainerConfigBridge
# ---------------------------------------------------------------------------


class TrainerConfigBridge:
    """
    Load a YAML trainer config and build the full agentic training stack.

    The config mirrors every feature in README.md:
      - algorithm selection (grpo / ppo / dpo / rloo / bco)
      - rollout engine (transformers / vllm / api)
      - tools (builtin + custom)
      - reward functions (named strings or callables)
      - LLM judge (absolute / relative rubric)
      - multi-agent graph (AgentTuneGraph with router)
      - PEFT / LoRA config (dict-style, no peft import needed)
      - decide bridge (extract training data from audit logs)
    """

    def __init__(self, config_path: str) -> None:
        self.config_path = Path(config_path)
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        self.config: dict[str, Any] = _resolve_envs(raw)

    # ------------------------------------------------------------------
    # Phase 1: Algorithm selection
    # ------------------------------------------------------------------

    @property
    def algorithm(self) -> str:
        return self.config.get("training", {}).get("algorithm", "grpo").lower()

    # ------------------------------------------------------------------
    # Phase 2: Rollout engine
    # ------------------------------------------------------------------

    def build_rollout_engine(self, rollout_cfg: dict | None = None):
        """
        Instantiate a rollout engine from config.

        Supported backends (matches README section):
          - "transformers"  — HuggingFace local
          - "vllm"          — vLLM GPU-optimised
          - "api"           — LiteLLM (Groq, OpenAI, Ollama)

        Returns:
            RolloutEngine instance
        """
        from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_engine

        cfg = rollout_cfg or self.config.get("training", {}).get("rollout", {})
        backend = cfg.get("backend", "transformers")

        kwargs: dict[str, Any] = {}
        if backend == "transformers":
            kwargs["model_path"] = cfg.get(
                "model_path", self.config.get("training", {}).get("model", "Qwen/Qwen3-0.6B")
            )
        elif backend == "vllm":
            kwargs["model_path"] = cfg.get(
                "model_path", self.config.get("training", {}).get("model")
            )
            if "gpu_memory_utilization" in cfg:
                kwargs["gpu_memory_utilization"] = cfg["gpu_memory_utilization"]
        elif backend == "api":
            kwargs["api_model"] = cfg.get("api_model", "")
            kwargs["api_base_url"] = cfg.get("api_base_url", "")
            kwargs["api_key"] = cfg.get("api_key", "")
        else:
            raise ValueError(
                f"Unknown rollout backend '{backend}'. Choose: transformers, vllm, api"
            )

        return create_rollout_engine(backend=backend, **kwargs)

    def build_rollout_fn(self, engine=None, rollout_cfg: dict | None = None):
        """
        Build a standalone rollout function from engine + config.

        The returned callable is fully standalone — usable outside
        any training loop for debugging, data collection, or inference
        (matches README 'Testing the Rollout Manually' section).

        Returns:
            Callable: rollout_fn(prompts) → dict
        """
        from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn

        cfg = rollout_cfg or self.config.get("training", {}).get("rollout", {})
        if engine is None:
            engine = self.build_rollout_engine(cfg)

        tools = self.build_tools()
        return create_rollout_fn(
            rollout_engine=engine,
            tools=tools,
            max_steps=cfg.get("max_steps", 3),
            system_prompt=cfg.get("system_prompt", ""),
        )

    # ------------------------------------------------------------------
    # Phase 3: Tools
    # ------------------------------------------------------------------

    def build_tools(self) -> list[Callable]:
        """
        Build tool callables from config.

        Supports:
          - Named builtin tools (sql, web_search, file, code, github, slack…)
          - Custom callables registered via module.class path

        Returns:
            List of callable tools compatible with create_rollout_fn
        """
        tools_cfg = self.config.get("training", {}).get("tools", [])
        tools: list[Callable] = []

        for tool_cfg in tools_cfg:
            tool_type = tool_cfg.get("type", "builtin")

            if tool_type == "builtin_sql":
                from agenttune.agentic.tools.builtin.sql import SQLDatabaseTool

                t = SQLDatabaseTool(tool_cfg.get("connection", "sqlite:///./data.db"))
                tools.append(t.execute)

            elif tool_type == "builtin_web_search":
                from agenttune.agentic.tools.builtin.web_search_tool import WebSearchTool

                t = WebSearchTool()
                tools.append(t.execute)

            elif tool_type == "builtin_code":
                from agenttune.agentic.tools.builtin.code_tools import RunPythonTool

                t = RunPythonTool()
                tools.append(t.execute)

            elif tool_type == "builtin_file":
                from agenttune.agentic.tools.builtin.file_tools import ReadFileTool

                t = ReadFileTool()
                tools.append(t.execute)

            elif tool_type == "builtin_bash":
                from agenttune.agentic.tools.builtin.code_tools import RunBashTool

                t = RunBashTool()
                tools.append(t.execute)

            elif tool_type == "openenv":
                tools.extend(self._build_openenv_tools(tool_cfg))

            elif tool_type == "custom":
                # Dynamically import a class from module.class path
                module_path = tool_cfg.get("module", "")
                class_name = tool_cfg.get("class", "")
                if module_path and class_name:
                    import importlib

                    mod = importlib.import_module(module_path)
                    cls = getattr(mod, class_name)
                    init_kwargs = tool_cfg.get("init_kwargs", {})
                    instance = cls(**init_kwargs)
                    tools.append(instance)

        return tools

    def _build_openenv_tools(self, tool_cfg: dict[str, Any]):
        """
        Build OpenEnvTool instances from a YAML tool block of type 'openenv'.

        Expected YAML shape::

            tools:
              - type: openenv
                base_url: "http://localhost:8000"   # required
                name_prefix: "sandbox_"             # optional
                tool_filter: ["echo_message"]       # optional
                connect_timeout_s: 10               # optional
                message_timeout_s: 60               # optional

        The lifecycle handle is registered via atexit so the WebSocket session
        is closed cleanly when the Python interpreter exits.

        Raises:
            ValueError: If base_url is missing from the config block.
        """
        import atexit

        from agenttune.agentic.tools.builtin.openenv_tool import create_openenv_tools

        base_url = tool_cfg.get("base_url")
        if not base_url:
            raise ValueError(
                "openenv tool config requires 'base_url'. "
                "Example: base_url: 'http://localhost:8000'"
            )

        tools, handle = create_openenv_tools(
            base_url=base_url,
            connect_timeout_s=float(tool_cfg.get("connect_timeout_s", 10.0)),
            message_timeout_s=float(tool_cfg.get("message_timeout_s", 60.0)),
            tool_filter=tool_cfg.get("tool_filter"),
            name_prefix=tool_cfg.get("name_prefix"),
        )

        # Register cleanup so the handle is always closed, even on crash
        atexit.register(handle.close)

        # Stash handle on the bridge instance so callers can close explicitly
        if not hasattr(self, "_openenv_handles"):
            self._openenv_handles: list = []
        self._openenv_handles.append(handle)

        return tools

    def close_openenv_handles(self) -> None:
        """Close all OpenEnv lifecycle handles opened by build_tools().

        Call this explicitly after training completes to release WebSocket
        connections and background threads promptly.  The handles are also
        closed via atexit as a fallback.
        """
        for handle in getattr(self, "_openenv_handles", []):
            handle.close()
        self._openenv_handles = []

    # ------------------------------------------------------------------
    # Phase 4: Reward functions + LLM judge
    # ------------------------------------------------------------------

    def build_reward_funcs(self) -> list[Any]:
        """
        Build reward function list from config.

        Accepts:
          - Named strings ("correctness_reward", "structure_reward", etc.)
          - Module.function paths ("my_module.my_reward_fn")

        Returns:
            List of reward functions (strings or callables)
        """
        reward_cfg = self.config.get("training", {}).get("reward_funcs", [])
        if not reward_cfg:
            return []

        resolved: list[Any] = []
        for rf in reward_cfg:
            if isinstance(rf, str):
                if "." in rf:
                    # Try to import as module.function
                    parts = rf.rsplit(".", 1)
                    try:
                        import importlib

                        mod = importlib.import_module(parts[0])
                        resolved.append(getattr(mod, parts[1]))
                    except (ImportError, AttributeError):
                        resolved.append(rf)  # fall back to name string
                else:
                    resolved.append(rf)
            elif callable(rf):
                resolved.append(rf)

        return resolved

    def build_judge(self) -> Any | None:
        """
        Build an LLMJudge instance from config.

        Supports absolute rubric (single trajectory) and relative rubric
        (pairwise comparison batch) — matches README LLM-as-Judge section.

        Returns:
            LLMJudge instance or None if not configured
        """
        judge_cfg = self.config.get("training", {}).get("judge")
        if not judge_cfg:
            return None

        from agenttune.agentic.rewards.llm_judge import LLMJudge

        kwargs: dict[str, Any] = {}

        backend = judge_cfg.get("backend")
        if backend:
            kwargs["backend"] = backend

        model = judge_cfg.get("model") or judge_cfg.get("model_path")
        if model:
            key = "model_path" if backend in ("transformers", "vllm") else "model"
            kwargs[key] = model

        for field in ("base_url", "api_key", "system_prompt"):
            if field in judge_cfg:
                kwargs[field] = judge_cfg[field]

        return LLMJudge(**kwargs)

    # ------------------------------------------------------------------
    # Phase 5: Multi-agent AgentTuneGraph
    # ------------------------------------------------------------------

    def build_multi_agent_graph(self):
        """
        Build an AgentTuneGraph from config.

        Matches README LangGraph Orchestrator section — composes multiple
        rollout engines and LLM judges with auto-wiring and optional router.

        Returns:
            Tuple[unified_rollout_fn, unified_reward_fn]
        """
        from agenttune.agentic.langgraph_orchestrator import AgentTuneGraph
        from agenttune.agentic.rewards.llm_judge import LLMJudge

        ma_cfg = self.config.get("training", {}).get("multi_agent", {})
        graph = AgentTuneGraph()

        # Register rollout nodes
        for rollout_cfg in ma_cfg.get("rollouts", []):
            engine = self.build_rollout_engine(rollout_cfg)
            rollout_fn = self.build_rollout_fn(engine, rollout_cfg)
            graph.add_rollout(rollout_cfg["name"], rollout_fn)

        # Register judge nodes
        for judge_cfg in ma_cfg.get("judges", []):
            kwargs: dict[str, Any] = {}
            for field in ("model", "base_url", "api_key", "system_prompt", "backend", "model_path"):
                if field in judge_cfg:
                    kwargs[field] = judge_cfg[field]
            judge = LLMJudge(**kwargs)
            aggregation = judge_cfg.get("aggregation", "mean")
            graph.add_judge(judge_cfg["name"], judge, aggregation=aggregation)

        # Optional router
        router_expr = ma_cfg.get("router")
        if router_expr:
            graph.set_router(
                eval(f"lambda prompts: {router_expr}")
            )  # noqa: S307 — user-controlled config

        # Final aggregation strategy
        final_agg = ma_cfg.get("final_aggregation")
        if final_agg:
            graph.set_final_aggregation(final_agg)

        return graph.compile_rollout(), graph.compile_reward()

    # ------------------------------------------------------------------
    # Phase 6: PEFT / LoRA
    # ------------------------------------------------------------------

    def get_peft_config(self) -> dict[str, Any] | None:
        """
        Return PEFT config dict (no peft import needed — matches README).

        Returns:
            Dict with LoRA parameters or None if not configured
        """
        return self.config.get("training", {}).get("peft_config")

    # ------------------------------------------------------------------
    # Phase 7: Build full trainer
    # ------------------------------------------------------------------

    def build_trainer(
        self,
        train_dataset=None,
        tools: list | None = None,
        reward_funcs: list | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        """
        Build the full agentic trainer from YAML config.

        Steps:
          1. Resolve algorithm
          2. Build rollout engine
          3. Build tools (from config or override)
          4. Build reward functions (from config or override)
          5. (Optional) build multi-agent graph
          6. Apply PEFT config
          7. Call create_agentic_trainer

        Args:
            train_dataset: Training dataset (overrides decide_bridge extraction)
            tools: Override tool list (defaults to config tools)
            reward_funcs: Override reward functions (defaults to config list)
            extra_kwargs: Additional kwargs forwarded to create_agentic_trainer

        Returns:
            Trainer instance. Call .train() to start training.
        """
        training_cfg = self.config.get("training", {})
        hyperparams = training_cfg.get("hyperparams", {})

        # Extract from audit if no dataset provided
        if train_dataset is None:
            train_dataset = self._extract_dataset_from_audit()

        if train_dataset is None:
            raise ValueError(
                "No training dataset provided and no decide_bridge configured. "
                "Pass train_dataset= or add decide_bridge section to YAML."
            )

        # Check multi-agent mode
        ma_cfg = training_cfg.get("multi_agent", {})
        if ma_cfg.get("enabled"):
            rollout_fn, reward_fn = self.build_multi_agent_graph()
        else:
            rollout_fn = None

        # Collect kwargs
        kwargs: dict[str, Any] = {
            "algorithm": self.algorithm,
            "model": training_cfg.get("model", "Qwen/Qwen3-1.7B"),
            "train_dataset": train_dataset,
            "output_dir": training_cfg.get("output_dir", "./output"),
            **hyperparams,
            **(extra_kwargs or {}),
        }

        # Tools and reward_funcs
        kwargs["tools"] = tools if tools is not None else self.build_tools()
        kwargs["reward_funcs"] = (
            reward_funcs if reward_funcs is not None else self.build_reward_funcs()
        )

        # Multi-agent graph overrides rollout
        if rollout_fn is not None:
            from agenttune.agentic.langgraph_orchestrator import make_grpo_rollout_func

            kwargs["rollout_func"] = make_grpo_rollout_func(rollout_fn)

        # PEFT
        peft_cfg = self.get_peft_config()
        if peft_cfg:
            kwargs["peft_config"] = peft_cfg

        return create_agentic_trainer(**kwargs)

    # ------------------------------------------------------------------
    # Decide bridge: extract dataset from audit log
    # ------------------------------------------------------------------

    def _extract_dataset_from_audit(self):
        """
        Extract training dataset from a Decide audit log.

        Called automatically by build_trainer() when no dataset is passed.
        Reads the decide_bridge section of the config.

        Returns:
            Dataset suitable for the configured algorithm, or None
        """
        bridge_cfg = self.config.get("decide_bridge", {})
        if not bridge_cfg.get("enabled"):
            return None

        from agenttune.decide.training_bridge import DecideToTrainerBridge

        audit_path = bridge_cfg.get("audit_path", "./audit.jsonl")
        stage_id = bridge_cfg.get("stage_id", "output")

        bridge = DecideToTrainerBridge(audit_path)

        if self.algorithm == "dpo":
            pairs = bridge.extract_dpo_pairs(stage_id)
            if not pairs:
                return None
            from datasets import Dataset as HFDataset

            return HFDataset.from_list(pairs)

        elif self.algorithm == "bco":
            labels = bridge.extract_bco_labels(stage_id)
            if not labels:
                return None
            from datasets import Dataset as HFDataset

            return HFDataset.from_list(labels)

        else:  # grpo / ppo / rloo
            ds = bridge.extract_trajectories(stage_id)
            return ds if len(ds) > 0 else None

    # ------------------------------------------------------------------
    # Step-wise builder (matches DECIDE_PLAN build phases)
    # ------------------------------------------------------------------

    def step_build_rollout_engine(self):
        """Phase 2: Build and return rollout engine only."""
        return self.build_rollout_engine()

    def step_build_rollout_fn(self):
        """Phase 2+3: Build rollout fn (engine + tools attached)."""
        return self.build_rollout_fn()

    def step_build_tools(self):
        """Phase 3: Build and return tool list only."""
        return self.build_tools()

    def step_build_reward_funcs(self):
        """Phase 4: Build and return reward function list only."""
        return self.build_reward_funcs()

    def step_build_judge(self):
        """Phase 4: Build and return LLMJudge only."""
        return self.build_judge()

    def step_build_graph(self):
        """Phase 5: Build and return (unified_rollout_fn, unified_reward_fn)."""
        return self.build_multi_agent_graph()

    def step_get_peft_config(self):
        """Phase 6: Return PEFT config dict."""
        return self.get_peft_config()

    def step_build_trainer(self, train_dataset=None, **kwargs):
        """Phase 7: Build and return full trainer."""
        return self.build_trainer(train_dataset=train_dataset, extra_kwargs=kwargs or None)


# ---------------------------------------------------------------------------
# Standalone helper — build from YAML path
# ---------------------------------------------------------------------------


def build_trainer_from_yaml(
    config_path: str,
    train_dataset=None,
    **kwargs,
):
    """
    One-line factory: load YAML and return a ready trainer.

    Args:
        config_path: Path to trainer_config.yaml
        train_dataset: Optional dataset override
        **kwargs: Additional kwargs forwarded to create_agentic_trainer

    Returns:
        Trainer instance
    """
    bridge = TrainerConfigBridge(config_path)
    return bridge.build_trainer(train_dataset=train_dataset, extra_kwargs=kwargs or None)
