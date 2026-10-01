"""LangGraph-based pipeline execution engine."""

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import random
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from langgraph.graph import END, StateGraph

from agenttune.decide.audit import AuditWriter
from agenttune.decide.config import ConfigLoader
from agenttune.decide.destinations.router import DestinationRouter
from agenttune.decide.stage_executor import StageExecutor
from agenttune.decide.stages.base import flatten_stage_outputs
from agenttune.decide.stages.rules import SafeEvaluator
from agenttune.decide.state import PipelineState

logger = logging.getLogger(__name__)


class GraphRunner:
    """
    Compile YAML template stages into LangGraph StateGraph and execute.

    Attributes:
        config: Merged configuration dictionary
        graph: Compiled LangGraph Runnable
    """

    _engine_cache: dict[str, Any] = {}

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize GraphRunner with configuration.

        Args:
            config: Merged configuration dictionary
        """
        self.config = config
        self._shared_engine = None
        self.graph = self._build_langgraph()

    @staticmethod
    def from_template(template_id: str, config_path: str = "./config.yaml") -> "GraphRunner":
        """
        Factory method to create GraphRunner from template.

        Args:
            template_id: Template identifier (e.g., "bfsi/kyc_triage")
            config_path: Path to global config.yaml

        Returns:
            Configured GraphRunner instance
        """
        config = ConfigLoader.load(template_id, config_path)
        return GraphRunner(config)

    def _get_engine(self) -> Any:
        """
        Get or create cached RolloutEngine for this GraphRunner.

        Returns:
            Cached RolloutEngine instance
        """
        if self._shared_engine is not None:
            return self._shared_engine

        backend = self.config.get("backend", "transformers")
        model = self.config.get("default_model", "Qwen/Qwen2.5-0.5B-Instruct")
        cache_key = f"{backend}:{model}"

        # Don't load a rollout engine for API-based models (Claude, GPT, etc.)
        api_prefixes = (
            "gpt-",
            "claude-",
            "groq-",
            "grok-",
            "gemini-",
            "command",
            "j2-",
            "palm-",
            "text-davinci",
            "text-curie",
            "together",
            "replicate",
            "openrouter",
        )
        if any(model.lower().startswith(p) for p in api_prefixes):
            return None

        if cache_key not in GraphRunner._engine_cache:
            try:
                from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_engine

                GraphRunner._engine_cache[cache_key] = create_rollout_engine(
                    backend=backend,
                    model_path=model,
                )
            except Exception:
                return None

        self._shared_engine = GraphRunner._engine_cache[cache_key]
        return self._shared_engine

    def run_sync(self, input_text: str) -> PipelineState:
        """
        Execute the pipeline synchronously (blocks until completion).

        Args:
            input_text: Input text for pipeline

        Returns:
            Final pipeline state
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.run(input_text)).result()
        else:
            return asyncio.run(self.run(input_text))

    def _build_langgraph(self) -> Any:
        """
        Compile YAML stages into LangGraph StateGraph.

        Returns:
            Compiled LangGraph Runnable
        """
        graph = StateGraph(PipelineState)
        stages = self.config.get("stages", [])

        # Add nodes (one per stage)
        for stage in stages:
            node_fn = self._make_stage_node(stage)
            graph.add_node(stage["id"], node_fn)

        # Set entry point (first stage)
        if stages:
            graph.set_entry_point(stages[0]["id"])

        # Build lookup of stages that have top-level edges leaving them
        top_level_edges = self.config.get("edges", [])
        top_edge_sources = {e["from"] for e in top_level_edges if e.get("from")}

        # Add per-stage routing edges (on_result, default, rules, next)
        for stage in stages:
            self._add_edges(graph, stage)

        # Stages whose routing was already wired above by _add_edges (via on_result,
        # default, rule on_fail targets, or "next"). config["edges"] is auto-generated
        # from these same per-stage fields (see ConfigLoader._generate_edges), so for
        # these stages it is a flattened list of every possible destination. Re-adding
        # those as unconditional graph.add_edge() calls would make LangGraph fan out to
        # *all* branches of a conditional stage in the same superstep (instead of just
        # the one the router picked), which crashes with INVALID_CONCURRENT_GRAPH_UPDATE
        # the moment two branches try to write the same state. Only apply top-level
        # edges for stages with no per-stage routing at all (a template authored purely
        # via a top-level `edges:` list).
        already_routed = {
            stage["id"]
            for stage in stages
            if stage.get("on_result")
            or stage.get("default")
            or "next" in stage
            or any(
                (rule.get("on_failure") or rule.get("on_fail")) for rule in stage.get("rules", [])
            )
        }

        # Add top-level edges from the 'edges:' array (skip __end__ — handled by finish point)
        for edge in top_level_edges:
            from_id = edge.get("from")
            to_id = edge.get("to")
            if not from_id or not to_id or to_id == "__end__":
                continue
            if from_id in already_routed:
                continue
            if from_id == to_id:
                # Self-loop: use conditional edge so is_complete can break the cycle
                try:
                    graph.add_conditional_edges(
                        from_id,
                        lambda s, _fid=from_id: END if s.is_complete else _fid,
                        {END: END, from_id: from_id},
                    )
                except Exception:
                    pass
            else:
                try:
                    graph.add_edge(from_id, to_id)
                except Exception:
                    pass

        # For stages with no explicit routing and no top-level edge, add sequential flow
        for i, stage in enumerate(stages):
            has_routing = bool(
                stage.get("on_result")
                or stage.get("default")
                or stage.get("rules")
                or "next" in stage
                or stage["id"] in top_edge_sources
            )
            if not has_routing and stage["type"] != "output" and i + 1 < len(stages):
                try:
                    graph.add_edge(stage["id"], stages[i + 1]["id"])
                except Exception:
                    pass

        # Set finish points (output stages)
        for stage in stages:
            if stage["type"] == "output":
                graph.set_finish_point(stage["id"])

        return graph.compile()

    def _make_stage_node(self, stage: dict[str, Any]) -> Callable:
        """
        Create a LangGraph node function for a stage.

        Args:
            stage: Stage configuration

        Returns:
            Async node function
        """

        async def stage_node(state: PipelineState) -> PipelineState:
            try:
                # Get stage handler
                handler_class = StageExecutor.get(stage["type"])
                handler = handler_class(stage)
                # Pass global config to handler for access to default values
                handler.global_config = self.config
                # Inject shared engine to avoid creating new one per call
                handler._shared_engine = self._get_engine()

                # Execute stage with global config as fallback
                result = await handler.execute(state, stage_config=stage)

                # Update state outputs
                state.stage_outputs[stage["id"]] = result.get("output", {})

                # Increment iteration counter
                state.stage_iterations[stage["id"]] = state.stage_iterations.get(stage["id"], 0) + 1

                # Update step tracking
                state.step_count += 1
                state.step_history.append(stage["id"])

                # Store routing decision if present (for router/rules stages)
                if "goto" in result and result["goto"]:
                    state.next_stage = result["goto"]

                # Log to audit
                audit_writer = AuditWriter(
                    self.config.get("audit", {}).get("path", "./audit.jsonl")
                )
                audit_writer.log_stage(state, stage, result)

                # Check global step limit
                max_steps = self.config.get("max_total_steps", 50)
                if state.step_count >= max_steps:
                    state.error = f"max_total_steps ({max_steps}) reached"
                    state.is_complete = True

            except Exception as e:
                state.error = str(e)
                state.error_stage = stage["id"]
                state.is_complete = True

            return state

        return stage_node

    def _add_edges(self, graph: Any, stage: dict[str, Any]) -> None:
        """
        Add edges to LangGraph for stage transitions.

        Args:
            graph: LangGraph StateGraph instance
            stage: Stage configuration
        """
        stage_id = stage["id"]

        # Check for on_result (conditional routing)
        on_result = stage.get("on_result", [])
        default = stage.get("default")

        # Check for rules with on_fail/on_failure
        rules = stage.get("rules", [])
        rule_targets = set()
        for rule in rules:
            on_failure = rule.get("on_failure") or rule.get("on_fail")
            if on_failure:
                target = on_failure.get("goto") if isinstance(on_failure, dict) else on_failure
                if target:
                    rule_targets.add(target)

        if on_result or default or rule_targets:
            # Create possible targets for conditional edges
            targets = {}
            for condition_path in on_result:
                goto = condition_path.get("goto")
                if goto:
                    targets[goto] = goto

            # Add default target
            if default:
                targets[default] = default

            # Add rule targets
            for target in rule_targets:
                targets[target] = target

            # Add next target as fallback
            if "next" in stage:
                targets[stage["next"]] = stage["next"]

            # Always allow routing straight to END once is_complete is set (see router
            # below) — LangGraph looks up the router's return value in this map, so END
            # must be a registered target or a mid-run stop raises KeyError('__end__').
            targets[END] = END

            # Create router function that checks state.next_stage or evaluates conditions
            def make_router(stage_cfg):
                def router(state: PipelineState) -> str | None:
                    # A prior node already hit max_total_steps or a fatal error and
                    # marked the run complete — stop routing instead of looping
                    # (e.g. a rules stage's on_fail -> extract retry loop) until
                    # LangGraph's own recursion limit aborts the run.
                    if state.is_complete:
                        return END

                    # First check if next_stage was set by the stage handler
                    if state.next_stage:
                        next_target = state.next_stage
                        state.next_stage = None  # Clear for next use
                        return next_target

                    # Otherwise evaluate conditions
                    on_result = stage_cfg.get("on_result", [])
                    for condition_path in on_result:
                        condition_str = condition_path.get("condition")
                        goto = condition_path.get("goto")
                        if condition_str and goto:
                            try:
                                if self._eval_condition(condition_str, state):
                                    return goto
                            except Exception:
                                continue

                    # Fall back to default or next
                    default = stage_cfg.get("default")
                    if default:
                        return default
                    if "next" in stage_cfg:
                        return stage_cfg["next"]
                    return None

                return router

            if targets:
                graph.add_conditional_edges(
                    stage_id,
                    make_router(stage),
                    targets,
                )
        else:
            # Success path (next field)
            if "next" in stage:
                graph.add_edge(stage_id, stage["next"])

    def _eval_condition(self, condition: str, state: PipelineState) -> bool:
        """
        Evaluate a condition string against pipeline state.

        Args:
            condition: Condition expression
            state: Pipeline state

        Returns:
            Boolean result of condition evaluation
        """
        evaluator = SafeEvaluator()
        try:
            flattened = flatten_stage_outputs(state.stage_outputs)
            return evaluator.eval(condition, flattened)
        except Exception as e:
            raise ValueError(f"Failed to evaluate condition '{condition}': {str(e)}")  # noqa: B904

    async def run_episodes(
        self,
        inputs: list[str],
        n_episodes: int,
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> list[PipelineState]:
        """
        Run N episodes, processing batch_size pipelines concurrently.

        Args:
            inputs: Pool of input texts to sample from (cycled if fewer than n_episodes)
            n_episodes: Total number of episodes to run
            batch_size: Number of pipelines to run concurrently per batch
            shuffle: Randomise input order before cycling
            seed: Random seed for reproducible shuffling

        Returns:
            List of PipelineState, one per episode
        """
        if shuffle:
            rng = random.Random(seed)
            shuffled = list(inputs)
            rng.shuffle(shuffled)
            inputs = shuffled

        episode_inputs = [inputs[i % len(inputs)] for i in range(n_episodes)]
        all_states: list[PipelineState] = []

        for batch_start in range(0, n_episodes, batch_size):
            batch = episode_inputs[batch_start : batch_start + batch_size]
            batch_states = await asyncio.gather(*[self.run(inp) for inp in batch])
            all_states.extend(batch_states)

        return all_states

    def _validate_observation(self, input_text: str, schema: dict[str, Any]) -> None:
        """
        Validate input_text against observation_schema.

        Raises:
            ValueError: if the input does not match the declared schema type
        """
        schema_type = schema.get("type")
        if not schema_type:
            return

        if schema_type == "string":
            if not isinstance(input_text, str):
                raise ValueError(
                    f"observation_schema type 'string' requires str input, "
                    f"got {type(input_text).__name__}"
                )

        elif schema_type == "object":
            try:
                parsed = json.loads(input_text) if isinstance(input_text, str) else input_text
            except json.JSONDecodeError:
                raise ValueError(
                    "observation_schema type 'object' requires valid JSON input"
                ) from None
            if not isinstance(parsed, dict):
                raise ValueError("observation_schema type 'object' requires a JSON object")
            for field in schema.get("required", []):
                if field not in parsed:
                    raise ValueError(
                        f"observation_schema: required field '{field}' missing from input"
                    )

        elif schema_type == "array":
            try:
                parsed = json.loads(input_text) if isinstance(input_text, str) else input_text
            except json.JSONDecodeError:
                raise ValueError(
                    "observation_schema type 'array' requires valid JSON input"
                ) from None
            if not isinstance(parsed, list):
                raise ValueError("observation_schema type 'array' requires a JSON array")

    async def run(self, input_text: str) -> PipelineState:
        """
        Execute the pipeline with given input.

        Args:
            input_text: Input text for pipeline

        Returns:
            Final pipeline state
        """
        # Validate input against observation_schema if defined
        obs_schema = self.config.get("observation_schema", {})
        if obs_schema:
            self._validate_observation(input_text, obs_schema)

        # Initialize state
        state = PipelineState(
            pipeline_id=str(uuid.uuid4()),
            template_id=self.config["id"],
            template_version=self.config["version"],
            input_text=input_text,
            input_hash=hashlib.sha256(input_text.encode()).hexdigest(),
            stage_outputs={},
            stage_iterations={},
            stage_traces=[],
            step_count=0,
            step_history=[],
            verdict=None,
            is_complete=False,
            error=None,
            timestamp_start=datetime.utcnow().isoformat(),
            config=self.config,
            next_stage=None,
        )

        # Run graph
        try:
            result = await self.graph.ainvoke(state)
            # Handle both PipelineState object and dict returns from LangGraph
            if isinstance(result, dict) and not isinstance(result, PipelineState):
                # Copy dict values back to state object
                for key, value in result.items():
                    if hasattr(state, key):
                        setattr(state, key, value)
                final_state = state
            else:
                final_state = result
        except Exception as e:
            final_state = state
            final_state.error = str(e)
            final_state.is_complete = True

        # Finalize timestamps
        final_state.timestamp_end = datetime.utcnow().isoformat()
        if final_state.timestamp_start:
            try:
                start = datetime.fromisoformat(final_state.timestamp_start)
                end = datetime.fromisoformat(final_state.timestamp_end)
                final_state.elapsed_seconds = (end - start).total_seconds()
            except Exception:
                final_state.elapsed_seconds = 0.0

        # Write audit completion
        audit_writer = AuditWriter(self.config.get("audit", {}).get("path", "./audit.jsonl"))
        audit_writer.write(final_state)

        # Route to destinations
        try:
            DestinationRouter.route(final_state, self.config)
        except Exception as e:
            # Log but don't fail
            logger.warning(f"Warning: Destination routing failed: {str(e)}")

        return final_state
