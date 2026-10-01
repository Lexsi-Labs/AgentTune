"""Parallel execution stage for fan-out/rejoin patterns."""

import asyncio
from typing import Any

from agenttune.decide.stage_executor import StageExecutor
from agenttune.decide.stages.base import StageHandler
from agenttune.decide.state import PipelineState


class ParallelStage(StageHandler):
    """
    Stage for parallel execution of multiple branches.

    Executes branches concurrently and collects results.
    """

    async def execute(
        self, state: PipelineState, stage_config: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Execute parallel branches.

        Args:
            state: Pipeline state
            stage_config: Stage configuration (optional, uses self.stage_config if not provided)

        Returns:
            Dictionary with branch results
        """
        config = stage_config or self.stage_config
        branches = config.get("branches", [])
        if not branches:
            return {"output": {}}

        # Create tasks for all branches
        tasks = [self._execute_branch(branch, state) for branch in branches]

        # Run all branches concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            return {
                "output": {},
                "error": f"Parallel execution failed: {str(e)}",
            }

        # Collect results keyed by branch ID
        output = {}
        for branch, result in zip(branches, results, strict=False):
            branch_id = branch.get("id")
            if isinstance(result, Exception):
                output[branch_id] = {"error": str(result)}
            else:
                output[branch_id] = result.get("output", result)

        return {"output": output}

    async def _execute_branch(self, branch: dict[str, Any], state: PipelineState) -> dict[str, Any]:
        """
        Execute a single branch.

        Supports two branch shapes:
        - Flat (the branch dict itself is one stage config, e.g. `{id, type: llm_call,
          prompt, ...}`) — the format every bundled template actually uses.
        - Nested (`{id, stages: [...]}`, a list of sub-stage configs run sequentially) —
          for a branch that's itself a small multi-stage pipeline.

        Args:
            branch: Branch configuration (flat stage config, or {id, stages: [...]})
            state: Pipeline state

        Returns:
            Branch execution result with combined outputs
        """
        try:
            stages = branch.get("stages")
            if stages:
                branch_output = {}

                # Execute stages in branch sequentially
                for stage_config in stages:
                    stage_type = stage_config.get("type")
                    if not stage_type:
                        continue

                    # Get handler for this stage type
                    handler_class = StageExecutor.get(stage_type)
                    handler = handler_class(stage_config)
                    # Inherit global_config/shared engine from parent parallel stage
                    if hasattr(self, "global_config"):
                        handler.global_config = self.global_config
                    if hasattr(self, "_shared_engine"):
                        handler._shared_engine = self._shared_engine

                    # Execute stage
                    result = await handler.execute(state, stage_config=stage_config)
                    stage_id = stage_config.get("id")
                    branch_output[stage_id] = result.get("output", {})

                return {"output": branch_output}

            # Flat branch: the branch dict itself is a single stage config.
            stage_type = branch.get("type")
            if not stage_type:
                return {"output": {}}

            handler_class = StageExecutor.get(stage_type)
            handler = handler_class(branch)
            if hasattr(self, "global_config"):
                handler.global_config = self.global_config
            if hasattr(self, "_shared_engine"):
                handler._shared_engine = self._shared_engine

            result = await handler.execute(state, stage_config=branch)
            return {"output": result.get("output", {}), "error": result.get("error")}
        except Exception as e:
            return {"output": None, "error": str(e)}
