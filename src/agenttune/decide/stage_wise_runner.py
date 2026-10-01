"""Stage-wise training runner for AgentTune Decide module."""

from typing import Any


class StageWiseRunner:
    """
    Runner that executes pipeline stages one at a time for training data collection.

    Wraps GraphRunner to expose a stage-at-a-time interface suitable for
    generating stage-specific training examples.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @staticmethod
    def from_template(template_id: str, config_path: str = "./config.yaml") -> "StageWiseRunner":
        from agenttune.decide.config import ConfigLoader

        config = ConfigLoader.load(template_id, config_path)
        return StageWiseRunner(config)

    async def run_stage_wise(
        self,
        input_text: str,
        collect_outputs: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Run each pipeline stage in sequence and return per-stage results.

        Args:
            input_text: Input to the pipeline.
            collect_outputs: Whether to collect stage outputs.

        Returns:
            List of dicts, one per stage, with keys: stage_id, output, step.
        """
        import hashlib
        import uuid

        from agenttune.decide.stage_executor import StageExecutor
        from agenttune.decide.state import PipelineState

        state = PipelineState(
            pipeline_id=str(uuid.uuid4()),
            template_id=self.config.get("id", "unknown"),
            template_version=self.config.get("version", "1.0.0"),
            input_text=input_text,
            input_hash=hashlib.sha256(input_text.encode()).hexdigest(),
        )

        results = []
        for stage in self.config.get("stages", []):
            handler_class = StageExecutor.get(stage["type"])
            handler = handler_class(stage)
            handler.global_config = self.config
            result = await handler.execute(state, stage_config=stage)
            state.stage_outputs[stage["id"]] = result.get("output", {})
            state.step_count += 1
            if collect_outputs:
                results.append(
                    {
                        "stage_id": stage["id"],
                        "output": result.get("output", {}),
                        "step": state.step_count,
                    }
                )

        return results
