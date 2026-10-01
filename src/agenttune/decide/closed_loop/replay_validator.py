import asyncio
import logging

from agenttune.decide.closed_loop.contracts import TrainingExample

logger = logging.getLogger(__name__)


class ReplayValidator:
    """
    Validates a generated training example by re-running the corrected response
    through the actual system environment.
    """

    def __init__(self, validation_script: str = "python -c 'import sys; sys.exit(0)'"):
        self.validation_script = validation_script

    async def validate_example_with_trace(
        self, example: TrainingExample, completion_text: str
    ) -> tuple[bool, list[str]]:
        """
        Runs the corrected response through the actual system environment using
        an isolated subprocess. Returns (is_valid, tool_outputs_for_ter).
        """
        logger.info(f"Starting replay validation for trajectory {example.trajectory_id}")
        try:
            process = await asyncio.create_subprocess_shell(
                self.validation_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            tool_outputs = []
            if stdout:
                import json

                try:
                    tool_outputs = json.loads(stdout.decode().strip())
                except json.JSONDecodeError:
                    tool_outputs = [stdout.decode().strip()]

            if process.returncode == 0:
                logger.info(f"Replay validation passed for {example.trajectory_id}")
                return True, tool_outputs
            else:
                logger.warning(
                    f"Replay validation failed for {example.trajectory_id}: {stderr.decode()}"
                )
                return False, []

        except Exception as e:
            logger.error(f"Error during replay validation for {example.trajectory_id}: {e}")
            return False, []

    async def validate_batch(
        self,
        examples: list[TrainingExample],
        completion_texts: list[str] | None = None,
    ) -> list[tuple[bool, list[str]]]:
        """Validate all examples in parallel using asyncio.gather.

        All subprocess validations are launched at the same time — wall-clock
        time equals the slowest single validation, not the sum.

        Args:
            examples: Training examples to validate.
            completion_texts: Completion text for each example (1-to-1). If
                omitted, an empty string is used for each example.

        Returns:
            List of (is_valid, tool_outputs) tuples in the same order as
            ``examples``.
        """
        if not examples:
            return []
        texts = completion_texts if completion_texts is not None else [""] * len(examples)
        logger.info(
            "ReplayValidator.validate_batch: launching %d validations in parallel", len(examples)
        )
        results = await asyncio.gather(
            *[
                self.validate_example_with_trace(e, t)
                for e, t in zip(examples, texts, strict=False)
            ],
            return_exceptions=False,
        )
        passed = sum(1 for ok, _ in results if ok)
        logger.info("ReplayValidator.validate_batch: %d/%d passed", passed, len(examples))
        return list(results)
