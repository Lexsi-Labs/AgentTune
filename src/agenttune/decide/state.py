"""Pipeline state dataclass for LangGraph execution."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

PASS_VERDICTS = frozenset({"PASS", "APPROVE", "COMPLETE"})


@dataclass
class PipelineState:
    """
    State object for pipeline execution in LangGraph.

    Attributes:
        pipeline_id: Unique pipeline execution identifier
        template_id: Template identifier (e.g., "bfsi/kyc_triage")
        template_version: Template version
        input_text: Original input text
        input_hash: SHA256 hash of input
        stage_outputs: Dictionary of stage_id -> output
        stage_iterations: Dictionary of stage_id -> iteration count
        stage_traces: List of execution traces for audit
        step_count: Total steps executed
        step_history: List of executed step IDs
        verdict: Final verdict (e.g., APPROVE, DENY)
        verdict_label: Categorical label for verdict
        confidence: Confidence score (0-10)
        reason: Explanation for verdict
        is_complete: Whether pipeline completed
        error: Error message if any
        error_stage: Stage ID where error occurred
        timestamp_start: ISO 8601 start timestamp
        timestamp_end: ISO 8601 end timestamp
        elapsed_seconds: Total elapsed time
        config: Merged configuration dictionary
    """

    pipeline_id: str
    template_id: str
    template_version: str
    input_text: str
    input_hash: str

    stage_outputs: dict[str, Any] = field(default_factory=dict)
    stage_iterations: dict[str, int] = field(default_factory=dict)
    stage_traces: list[dict[str, Any]] = field(default_factory=list)

    step_count: int = 0
    step_history: list[str] = field(default_factory=list)

    verdict: str | None = None
    verdict_label: str | None = None
    confidence: int | None = None
    reason: str | None = None

    is_complete: bool = False
    error: str | None = None
    error_stage: str | None = None

    timestamp_start: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    timestamp_end: str | None = None
    elapsed_seconds: float = 0.0

    config: dict[str, Any] = field(default_factory=dict)
    next_stage: str | None = None

    def __post_init__(self) -> None:
        """Initialize timestamps if not already set."""
        if not self.timestamp_start:
            self.timestamp_start = datetime.utcnow().isoformat()

    def get_stage_output(self, stage_id: str) -> Any:
        """
        Get output from a specific stage.

        Args:
            stage_id: Stage identifier

        Returns:
            Output dictionary from the stage
        """
        return self.stage_outputs.get(stage_id, {})

    def get_stage_iteration(self, stage_id: str) -> int:
        """
        Get iteration count for a specific stage.

        Args:
            stage_id: Stage identifier

        Returns:
            Iteration count (0 if stage not executed)
        """
        return self.stage_iterations.get(stage_id, 0)

    def add_stage_output(self, stage_id: str, output: Any) -> None:
        """
        Add output from a stage execution.

        Args:
            stage_id: Stage identifier
            output: Output dictionary/value to store
        """
        self.stage_outputs[stage_id] = output

    def increment_stage_iteration(self, stage_id: str) -> int:
        """
        Increment iteration count for a stage.

        Args:
            stage_id: Stage identifier

        Returns:
            The new iteration count after incrementing
        """
        self.stage_iterations[stage_id] = self.stage_iterations.get(stage_id, 0) + 1
        return self.stage_iterations[stage_id]

    def add_trace(self, trace: dict[str, Any]) -> None:
        """
        Add execution trace entry.

        Args:
            trace: Trace dictionary with execution details
        """
        self.stage_traces.append(trace)
