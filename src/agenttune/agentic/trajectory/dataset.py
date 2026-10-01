import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Rollout metadata that is token-level training state, not part of the record.
_TRAINING_ONLY_METADATA = ("prompt_ids", "completion_ids", "tool_mask", "completions_list")


def _portable_call(call: dict[str, Any]) -> dict[str, Any]:
    """One parsed call (OpenAI ``{"function": {...}}`` or flat) as
    ``{"id", "name", "arguments": dict}``. Arguments that are not a JSON object
    stay the raw string, so a consumer can flag them instead of seeing ``{}``."""
    fn = call["function"] if isinstance(call.get("function"), dict) else call
    name = (
        fn.get("name")
        or fn.get("tool_name")
        or (call.get("function") if isinstance(call.get("function"), str) else None)
    )
    args = fn.get("arguments", fn.get("parameters", {}))
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            args = parsed
    return {"id": call.get("id"), "name": name, "arguments": args}


@dataclass
class Step:
    step_number: int
    state: str
    action: dict[str, Any]  # {"name": "read_file", "arguments": {...}}
    observation: str
    thought: str | None = None
    reward: float | None = None


@dataclass
class Trajectory:
    task: str
    steps: list[Step]
    reward: float = 0.0
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    final_response: str = ""
    logprobs: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The portable JSONL record read by :meth:`TrajectoryDataset.from_jsonl` and
        AuditKIT's ``episodes_from_agenttune``. Each step's tool calls are written as
        ``{"id", "name", "arguments": dict}`` whatever the model family's native
        format; logprobs and token ids are dropped."""
        d = asdict(self)
        d.pop("logprobs", None)
        for key in _TRAINING_ONLY_METADATA:
            d["metadata"].pop(key, None)
        for step in d["steps"]:
            action = step["action"]
            if isinstance(action, dict) and isinstance(action.get("tool_calls"), list):
                calls = [c for c in action["tool_calls"] if isinstance(c, dict)]
                step["action"] = {"tool_calls": [_portable_call(c) for c in calls]}
        return d

    def to_trl_format(self) -> dict[str, Any]:
        """Convert to TRL-compatible message format."""
        messages = []

        for step in self.steps:

            # Case 1: Tool call step
            if step.action and step.action.get("name"):
                call_id = f"call_{step.step_number}"

                # Assistant tool call message
                messages.append(
                    {
                        "role": "assistant",
                        "content": step.thought or "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": step.action["name"],
                                    "arguments": json.dumps(step.action.get("arguments", {})),
                                },
                            }
                        ],
                    }
                )

                # Tool response message
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": step.observation,
                    }
                )

            # Case 2: Normal assistant response (no tool)
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": step.observation,
                    }
                )

        return {
            "messages": messages,
            "reward": self.reward,
        }


class TrajectoryDataset:
    """
    Wraps a list of Trajectory objects for use with TRL dataloaders.
    Compatible with torch Dataset interface.
    """

    def __init__(self, trajectories: list[Trajectory]):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx].to_trl_format()

    @classmethod
    def from_jsonl(cls, path: str) -> "TrajectoryDataset":
        trajectories = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                steps = [Step(**s) for s in data.get("steps", [])]
                trajectories.append(
                    Trajectory(
                        task=data["task"],
                        steps=steps,
                        reward=data.get("reward", 0.0),
                        trajectory_id=data.get("trajectory_id", str(uuid.uuid4())),
                        final_response=data.get("final_response", ""),
                        metadata=data.get("metadata", {}),
                    )
                )
        return cls(trajectories)


def trajectory_writer(path: str | Path, keep_reward: bool = False) -> Callable[[Trajectory], None]:
    """An ``on_trajectory_end`` hook that appends each trajectory to ``path`` as
    one :meth:`Trajectory.to_dict` JSON line.

    ``reward`` is written as null unless ``keep_reward``: trainers score a rollout
    after this hook runs, so ``Trajectory.reward`` still holds its 0.0 default there.
    Keep it when ``create_rollout_fn`` itself was given a ``reward_fn``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _write(traj: Trajectory) -> None:
        record = traj.to_dict()
        if not keep_reward:
            record["reward"] = None
        # ponytail: every rank appends one line per write; per-rank files if interleaving shows up.
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    return _write
