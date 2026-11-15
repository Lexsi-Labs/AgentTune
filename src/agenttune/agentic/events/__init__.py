"""Normalized agent event stream — the shared trace model (Phase 1 of the spine).

Additive layer: projects the existing full-tier ``agentic.trajectory.Trajectory``
and light-tier DECIDE/eval structures into one ``EventLog``. Nothing existing is
modified.

Two tiers:
  - full: carries token_span + logprobs (from the harness/rollout path); trainable.
  - light: observational only (DECIDE PipelineState / eval dicts); UI/eval/heal read it.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum


class EventKind(Enum):
    TEXT = "text"  # model-emitted text
    REASONING = "reasoning"  # model-emitted reasoning/thought
    TOOL_CALL = "tool_call"  # a tool/action invocation
    TOOL_RESULT = "tool_result"  # the tool's returned output
    OBSERVATION = "observation"  # environment/state fed back to the agent
    TURN_COMPLETE = "turn_complete"  # end of one agent turn
    REWARD = "reward"  # a reward signal (see scope)
    MEMORY_OP = "memory_op"  # ADD/UPDATE/DELETE/NOOP on memory


@dataclass
class Event:
    kind: EventKind
    payload: dict
    token_span: tuple[int, int] | None = None  # policy-emitted span; full tier only
    logprobs: list[float] | None = None  # full tier only
    scope: str | None = None  # REWARD only: step|turn|episode


@dataclass
class EventLog:
    events: list[Event] = field(default_factory=list)
    tier: str = "light"  # "light" (observational) | "full" (maskable)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def append(self, event: Event) -> None:
        self.events.append(event)

    def __iter__(self):
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def rewards(self, scope: str) -> list[float]:
        return [
            e.payload["value"]
            for e in self.events
            if e.kind is EventKind.REWARD and e.scope == scope
        ]

    def _require_full(self, op: str) -> None:
        if self.tier != "full":
            raise ValueError(
                f"{op} requires a full-tier EventLog (token spans/logprobs come only "
                f"from the harness/rollout path); this log is tier={self.tier!r}."
            )

    def masked_tokens(self) -> list[tuple[int, int]]:
        self._require_full("masked_tokens()")
        return [e.token_span for e in self.events if e.token_span is not None]

    @staticmethod
    def _unwrap_tool_calls(action: dict) -> list[dict]:
        """Normalise a recorded step action into bare per-call dicts that both
        ``_extract_tool_calls`` (inference parser) and the SFT chat template
        expect: ``{"name": <fn>, "arguments": <json object>}``.

        Real rollouts record ``action = {"tool_calls": [openai-function-call, ...]}``
        (see ``rollout_engines/rollout_factory.py``), while spine/eval trajectories
        may record a bare ``{"name": ..., "arguments": ...}`` directly. OpenAI-style
        calls (``{"type": "function", "function": {...}}``) are converted to the
        bare shape, and string-encoded arguments are parsed back to objects so the
        model learns to emit JSON rather than a JSON-encoded string.
        """
        if isinstance(action, dict) and isinstance(action.get("tool_calls"), list):
            calls = action["tool_calls"]
        else:
            calls = [action]

        bare: list[dict] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            if isinstance(call.get("function"), dict):
                fn = call["function"]
                name = fn.get("name")
                arguments = fn.get("arguments", {})
            else:
                name = call.get("name")
                arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
            if name:
                bare.append({"name": name, "arguments": arguments})
        return bare

    def as_dataset_rows(self, fmt: str) -> list[dict]:
        self._require_full("as_dataset_rows()")
        if fmt != "sft":
            raise ValueError(f"unknown fmt {fmt!r}; supported: 'sft'")

        messages: list[dict] = []
        segment_weights: list[float] = []  # 1.0 for reasoning, 2.0 for action, 0.0 for observation

        for e in self.events:
            if e.kind in (EventKind.REASONING, EventKind.TEXT):
                messages.append({"role": "assistant", "content": e.payload.get("text", "")})
                segment_weights.append(1.0)  # Normal loss for reasoning (thought)
            elif e.kind is EventKind.TOOL_CALL:
                # Must be valid JSON in the bare {"name", "arguments"} shape, not
                # Python repr - rollout_factory._extract_tool_calls() does
                # json.loads() on this tag's content at inference time, and only
                # accepts a dict carrying a top-level "name" key. Real rollouts
                # store the OpenAI wrapper {"tool_calls": [...]}, so it is unwrapped
                # here - otherwise an f-string interpolating the raw wrapper
                # silently teaches the model a format its own parser can't read
                # (one message per unwrapped call keeps multi-call steps parseable).
                bare_calls = self._unwrap_tool_calls(e.payload.get("action") or {})
                if not bare_calls:
                    bare_calls = [e.payload.get("action") or {}]
                for call in bare_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"<tool_call>{json.dumps(call)}</tool_call>",
                        }
                    )
                    segment_weights.append(
                        2.0
                    )  # Double penalty for messing up syntax/actions (Segment-Aware Loss)
            elif e.kind is EventKind.TOOL_RESULT:
                messages.append({"role": "tool", "content": str(e.payload.get("output", ""))})
                segment_weights.append(0.0)  # E1 Masking: Do not train on environment observations
            elif e.kind is EventKind.OBSERVATION:
                messages.append({"role": "user", "content": str(e.payload.get("text", ""))})
                segment_weights.append(0.0)

        return [
            {
                "messages": messages,
                "segment_weights": segment_weights,
                "loss_mask": [w == 0.0 for w in segment_weights],
            }
        ]

    def to_eval_dict(self) -> dict:
        """Project into the dict the existing ``TrajectoryEvaluator`` consumes:
        ``{tool_calls, tool_outputs}``. Inverse of ``from_eval_dict`` and the
        eval-relevant slice of ``from_trajectory`` — both store a tool call under
        ``payload['action']`` and its result under ``payload['output']``. This is the
        wrap that lets a trajectory reach the real evaluator with no rewrite."""
        tool_calls = [
            e.payload.get("action")
            for e in self.events
            if e.kind is EventKind.TOOL_CALL and "action" in e.payload
        ]
        tool_outputs = [
            e.payload.get("output") for e in self.events if e.kind is EventKind.TOOL_RESULT
        ]
        return {"tool_calls": tool_calls, "tool_outputs": tool_outputs}

    def to_audit_records(self) -> list[dict]:
        """Project into the audit-log records the existing closed-loop ``FailureDetector``
        consumes (one record per tool call), so its REAL loop_collapse/tool_crash logic
        applies to a spine trajectory. ``trajectory_id`` = this log's id; a following
        ``TOOL_RESULT`` carrying an ``error`` marks its call crashed.

        ``stage_name`` encodes the full action identity (tool name **plus** arguments), not
        just the tool name. The detector's loop_collapse counts identical ``stage_name``s: in
        DECIDE a stage is a fixed pipeline node so a revisit is a routing loop, but a tool can
        legitimately be re-called with *different* args (normal ReAct). Folding args in means
        only a truly repeated identical action — the real agentic loop — fires loop_collapse."""
        import json as _json

        records: list[dict] = []
        for e in self.events:
            if e.kind is EventKind.TOOL_CALL:
                action = e.payload.get("action") or {}
                name = action.get("name") if isinstance(action, dict) else str(action)
                args = action.get("arguments", {}) if isinstance(action, dict) else {}
                name = name or "unknown"
                stage_name = name
                if args:
                    stage_name = f"{name} {_json.dumps(args, sort_keys=True, default=str)}"
                records.append(
                    {
                        "trajectory_id": self.id,
                        "stage_name": stage_name,
                        "tool_name": name,
                        "stage_type": "tool_call",
                        "state_snapshot": {"arguments": args},
                        "status": e.payload.get("status", "ok"),
                    }
                )
            elif e.kind is EventKind.TOOL_RESULT and e.payload.get("error") and records:
                records[-1]["status"] = "error"
                records[-1]["error_details"] = str(e.payload.get("error"))
        return records

    # ---- projections (wrap, don't rewrite) ----

    @classmethod
    def from_trajectory(cls, traj) -> EventLog:
        """Project the existing full-tier agentic Trajectory into an EventLog."""
        log = cls(tier="full")
        # The original task the whole trajectory is a response to -- without this,
        # as_dataset_rows()'s "messages" starts directly at the first
        # REASONING/TOOL_CALL (role="assistant"), training a student on a
        # conversation that never records what it was actually asked (and, for a
        # chat template that requires the conversation to open on "user", like
        # CohereLabs/tiny-aya-fire's, crashes outright). _demo_trajectory (the
        # sibling "demonstrations" trajectory builder in distill_trainer.py)
        # already gets this right the same way; from_trajectory just never read
        # `traj.task` at all.
        if getattr(traj, "task", None):
            log.append(Event(EventKind.OBSERVATION, {"text": traj.task}))
        for step in traj.steps:
            if step.thought:
                log.append(Event(EventKind.REASONING, {"text": step.thought}))
            log.append(Event(EventKind.TOOL_CALL, {"action": step.action}))
            log.append(Event(EventKind.TOOL_RESULT, {"output": step.observation}))
            if step.reward is not None:
                log.append(Event(EventKind.REWARD, {"value": step.reward}, scope="step"))
            log.append(Event(EventKind.TURN_COMPLETE, {"step": step.step_number}))
        logprobs = traj.logprobs if isinstance(traj.logprobs, list) and traj.logprobs else None
        span = (0, len(logprobs)) if logprobs else None
        log.append(
            Event(EventKind.TEXT, {"text": traj.final_response}, token_span=span, logprobs=logprobs)
        )
        log.append(Event(EventKind.REWARD, {"value": traj.reward}, scope="episode"))
        return log

    @classmethod
    def from_pipeline_state(cls, state) -> EventLog:
        """Project a DECIDE PipelineState into a light-tier (observational) EventLog."""
        log = cls(tier="light")
        log.append(Event(EventKind.OBSERVATION, {"text": state.input_text}))
        for stage_id in state.step_history:
            log.append(Event(EventKind.TOOL_CALL, {"stage": stage_id}))
            log.append(Event(EventKind.TOOL_RESULT, {"output": state.stage_outputs.get(stage_id)}))
        log.append(Event(EventKind.TEXT, {"text": state.reason or ""}))
        if state.confidence is not None:
            log.append(Event(EventKind.REWARD, {"value": state.confidence / 10.0}, scope="episode"))
        return log

    @classmethod
    def from_eval_dict(cls, d: dict) -> EventLog:
        """Project an eval trajectory dict ({tool_calls, tool_outputs}) into a light log."""
        log = cls(tier="light")
        for call in d.get("tool_calls", []):
            log.append(Event(EventKind.TOOL_CALL, {"action": call}))
        for out in d.get("tool_outputs", []):
            log.append(Event(EventKind.TOOL_RESULT, {"output": out}))
        return log
