"""OpenEnvHarness — the OpenEnv adapter behind the Harness contract (Phase 2b).

A remote/local OpenEnv ``Environment`` is gym-like (``reset()`` / ``step(action)``),
which is exactly the ``Harness`` shape. This module wraps such an env object as a
``Harness`` so the spine's conformance/replay/EventLog machinery applies to it
unchanged.

Design notes:
- The env is INJECTED (dependency injection). Nothing here imports ``openenv`` — the
  wrapper only talks to a plain object exposing ``reset()`` / ``step(action)``. So this
  module imports fine on a machine without the openenv extra, and the tests drive it
  with a small fake env (no GPU / network / openenv). Any convenience constructor that
  builds a real OpenEnv env guards the import lazily via ``agenttune.utils.optional``.
- OpenEnv's ``step()`` returns a StepResult-like object (fields ``observation`` /
  ``reward`` / ``done``); we extract those tolerantly (``_extract``) so a bare
  observation object also works. Exact field names are treated as best-effort — see
  ``_extract`` — because the real StepResult schema is only exercised behind the
  optional openenv install.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agenttune.agentic.events import Event, EventKind
from agenttune.agentic.harness import Harness, HarnessCapabilities, Observation


def _obs_text(obs: Any) -> str:
    """Best-effort textualization of an OpenEnv observation."""
    if obs is None:
        return ""
    text = getattr(obs, "text", None)
    if isinstance(text, str):
        return text
    return str(obs)


def _extract(result: Any) -> tuple[Any, float, bool]:
    """Pull (observation, reward, done) from an OpenEnv StepResult-like object.

    Tolerant on purpose: a StepResult exposes ``observation`` / ``reward`` / ``done``,
    but a plain observation object (no wrapper) falls back to itself with reward 0.0
    and done False.
    """
    obs = getattr(result, "observation", result)
    reward = getattr(result, "reward", None)
    reward = 0.0 if reward is None else float(reward)
    done = bool(getattr(result, "done", False))
    return obs, reward, done


class OpenEnvHarness(Harness):
    """Adapt an OpenEnv ``Environment`` object to the ``Harness`` interface.

    Args:
        env: The OpenEnv env object. Must expose ``reset()`` and ``step(action)``;
            ``step`` should return a StepResult-like object (``observation`` /
            ``reward`` / ``done``) or a bare observation. Injected by the caller so
            this class never needs to import openenv.
        action_space: Static list of action specs (dicts). OpenEnv envs don't expose
            a uniform action space, so it is supplied here (or overridden by the env's
            own ``action_space()`` if it has one).
        action_adapter: Optional ``Callable[[dict], Any]`` converting a Harness dict
            action into the env's native Action before calling ``env.step``. Default
            passes the dict through unchanged (fine for fakes; real OpenEnv envs inject
            a converter here).
        max_steps: Optional step cap; when reached ``step`` forces ``done=True``
            (mirrors ``DictToolHarness``).
    """

    def __init__(
        self,
        env: Any,
        *,
        action_space: list[dict] | None = None,
        action_adapter: Callable[[dict], Any] | None = None,
        max_steps: int | None = None,
    ) -> None:
        self._env = env
        self._action_space = action_space if action_space is not None else []
        self._action_adapter = action_adapter or (lambda a: a)
        # Remote OpenEnv envs have no in-process snapshot/restore; leave snapshot at the
        # base NotImplementedError so run_conformance sees a consistent (False) capability.
        self.capabilities = HarnessCapabilities(
            supports_snapshot=False,
            supports_streaming=False,
            supports_tool_boundary_interrupt=False,
            supports_stepwise_turns=True,
            max_steps=max_steps,
        )
        self.event_log: list[Event] = []
        self._task = ""
        self._steps = 0

    def reset(self, task: str) -> Observation:
        self.event_log = []
        self._task = task
        self._steps = 0
        result = self._env.reset()
        obs, _reward, _done = _extract(result)
        # OpenEnv reset() takes no task; the task string seeds the observation the agent
        # sees. Prefer the env's own observation text if it carried one, else the task.
        text = _obs_text(obs) or task
        self.event_log.append(Event(EventKind.OBSERVATION, {"text": text}))
        return Observation(text=text)

    def step(self, action: dict) -> tuple[Observation, float, bool, dict]:
        # Record the ORIGINAL dict action so replay()/to_eval_dict() round-trip; convert
        # only for the env call.
        self.event_log.append(Event(EventKind.TOOL_CALL, {"action": action}))
        self._steps += 1
        result = self._env.step(self._action_adapter(action))
        obs, reward, done = _extract(result)
        text = _obs_text(obs)
        self.event_log.append(Event(EventKind.TOOL_RESULT, {"output": text}))
        self.event_log.append(Event(EventKind.TURN_COMPLETE, {"step": self._steps}))
        if self.capabilities.max_steps is not None and self._steps >= self.capabilities.max_steps:
            done = True
        return Observation(text=text, metadata={"done": done}), reward, done, {"steps": self._steps}

    def action_space(self) -> list[dict]:
        env_space = getattr(self._env, "action_space", None)
        if callable(env_space):
            return env_space()
        return list(self._action_space)
