"""OpenEnvHarness against a *real* OpenEnv Environment — not a hand-rolled fake.

`OpenEnvHarness` wraps a gym-like OpenEnv ``Environment`` (``reset`` / ``step``) so the spine's
conformance / replay / EventLog machinery applies to it. Its unit tests drive it with a small
in-file `FakeEnv` (by design — the harness must import and work even if `openenv` weren't
installed). This example does the part the tests can't: it builds a **genuine** OpenEnv environment
— a real `openenv.core.Environment` subclass, with real `openenv` `Observation` / `Action`
Pydantic types and a real `openenv` `Rubric` computing the reward — and drives it through the
full spine.

The environment is a real interactive game (higher/lower number guessing). A binary-search agent
plays it through `OpenEnvHarness` + `run_episode`, reading the env's real observations to narrow
its range. Then the same real env is checked with `run_conformance` (the harness honours its
declared capabilities) and `replay` (a recorded log re-executes deterministically through the env).

Honest scope: everything OpenEnv here is real — the `Environment` base class, the `Observation`
and `Action` types, the `Rubric` reward. This is the *in-process* real env; a fully *remote*
OpenEnv env (HTTP `SyncEnvClient` ↔ a containerised server) is the same wrapper over a network
transport and is what real remote OpenEnv deployments use. The agent policy is a deterministic
binary search — the point proven here is the harness ↔ real-OpenEnv integration, not model
quality (that lives in the other `examples/` scripts).

`openenv` is a base dependency of `agenttune` — no extra install needed. No GPU, no network, no Docker.

    python examples/openenv_harness_real.py
"""

from __future__ import annotations

try:
    from openenv.core import Action, Environment, Observation, State
    from openenv.core.rubrics import Rubric
except Exception as exc:  # pragma: no cover - exercised only if openenv is somehow missing
    raise SystemExit(  # noqa: B904
        "openenv failed to import even though it is a base dependency of agenttune — "
        f"reinstall with: pip install -e .\n(import failed: {exc})"
    )

from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.harness import replay, run_conformance
from agenttune.agentic.harness_openenv import OpenEnvHarness
from agenttune.agentic.strategy import ReActStrategy, run_episode

# ---- real OpenEnv types ---------------------------------------------------


class GuessObs(Observation):
    """A real openenv Observation (inherits done / reward / metadata), plus the text the
    agent reads."""

    text: str = ""


class GuessAction(Action):
    """A real openenv Action carrying the guessed number."""

    n: int = 0


class CorrectGuessRubric(Rubric):
    """A real openenv Rubric: reward 1.0 exactly when the episode terminated on a correct
    guess, else 0.0. Computed by the Environment via `_apply_rubric`."""

    def forward(self, action, observation) -> float:
        return 1.0 if getattr(observation, "done", False) else 0.0


class HigherLowerEnv(Environment):
    """A genuine OpenEnv environment: guess the secret number in [lo, hi]. `step` returns a
    real GuessObs whose reward comes from the real Rubric."""

    def __init__(self, secret: int, lo: int = 1, hi: int = 100):
        super().__init__(rubric=CorrectGuessRubric())
        self._secret, self._lo, self._hi = secret, lo, hi
        self._steps = 0

    @property
    def state(self) -> State:
        return State(step_count=self._steps)

    def reset(self, seed=None, episode_id=None, **kwargs) -> GuessObs:
        self._reset_rubric()
        self._steps = 0
        return GuessObs(
            text=f"I'm thinking of a number between {self._lo} and {self._hi}. " f"Guess it."
        )

    def step(self, action: GuessAction, timeout_s=None, **kwargs) -> GuessObs:
        self._steps += 1
        guess = int(action.n)
        correct = guess == self._secret
        text = "Correct!" if correct else ("higher" if guess < self._secret else "lower")
        obs = GuessObs(text=text, done=correct)
        obs.reward = self._apply_rubric(action, obs)  # real rubric-computed reward
        return obs


# ---- a deterministic binary-search agent that reads the env's feedback ----


def make_binary_search_policy(lo: int, hi: int):
    def policy(state):
        sc = state.scratch
        sc.setdefault("lo", lo)
        sc.setdefault("hi", hi)
        last = state.events[-1].payload.get("text", "") if state.events else ""
        if "higher" in last and "guess" in sc:
            sc["lo"] = sc["guess"] + 1
        elif "lower" in last and "guess" in sc:
            sc["hi"] = sc["guess"] - 1
        sc["guess"] = (sc["lo"] + sc["hi"]) // 2
        return {
            "name": "guess",
            "arguments": {"n": sc["guess"]},
            "thought": f"range [{sc['lo']}, {sc['hi']}] -> guess {sc['guess']}",
        }

    return policy


LO, HI, SECRET = 1, 100, 73


def build_harness() -> OpenEnvHarness:
    """A fresh harness over a fresh real env — dict action -> real GuessAction via the adapter."""
    return OpenEnvHarness(
        HigherLowerEnv(SECRET, LO, HI),
        action_space=[{"name": "guess", "arguments": {"n": 0}}],
        action_adapter=lambda a: GuessAction(n=int(a["arguments"]["n"])),
        max_steps=12,
    )


def main() -> None:
    print(
        f"[env]     real openenv Environment  (HigherLowerEnv, secret={SECRET}, "
        f"rubric={CorrectGuessRubric.__name__})"
    )

    # 1. one explicit reset/step to show the real reward path through OpenEnvHarness.step
    h0 = build_harness()
    obs0 = h0.reset("guess the number")
    print(f"[reset]   obs -> {obs0.text!r}")
    o, reward, done, info = h0.step({"name": "guess", "arguments": {"n": SECRET}})
    print(
        f"[step]    guess {SECRET} -> obs={o.text!r} reward={reward} done={done}  "
        f"(reward from the real Rubric)"
    )

    # 2. full episode: binary-search agent plays the real env through run_episode
    harness = build_harness()
    strategy = ReActStrategy(make_binary_search_policy(LO, HI), max_steps=12)
    log = run_episode(strategy, harness, "guess the secret number")
    guesses = [e.payload["action"]["arguments"]["n"] for e in log if e.kind is EventKind.TOOL_CALL]
    results = [e.payload["output"] for e in log if e.kind is EventKind.TOOL_RESULT]
    solved = results[-1] == "Correct!" if results else False
    print(f"[episode] agent guesses: {guesses}")
    print(
        f"[episode] solved in {len(guesses)} guesses (log tier={log.tier}, {len(log)} events): "
        f"{solved}"
    )

    # 3. conformance: the harness honours its declared capabilities, over the real env
    rep = run_conformance(build_harness())
    print(f"[conform] run_conformance -> passed={rep.passed}, drift={rep.drift}")

    # 4. replay: a recorded log re-executes deterministically through a fresh real env
    src = EventLog(
        events=[
            Event(EventKind.OBSERVATION, {"text": "start"}),
            Event(EventKind.TOOL_CALL, {"action": {"name": "guess", "arguments": {"n": 50}}}),
            Event(EventKind.TOOL_CALL, {"action": {"name": "guess", "arguments": {"n": 73}}}),
        ]
    )
    out = replay(build_harness(), src)
    replayed = [e.payload["output"] for e in out if e.kind is EventKind.TOOL_RESULT]
    print(f"[replay]  re-executed {len(replayed)} tool calls through the real env -> {replayed}")

    ok = solved and rep.passed and reward == 1.0 and replayed == ["higher", "Correct!"]
    print(
        f"[verdict] real OpenEnv env driven through run_episode + conformance + replay, "
        f"real rubric reward: {ok}"
    )


if __name__ == "__main__":
    main()
