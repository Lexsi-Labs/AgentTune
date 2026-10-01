"""FastAPI service over the Project lifecycle.

``/collect_rollout`` drives whatever ``RolloutEngine`` you pass to ``create_app``.
The default is ``DemoRolloutEngine`` so unit tests stay GPU-free. A Colab/GPU
caller passes a transformers (or vLLM) engine and the same tools the trainer uses.

GRPO training and the full self-heal loop (classify→regenerate→retrain) are reported
as ``wired-runs-on-gpu`` rather than executed in-process.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

_STATIC = Path(__file__).parent / "static"

from agenttune.agentic.events import Event, EventKind, EventLog
from agenttune.agentic.heal_loop import SelfHealLoop
from agenttune.agentic.project import Project, agentic_metrics
from agenttune.agentic.rollout_engines.demo_engine import DemoRolloutEngine
from agenttune.decide.closed_loop.contracts import ClassifiedFailure, TrainingExample

# ---- request models ----


class CollectRequest(BaseModel):
    tasks: list[str]
    max_steps: int = 4


class DistillRequest(BaseModel):
    student: str = "student-1.5B"


class TrainRequest(BaseModel):
    fmt: str = "grpo"


class StrategyRequest(BaseModel):
    strategy: str = "react"


class MemoryRequest(BaseModel):
    driver: str = "vector"


# ---- serialization helpers ----


def _event_dict(ev) -> dict:
    return {"stage": ev.stage, "kind": ev.kind, "data": ev.data}


def _traj_summary(log) -> dict:
    text = ""
    n_tool_calls = 0
    for e in log:
        if e.kind is EventKind.TOOL_CALL:
            n_tool_calls += 1
        if e.kind is EventKind.TEXT:
            text = str((e.payload or {}).get("text") or "")
    return {
        "id": log.id,
        "tier": log.tier,
        "n_events": len(log),
        "n_tool_calls": n_tool_calls,
        "text": text,
    }


# ---- GPU-free demo runs of each agent design / memory driver ----


def _demo_strategy(name: str):
    """Build one of the five agent designs with a canned, model-free policy that reaches
    the answer '42'. Returns (strategy, harness) or raises KeyError for an unknown name."""
    from agenttune.agentic import (
        DictToolHarness,
        InContextMemory,
        MemoryReActStrategy,
        PlanExecuteStrategy,
        ReActStrategy,
        ReflexionStrategy,
        TreeOfThoughtsStrategy,
    )

    harness = DictToolHarness({"search": lambda q="": f"result for {q}"}, max_steps=5)

    def react_policy(state):
        if state.step == 0:
            return {"name": "search", "arguments": {"q": "answer"}, "thought": "look it up"}
        return {"name": "finish", "arguments": {"answer": "42"}}

    if name == "react":
        return ReActStrategy(react_policy, max_steps=5), harness
    if name == "plan_execute":
        plan = [
            {"name": "search", "arguments": {"q": "answer"}, "thought": "step 1: search"},
            {"name": "finish", "arguments": {"answer": "42"}},
        ]
        return PlanExecuteStrategy(lambda s: list(plan), max_steps=5), harness
    if name == "reflexion":
        return (
            ReflexionStrategy(
                policy=lambda s: {"name": "finish", "arguments": {"answer": "42"}},
                reflect=lambda log: "be more direct next time",
                max_attempts=1,
                max_steps=5,
            ),
            harness,
        )
    if name == "tot":
        cands = [
            {"name": "search", "arguments": {"q": "answer"}},
            {"name": "finish", "arguments": {"answer": "42"}, "thought": "commit the answer"},
        ]
        return (
            TreeOfThoughtsStrategy(
                propose_candidates=lambda s: cands,
                score=lambda s, c: 1.0 if c["name"] == "finish" else 0.3,
                beam_width=2,
                max_steps=5,
            ),
            harness,
        )
    if name == "memory":
        return (
            MemoryReActStrategy(policy=react_policy, memory=InContextMemory(), max_steps=5),
            harness,
        )
    raise KeyError(name)


def _demo_memory(driver: str) -> list[str]:
    """Seed a driver with a few items and return what a query recalls. Shows the
    difference between semantic (vector) and relational (graph) retrieval, model-free."""
    from agenttune.agentic import GraphMemory, MemoryItem, VectorMemory

    if driver == "vector":
        vocab = ["cat", "dog", "car"]
        m = VectorMemory(embed=lambda x: [float(str(x).lower().count(w)) for w in vocab])
        for c in ["cat cat", "dog dog", "car"]:
            m.write(MemoryItem(content=c))
        return [i.content for i in m.read("cat", k=2)]
    if driver == "graph":
        m = GraphMemory()
        a = m.write(MemoryItem(content="Alice"))
        b = m.write(MemoryItem(content="Bob"))
        m.write(MemoryItem(content="Dave"))  # unrelated
        m.link(a, b, "friend")
        return [i.content for i in m.read("Alice", k=5)]
    raise KeyError(driver)


def create_app(rollout_engine=None, tools=None, system_prompt: str | None = None) -> FastAPI:
    app = FastAPI(title="AgentTune Spine", version="1.0")
    projects: dict[str, Project] = {}
    engine = rollout_engine if rollout_engine is not None else DemoRolloutEngine()
    bound_tools = list(tools or [])

    def _get(pid: str) -> Project:
        proj = projects.get(pid)
        if proj is None:
            raise HTTPException(status_code=404, detail=f"unknown project {pid!r}")
        return proj

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.post("/projects")
    def create_project() -> dict:
        pid = uuid.uuid4().hex[:12]
        projects[pid] = Project()
        return {"project_id": pid}

    @app.post("/projects/{pid}/collect_rollout")
    def collect_rollout(pid: str, req: CollectRequest) -> dict:
        proj = _get(pid)
        logs = proj.collect_rollout(
            engine,
            req.tasks,
            tools=bound_tools,
            max_steps=req.max_steps,
            system_prompt=system_prompt,
        )
        return {"n_trajectories": len(logs), "trajectories": [_traj_summary(l) for l in logs]}

    @app.post("/projects/{pid}/evaluate")
    def evaluate(pid: str) -> dict:
        proj = _get(pid)
        # score the trajectories already collected, via the real evaluator metrics
        proj._emit("evaluate_agentic", "started", n=len(proj.trajectories))
        rows = [agentic_metrics(l) for l in proj.trajectories]
        keys = rows[0].keys() if rows else []
        metrics = {k: (sum(r[k] for r in rows) / len(rows) if rows else 0.0) for k in keys}
        proj._emit("evaluate_agentic", "eval_done", n=len(rows))
        return {"n": len(rows), "metrics": metrics}

    @app.post("/projects/{pid}/distill")
    def distill(pid: str, req: DistillRequest) -> dict:
        proj = _get(pid)
        rows = proj.sft_dataset()
        proj._emit("distill", "dataset_ready", student=req.student, n_rows=len(rows))
        return {
            "student": req.student,
            "n_rows": len(rows),
            "sample_row": rows[0] if rows else {},
            "status": "wired-runs-on-gpu",
        }

    @app.post("/projects/{pid}/train")
    def train(pid: str, req: TrainRequest) -> dict:
        proj = _get(pid)
        proj._emit("train", "requested", fmt=req.fmt)
        return {
            "fmt": req.fmt,
            "status": "wired-runs-on-gpu",
            "detail": "RL/SFT training runs on GPU via the wired trainer; "
            "the service exposes the wiring, not an in-process training run.",
        }

    @app.post("/projects/{pid}/heal")
    def heal(pid: str) -> dict:
        proj = _get(pid)
        failures = proj.heal()
        return {
            "failures": [
                {
                    "trajectory_id": f.trajectory_id,
                    "failure_type": f.failure_type,
                    "failed_stage_name": f.failed_stage_name,
                }
                for f in failures
            ]
        }

    @app.post("/projects/{pid}/run_strategy")
    def run_strategy(pid: str, req: StrategyRequest) -> dict:
        """Run one of the five agent designs (react / plan_execute / reflexion / tot /
        memory) with a canned, model-free policy — showcases the strategy pillar GPU-free."""
        from agenttune.agentic.strategy import run_episode

        proj = _get(pid)
        try:
            strategy, harness = _demo_strategy(req.strategy)
        except KeyError:
            raise HTTPException(
                status_code=400, detail=f"unknown strategy {req.strategy!r}"
            ) from None
        proj._emit("run_strategy", "started", strategy=req.strategy)
        log = run_episode(strategy, harness, "what is the answer?")
        proj._trajectories.append(log)
        answer = None
        for e in log:
            act = e.payload.get("action") if e.kind is EventKind.TOOL_CALL else None
            if isinstance(act, dict) and act.get("name") == "finish":
                answer = act.get("arguments", {}).get("answer")
        proj._emit("run_strategy", "done", strategy=req.strategy, n_events=len(log))
        return {"strategy": req.strategy, "n_events": len(log), "answer": answer}

    @app.post("/projects/{pid}/memory_demo")
    def memory_demo(pid: str, req: MemoryRequest) -> dict:
        """Seed a memory driver and show what a query recalls — semantic (vector) vs
        relational (graph). Model-free."""
        _get(pid)
        try:
            recalled = _demo_memory(req.driver)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"unknown driver {req.driver!r}") from None
        return {"driver": req.driver, "recalled": recalled}

    @app.post("/projects/{pid}/seed_demo_failure")
    def seed_demo_failure(pid: str) -> dict:
        """Inject a stuck (looping) agent trajectory so the self-heal demo has a real
        failure to detect. Mirrors what a genuinely looping agent would produce."""
        proj = _get(pid)
        log = EventLog(tier="light")
        for _ in range(4):  # same tool 4x -> loop_collapse (> max_revisits)
            log.append(Event(EventKind.TOOL_CALL, {"action": {"name": "search", "arguments": {}}}))
            log.append(Event(EventKind.TOOL_RESULT, {"output": "same result again"}))
        proj.add_trajectory(log)
        return {"seeded": True, "trajectory_id": log.id}

    @app.post("/projects/{pid}/heal_loop")
    def heal_loop(pid: str) -> dict:
        """Run the full self-heal loop over the project: real detection (FailureDetector)
        → classify → generate → training-dataset. The litellm-bound classify/generate
        stages are deterministic DEMO stand-ins here (the real loop wraps the async
        FailureClassifier/TrainingExampleGenerator on GPU/litellm)."""
        proj = _get(pid)

        def demo_classifier(failures):
            return [
                ClassifiedFailure(
                    failure=f,
                    root_cause=f.failure_type,
                    confidence=0.8,
                    analysis=f"demo analysis: {f.failure_type}",
                )
                for f in failures
            ]

        def demo_generator(classified):
            return [
                TrainingExample(
                    trajectory_id=cf.failure.trajectory_id,
                    original_failure_type=cf.failure.failure_type,
                    root_cause=cf.root_cause,
                    prompt=[{"role": "user", "content": f"Recover from {cf.failure.failure_type}"}],
                    chosen=[{"role": "assistant", "content": "a corrected, non-looping response"}],
                    rejected=[{"role": "assistant", "content": "the repeated failing action"}],
                )
                for cf in classified
            ]

        summary = SelfHealLoop(demo_classifier, demo_generator).run_on(proj)
        return {
            "n_failures": summary["n_failures"],
            "n_classified": summary["n_classified"],
            "n_generated": summary["n_generated"],
            "n_dataset_rows": summary["n_dataset_rows"],
            "sample_row": summary["dataset"][0] if summary["dataset"] else {},
            "status": "demo-stages",
        }

    @app.get("/projects/{pid}/events")
    def events(pid: str) -> dict:
        return {"events": [_event_dict(e) for e in _get(pid).events()]}

    @app.get("/projects/{pid}/trajectories")
    def trajectories(pid: str) -> dict:
        return {"trajectories": [_traj_summary(l) for l in _get(pid).trajectories]}

    @app.websocket("/projects/{pid}/events/ws")
    async def events_ws(websocket: WebSocket, pid: str) -> None:
        proj = projects.get(pid)
        if proj is None:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        sent = 0
        try:
            while True:
                evs = proj.events()
                while sent < len(evs):
                    await websocket.send_json(_event_dict(evs[sent]))
                    sent += 1
                # Poll for a client message instead of a blind sleep, so an *idle* client
                # disconnect is noticed promptly (rather than only on the next send) and the
                # handler exits — otherwise disconnected-but-quiet clients leak until shutdown.
                try:
                    msg = await asyncio.wait_for(websocket.receive(), timeout=0.03)
                except TimeoutError:
                    continue
                # This stream is server->client only; any inbound frame other than a disconnect
                # is consumed and discarded. If this endpoint ever becomes bidirectional, handle
                # those messages here rather than dropping them.
                if msg.get("type") == "websocket.disconnect":
                    return
        except (WebSocketDisconnect, asyncio.CancelledError):
            # client hung up, or the server is shutting down and cancelled this task —
            # both are normal terminations of the stream, not errors.
            return

    return app
