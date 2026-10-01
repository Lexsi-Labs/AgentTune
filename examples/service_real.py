"""Stand the FastAPI service up as a *real* server and drive it over a real network.

The service (`agenttune.agentic.service.app.create_app`) is the operator control surface for
the spine. Its unit tests only ever exercise it through `fastapi.testclient.TestClient`, which
calls the ASGI app in-process — the app is never bound to a socket, the operator UI is never
served, and no real HTTP/WebSocket client ever connects. This example closes exactly that gap.

What is real here:
  * a real `uvicorn` server bound to a real TCP port, started and cleanly torn down;
  * **every** endpoint driven over real HTTP with an `httpx` client (no TestClient);
  * a real `websockets` client connected to `/projects/{pid}/events/ws`, observing the
    lifecycle events stream *live* as the REST calls drive the project forward;
  * the real operator UI (`static/index.html`) fetched over HTTP and asserted on.

Honest scope — read this before reading the output. The *server, transport (REST + WebSocket),
and operator UI* are what was never proven and are now proven real. The *endpoints* run the
library's documented GPU-free control path: real spine machinery (Project lifecycle, real
`agentic_metrics`, real `run_episode`/`ReActStrategy`, real `VectorMemory`, real `FailureDetector`
+ `SelfHealLoop`) driven by model-free policies — exactly what `service/app.py`'s docstring
claims, no more. The real *model* execution these controls represent is the rest of this
`examples/` suite (the nine trainers + `agentic_strategy_real.py` + `self_heal_llm_real.py`).
This example does not silently upgrade the demo backend to run a model; it proves the server is
a real server. If you want the endpoints themselves model-backed, that is a separate, larger
service-integration piece.

Requires: uvicorn, httpx, websockets (all base deps of the service extra). No GPU, no network.

    python examples/service_real.py
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import httpx
import uvicorn
import websockets

from agenttune.agentic.service.app import create_app


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_server(app, host: str, port: int):
    """Start uvicorn in a background thread; return (server, thread) once it is accepting."""
    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=3
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):  # up to ~10s
        if server.started:
            break
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("uvicorn did not start")
    return server, thread


async def _watch_events(url: str, stop: asyncio.Event) -> list[dict]:
    """A real WebSocket client: connect and collect streamed lifecycle events until told to stop."""
    seen: list[dict] = []
    async with websockets.connect(url) as ws:
        while not stop.is_set():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except TimeoutError:
                continue
            seen.append(msg)
    return seen


def main() -> None:
    host, port = "127.0.0.1", _free_port()
    base = f"http://{host}:{port}"
    ws_base = f"ws://{host}:{port}"

    server, thread = _start_server(create_app(), host, port)
    print(f"[server]  uvicorn up on {base}  (server.started={server.started})")

    try:
        with httpx.Client(base_url=base, timeout=10.0) as http:
            # 1. the operator UI, served over real HTTP
            r = http.get("/")
            ui_ok = r.status_code == 200 and "text/html" in r.headers.get("content-type", "")
            has_shell = "AgentTune" in r.text and "<html" in r.text.lower()
            print(
                f"[ui]      GET /  ->  {r.status_code}  ({len(r.text)} bytes, "
                f"html={ui_ok}, operator-shell={has_shell})"
            )

            # 2. create a project
            pid = http.post("/projects").json()["project_id"]
            print(f"[project] POST /projects  ->  {pid}")

            # Connect the real WebSocket client BEFORE driving the lifecycle, so it observes
            # the events live. Run the REST calls from a worker thread while the asyncio loop
            # owns the socket.
            stop = asyncio.Event()

            async def drive_and_watch() -> tuple[list[dict], dict]:
                loop = asyncio.get_running_loop()
                watcher = asyncio.create_task(
                    _watch_events(f"{ws_base}/projects/{pid}/events/ws", stop)
                )
                await asyncio.sleep(0.2)  # let the client connect

                rest_results: dict = {}

                def rest_calls() -> None:
                    # 3. collect rollouts (real DemoRolloutEngine through Project.collect_rollout)
                    rest_results["collect"] = http.post(
                        f"/projects/{pid}/collect_rollout",
                        json={"tasks": ["ticket-1", "ticket-2", "ticket-3"], "max_steps": 4},
                    ).json()
                    # 4. evaluate (real agentic_metrics over the collected trajectories)
                    rest_results["evaluate"] = http.post(f"/projects/{pid}/evaluate").json()
                    # 5. distill dataset preview (real Project.sft_dataset)
                    rest_results["distill"] = http.post(
                        f"/projects/{pid}/distill", json={"student": "student-1.5B"}
                    ).json()
                    # 6. train wiring (reports wired-runs-on-gpu, as the library does)
                    rest_results["train"] = http.post(
                        f"/projects/{pid}/train", json={"fmt": "grpo"}
                    ).json()
                    # 7. run every one of the five agent designs (real run_episode / strategies)
                    rest_results["strategies"] = {
                        s: http.post(f"/projects/{pid}/run_strategy", json={"strategy": s}).json()
                        for s in ("react", "plan_execute", "reflexion", "tot", "memory")
                    }
                    # 8. memory recall — real VectorMemory and real GraphMemory
                    rest_results["memory"] = {
                        d: http.post(f"/projects/{pid}/memory_demo", json={"driver": d}).json()
                        for d in ("vector", "graph")
                    }
                    # 9. seed a looping failure, then detect + self-heal loop (real detector)
                    rest_results["seed"] = http.post(f"/projects/{pid}/seed_demo_failure").json()
                    rest_results["heal"] = http.post(f"/projects/{pid}/heal").json()
                    rest_results["heal_loop"] = http.post(f"/projects/{pid}/heal_loop").json()
                    # 10. read back the event log and trajectories over HTTP
                    rest_results["events"] = http.get(f"/projects/{pid}/events").json()
                    rest_results["trajectories"] = http.get(f"/projects/{pid}/trajectories").json()

                await loop.run_in_executor(None, rest_calls)
                await asyncio.sleep(0.3)  # let the stream drain to the ws client
                stop.set()
                streamed = await watcher
                return streamed, rest_results

            streamed, results = asyncio.run(drive_and_watch())

        # ---- report ----
        col = results["collect"]
        print(f"[collect] POST /collect_rollout  ->  {col['n_trajectories']} trajectories")
        ev = results["evaluate"]
        metrics = ", ".join(f"{k}={v:.2f}" for k, v in ev["metrics"].items())
        print(f"[eval]    POST /evaluate  ->  n={ev['n']}  metrics: {metrics}")
        di = results["distill"]
        print(f"[distill] POST /distill  ->  {di['n_rows']} rows, status={di['status']!r}")
        tr = results["train"]
        print(f"[train]   POST /train  ->  status={tr['status']!r}")
        for s, res in results["strategies"].items():
            print(
                f"[strat]   run_strategy({s:<12}) -> answer={res['answer']!r} "
                f"({res['n_events']} events)"
            )
        for d, res in results["memory"].items():
            print(f"[memory]  memory_demo({d:<6}) -> recalled {res['recalled']}")
        hl = results["heal"]
        print(
            f"[heal]    POST /heal  ->  {len(hl['failures'])} failure(s): "
            f"{[f['failure_type'] for f in hl['failures']]}"
        )
        loop_res = results["heal_loop"]
        print(
            f"[heal+]   POST /heal_loop  ->  classified={loop_res['n_classified']} "
            f"generated={loop_res['n_generated']} rows={loop_res['n_dataset_rows']} "
            f"status={loop_res['status']!r}"
        )
        n_events = len(results["events"]["events"])
        n_traj = len(results["trajectories"]["trajectories"])
        print(f"[read]    GET /events -> {n_events} events   GET /trajectories -> {n_traj}")
        print(
            f"[ws]      real websockets client streamed {len(streamed)} live events "
            f"while the REST calls ran"
        )

        ok = (
            ui_ok
            and has_shell
            and n_events > 0
            and len(streamed) > 0
            and all(r["answer"] == "42" for r in results["strategies"].values())
        )
        print(
            f"[verdict] server real, UI served, {n_events} events over REST and "
            f"{len(streamed)} over WebSocket, all 5 strategies answered: {ok}"
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        alive = thread.is_alive()
        print(f"[server]  shutdown requested, server thread stopped={not alive}")


if __name__ == "__main__":
    main()
