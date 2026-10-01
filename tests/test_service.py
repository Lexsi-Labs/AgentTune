"""Phase 8 — the service layer over Project (the EMNLP demo backend).

A FastAPI app exposing the GPU-free live spine path (build -> collect -> evaluate ->
distill-dataset -> heal) plus a live lifecycle event stream. GRPO training and full
self-heal are represented as 'wired, runs on GPU', not faked here. Tested with the
FastAPI TestClient — no server, no GPU.
"""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from agenttune.agentic.service.app import create_app


@pytest.fixture()
def client():
    return TestClient(create_app())


def test_index_serves_the_demo_frontend(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "AgentTune" in r.text and "/events/ws" in r.text
    assert "heal_loop" in r.text  # the full closed-loop control is wired in the UI
    assert "run_strategy" in r.text and "memory_demo" in r.text  # strategy + memory pickers


def test_create_project_and_collect_rollout(client):
    pid = client.post("/projects").json()["project_id"]
    r = client.post(
        f"/projects/{pid}/collect_rollout", json={"tasks": ["what is 6x7?", "capital of France?"]}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n_trajectories"] == 2
    assert all(t["tier"] == "full" for t in body["trajectories"])


def test_full_gpu_free_lifecycle_over_http(client):
    pid = client.post("/projects").json()["project_id"]
    client.post(f"/projects/{pid}/collect_rollout", json={"tasks": ["q1", "q2"]})

    ev = client.post(f"/projects/{pid}/evaluate").json()
    assert ev["n"] == 2 and "arr" in ev["metrics"]

    ds = client.post(f"/projects/{pid}/distill", json={"student": "student-1.5B"}).json()
    assert ds["n_rows"] == 2 and ds["status"] == "wired-runs-on-gpu"
    assert "messages" in ds["sample_row"]

    heal = client.post(f"/projects/{pid}/heal").json()
    assert "failures" in heal

    events = client.get(f"/projects/{pid}/events").json()["events"]
    stages = {e["stage"] for e in events}
    assert {"collect_rollout", "evaluate_agentic", "distill", "heal"} <= stages


def test_grpo_is_reported_as_wired_not_run(client):
    pid = client.post("/projects").json()["project_id"]
    r = client.post(f"/projects/{pid}/train", json={"fmt": "grpo"})
    assert r.status_code == 200
    assert r.json()["status"] == "wired-runs-on-gpu"


def test_unknown_project_404(client):
    assert client.post("/projects/nope/evaluate").status_code == 404


def test_heal_loop_demo_surfaces_full_closed_loop(client):
    pid = client.post("/projects").json()["project_id"]
    # seed a stuck (looping) agent so the detector has a real failure to find
    seeded = client.post(f"/projects/{pid}/seed_demo_failure")
    assert seeded.status_code == 200 and seeded.json()["seeded"] is True

    r = client.post(f"/projects/{pid}/heal_loop")
    assert r.status_code == 200
    body = r.json()
    # detect -> classify -> generate -> dataset all reported
    assert body["n_failures"] >= 1
    assert body["n_classified"] == body["n_failures"]
    assert body["n_dataset_rows"] >= 1
    assert "prompt" in body["sample_row"] and "chosen" in body["sample_row"]
    assert body["status"] == "demo-stages"  # honest: litellm stages are demo stand-ins

    # lifecycle stream shows the heal detection ran
    stages = {e["stage"] for e in client.get(f"/projects/{pid}/events").json()["events"]}
    assert "heal" in stages


@pytest.mark.parametrize("strat", ["react", "plan_execute", "reflexion", "tot", "memory"])
def test_run_strategy_demo_runs_each_agent_design(client, strat):
    pid = client.post("/projects").json()["project_id"]
    r = client.post(f"/projects/{pid}/run_strategy", json={"strategy": strat})
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == strat
    assert body["n_events"] > 0
    assert body["answer"] == "42"  # each canned demo reaches the answer
    stages = {e["stage"] for e in client.get(f"/projects/{pid}/events").json()["events"]}
    assert "run_strategy" in stages


def test_run_strategy_unknown_400(client):
    pid = client.post("/projects").json()["project_id"]
    assert (
        client.post(f"/projects/{pid}/run_strategy", json={"strategy": "nope"}).status_code == 400
    )


@pytest.mark.parametrize(
    "driver,query,expect",
    [
        ("vector", "cat", "cat cat"),  # semantic: closest to 'cat'
        ("graph", "Alice", "Bob"),  # relational: Alice's neighbor
    ],
)
def test_memory_demo_recalls_per_driver(client, driver, query, expect):
    pid = client.post("/projects").json()["project_id"]
    r = client.post(f"/projects/{pid}/memory_demo", json={"driver": driver})
    assert r.status_code == 200
    body = r.json()
    assert body["driver"] == driver
    assert expect in body["recalled"]
    assert "Dave" not in body["recalled"] if driver == "graph" else True


def test_event_websocket_streams_lifecycle(client):
    pid = client.post("/projects").json()["project_id"]
    with client.websocket_connect(f"/projects/{pid}/events/ws") as ws:
        client.post(f"/projects/{pid}/collect_rollout", json={"tasks": ["q1"]})
        # drain a couple of frames; at least one collect_rollout event must arrive
        seen = [ws.receive_json()["stage"] for _ in range(2)]
        assert "collect_rollout" in seen
