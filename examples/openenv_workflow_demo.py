import atexit
import json
import logging
import os
import socket
import subprocess
import sys
import time

logging.basicConfig(level=logging.WARNING)

# ── Self-contained echo_env server bootstrap ────────────────────────────────
# create_openenv_tools() below needs a live OpenEnv server on SERVER_URL. This
# clones OpenEnv (same layout as tests/test_openenv_tool.py's documented setup:
# `cd /tmp/OpenEnv/envs/echo_env && uvicorn server.app:app --port 8765`) and
# starts its echo_env server if one isn't already running, so this script is
# runnable standalone: `python examples/openenv_workflow_demo.py`.
OPENENV_REPO = "/tmp/OpenEnv"
ECHO_ENV_DIR = os.path.join(OPENENV_REPO, "envs", "echo_env")
SERVER_HOST = "localhost"
SERVER_PORT = 8765
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def _port_open(host, port, timeout=0.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ensure_echo_env_server():
    """Start a local echo_env server if one isn't already listening on SERVER_PORT.

    Returns the Popen handle if we started it (caller should stop it on exit),
    or None if a server was already running (leave it alone).
    """
    if _port_open(SERVER_HOST, SERVER_PORT):
        print(f"[+] echo_env server already running on {SERVER_URL}")
        return None

    if not os.path.isdir(ECHO_ENV_DIR):
        print(f"[+] OpenEnv not found locally -- cloning into {OPENENV_REPO} ...")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/meta-pytorch/OpenEnv.git",
                OPENENV_REPO,
            ],
            check=True,
        )

    print(f"[+] Starting echo_env server on {SERVER_URL} ...")
    log_path = "/tmp/agenttune-echo-env-server.log"
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(SERVER_PORT),
        ],
        cwd=ECHO_ENV_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    for _ in range(60):  # ~30s
        if _port_open(SERVER_HOST, SERVER_PORT):
            print("[+] echo_env server is up.")
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"echo_env server exited before becoming ready -- see {log_path}")
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"Timed out waiting for echo_env server to start -- see {log_path}")


_server_proc = _ensure_echo_env_server()
if _server_proc is not None:

    def _stop_echo_env_server():
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()

    atexit.register(_stop_echo_env_server)


from agenttune.agentic.tools.builtin.openenv_tool import create_openenv_tools

print("=" * 60)
print("AgentTune x OpenEnv -- Live Tool Workflow Demo")
print("=" * 60)

tools, handle = create_openenv_tools(
    base_url=SERVER_URL,
    name_prefix="sandbox_",
)

print(f"\n[+] Connected. Tools: {[t.name for t in tools]}")

for tool in tools:
    schema = tool.to_schema()["function"]
    print(f"\n  Tool      : {tool.name}")
    print(f"  Remote    : {tool._remote_name}")
    print(f"  Desc      : {tool.description}")
    print(f"  Schema    : {json.dumps(schema['parameters']['properties'], indent=12)}")

print("\n" + "-" * 60)
print("WORKFLOW: input -> output for each call type")
print("-" * 60)

echo = next(t for t in tools if "echo_message" in t.name)
echo_len = next(t for t in tools if "echo_with_length" in t.name)

# Call 1: success path
input_1 = {"message": "hello from agenttune rollout"}
t0 = time.monotonic()
r1 = echo.execute(**input_1)
ms1 = (time.monotonic() - t0) * 1000

print("\n[Call 1] echo_message -- success path")
print(f"  input   : {json.dumps(input_1)}")
print(f"  output  : {json.dumps(r1.output)}")
print(f"  success : {r1.success}")
print(f"  latency : {ms1:.1f} ms")
print(f"  metadata: {r1.metadata}")

# Call 2: structured dict result
input_2 = {"message": "agenttune"}
t0 = time.monotonic()
r2 = echo_len.execute(**input_2)
ms2 = (time.monotonic() - t0) * 1000

print("\n[Call 2] echo_with_length -- structured dict output")
print(f"  input   : {json.dumps(input_2)}")
print(f"  output  : {json.dumps(r2.output)}")
print(f"  success : {r2.success}")
print(f"  latency : {ms2:.1f} ms")

# Call 3: error path
input_3 = {"wrong_param": "this should fail"}
t0 = time.monotonic()
r3 = echo.execute(**input_3)
ms3 = (time.monotonic() - t0) * 1000

print("\n[Call 3] echo_message -- error path (wrong param)")
print(f"  input   : {json.dumps(input_3)}")
print(f"  output  : {json.dumps(r3.output)}")
print(f"  success : {r3.success}")
print(f"  error   : {r3.error[:100]}")
print(f"  latency : {ms3:.1f} ms")
print(f"  failure counter trips: {isinstance(r3.output, dict) and 'error' in r3.output}")

print("\n" + "-" * 60)
print("Contract:")
print("  success=True  -> output = raw result (str / dict / list)")
print("  success=False -> output = {'error': '<msg>'}  (triggers rollout failure counter)")
print("-" * 60)

handle.close()
print("\n[+] Handle closed.")
