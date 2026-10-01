"""
Safety demo: Remote OpenEnv sandbox vs local RunBashTool
=========================================================

This script demonstrates AgentTune's Layer 1 OpenEnv tool adapter by showing
the **security difference** between running model-generated code:

  A) Locally via RunBashTool (model code executes in the training process)
  B) Remotely via OpenEnvTool (model code executes in an isolated container)

The demo uses the echo_env for the integration test and a conceptual coding_env
example for the safety narrative.  An actual coding_env server is not required
to see the echo_env part; the coding_env section is shown as commented-out
config with explanatory notes.

Prerequisites:
    # openenv is a base dependency of agenttune (pip install -e .) — nothing extra to install.

    # In a separate terminal, start the echo environment:
    cd /tmp/OpenEnv/envs/echo_env
    uvicorn server.app:app --host 0.0.0.0 --port 8765

Run:
    python examples/grpo_with_openenv_sandbox.py

    # For live end-to-end with a real model (uses Qwen3-0.6B-FP8):
    AGENTTUNE_LIVE_TEST=1 python examples/grpo_with_openenv_sandbox.py
"""

from __future__ import annotations

import json
import logging
import os
import textwrap

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("openenv_demo")


# ---------------------------------------------------------------------------
# Part 1: Security comparison — local vs remote execution
# ---------------------------------------------------------------------------


def demo_security_comparison():
    """Show what happens when a model calls os.listdir('/') in each mode."""
    print("\n" + "=" * 70)
    print("PART 1: Security comparison — local vs remote code execution")
    print("=" * 70)

    malicious_like_code = "import os; print(sorted(os.listdir('/')))"

    # ---- A: Local RunBashTool ----
    print("\n[A] Local RunBashTool — code runs IN the training process:")
    from agenttune.agentic.tools.builtin.code_tools import RunPythonTool

    local_tool = RunPythonTool()
    local_result = local_tool.execute(code=malicious_like_code)
    print(f"    success={local_result.success}")
    print(f"    output (first 200 chars): {str(local_result.output)[:200]!r}")
    print("    ⚠️  Model-generated code read the HOST filesystem.")

    # ---- B: OpenEnvTool (echo_env demo) ----
    print("\n[B] OpenEnvTool — code would run in an ISOLATED container:")
    print("    (Using echo_env as a proxy since it's lightweight to start)")
    print()
    print(
        textwrap.dedent(
            """\
        In a production setup you'd use the coding_env or repl_env:

          from agenttune.agentic.tools.builtin.openenv_tool import create_openenv_tools

          tools, handle = create_openenv_tools(
              base_url="http://localhost:8000"  # coding_env server
          )
          # The rollout loop calls: tool.execute(code="import os; ...")
          # → that code runs inside the Docker container, NOT in your training process
          # → os.listdir('/') sees the container's /tmp, not your host's secrets

          # Security delta vs RunBashTool:
          #   RunBashTool: model can read model weights, env vars, SSH keys, etc.
          #   OpenEnvTool: model is confined to the container's filesystem
    """
        )
    )


# ---------------------------------------------------------------------------
# Part 2: Echo env live demo
# ---------------------------------------------------------------------------


def demo_echo_env(base_url: str = "http://localhost:8765"):
    """
    Connect to a running echo_env and run tool calls through OpenEnvTool.
    Shows the full path: schema introspection → execute() → ToolResult.
    """
    print("\n" + "=" * 70)
    print("PART 2: Live echo_env round-trip via OpenEnvTool")
    print("=" * 70)

    try:
        from agenttune.utils.optional import OPENENV_AVAILABLE

        if not OPENENV_AVAILABLE:
            print("  ⚠️  openenv not installed. Run: pip install 'agenttune[openenv]'")
            return
    except ImportError:
        print("  ⚠️  agenttune.utils.optional not found.")
        return

    from agenttune.agentic.tools.builtin.openenv_tool import create_openenv_tools

    print(f"\n  Connecting to {base_url} …")
    try:
        tools, handle = create_openenv_tools(base_url=base_url)
    except Exception as exc:
        print(f"  ✗  Could not connect: {exc}")
        print(
            "  Start echo_env with:\n"
            "    cd /tmp/OpenEnv/envs/echo_env && "
            "uvicorn server.app:app --port 8765"
        )
        return

    print(f"  ✓  Connected. Tools: {[t.name for t in tools]}")

    try:
        # Show schema introspection
        for tool in tools:
            schema = tool.to_schema()
            print(f"\n  Tool: {tool.name}")
            print(f"    description: {tool.description}")
            print(f"    schema: {json.dumps(schema['function']['parameters'], indent=6)}")

        # echo_message
        echo = next((t for t in tools if "echo_message" == t.name), None)
        if echo:
            print("\n  Calling echo_message(message='hello from agenttune') …")
            result = echo.execute(message="hello from agenttune")
            print(f"    success={result.success}")
            print(f"    output={result.output!r}")
            print(f"    latency={result.metadata.get('latency_ms', 0):.1f}ms")
            assert (
                result.success and result.output == "hello from agenttune"
            ), "echo_message did not return the expected value"
            print("    ✓  Assertion passed")

        # echo_with_length
        echo_len = next((t for t in tools if "echo_with_length" == t.name), None)
        if echo_len:
            print("\n  Calling echo_with_length(message='test') …")
            result = echo_len.execute(message="test")
            print(f"    success={result.success}")
            print(f"    output={result.output!r}")
            if result.success:
                assert (
                    result.output.get("length") == 4
                ), "echo_with_length returned unexpected length"
                print("    ✓  Assertion passed")

        # Error path demo
        print("\n  Calling echo_message with wrong arg (error path) …")
        result = echo.execute(bad_arg="should fail")
        print(f"    success={result.success}")
        print(
            f"    output keys: {list(result.output.keys()) if isinstance(result.output, dict) else type(result.output).__name__}"
        )
        if not result.success:
            print("    ✓  Error correctly surfaced as ToolResult(success=False)")

    finally:
        handle.close()
        print("\n  ✓  Handle closed. WebSocket session released.")


# ---------------------------------------------------------------------------
# Part 3: YAML config demo (shows how TrainerConfigBridge wires it in)
# ---------------------------------------------------------------------------


def demo_yaml_config():
    """Show the YAML config pattern for using OpenEnv tools via TrainerConfigBridge."""
    print("\n" + "=" * 70)
    print("PART 3: YAML config — wiring OpenEnv into a training run")
    print("=" * 70)

    example_yaml = textwrap.dedent(
        """\
        # trainer_config.yaml
        training:
          algorithm: grpo
          model: Qwen/Qwen3-0.6B-FP8

          rollout:
            backend: transformers
            max_steps: 3
            system_prompt: "You are an agent that can call remote tools."

          # Mix of local and remote tools
          tools:
            - type: builtin_file          # local, runs in trainer process
            - type: openenv               # remote, runs in Docker container
              base_url: "http://localhost:8000"
              name_prefix: "sandbox_"     # avoids name collision with builtins
              connect_timeout_s: 10
              message_timeout_s: 60
              # tool_filter: ["run_code"]  # optional: only wrap specific tools

          reward_funcs:
            - correctness_reward
    """
    )

    print("\nExample YAML config (trainer_config.yaml):\n")
    print(textwrap.indent(example_yaml, "  "))

    example_python = textwrap.dedent(
        """\
        from agenttune.decide.trainer_config_bridge import TrainerConfigBridge

        bridge = TrainerConfigBridge("trainer_config.yaml")
        trainer = bridge.build_trainer(train_dataset=my_dataset)

        try:
            trainer.train()
        finally:
            bridge.close_openenv_handles()   # release WebSocket session
    """
    )

    print("Python usage:\n")
    print(textwrap.indent(example_python, "  "))


# ---------------------------------------------------------------------------
# Part 4: Live end-to-end test with a real model (opt-in)
# ---------------------------------------------------------------------------


def demo_live_rollout(base_url: str = "http://localhost:8765"):
    """
    Run an actual rollout using Qwen3-0.6B-FP8 with an OpenEnvTool.
    Only runs when AGENTTUNE_LIVE_TEST=1 is set.
    """
    print("\n" + "=" * 70)
    print("PART 4: Live rollout test (Qwen/Qwen3-0.6B-FP8 + OpenEnvTool)")
    print("=" * 70)

    if not os.environ.get("AGENTTUNE_LIVE_TEST"):
        print(
            "  Skipped. Set AGENTTUNE_LIVE_TEST=1 to run this section.\n"
            "  Requires GPU and a running echo_env at OPENENV_ECHO_URL "
            "(default: http://localhost:8765)."
        )
        return

    from agenttune.utils.optional import OPENENV_AVAILABLE

    if not OPENENV_AVAILABLE:
        print("  ⚠️  openenv not installed.")
        return

    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
    from agenttune.agentic.tools.builtin.openenv_tool import create_openenv_tools

    model_name = os.environ.get("AGENTTUNE_TEST_MODEL", "Qwen/Qwen3-0.6B-FP8")

    print(f"\n  Model: {model_name}")
    print(f"  echo_env: {base_url}")

    # Build tools from running echo_env
    try:
        tools, handle = create_openenv_tools(base_url=base_url)
    except Exception as exc:
        print(f"  ✗  Could not connect to echo_env: {exc}")
        return

    try:
        print(f"  Remote tools: {[t.name for t in tools]}")

        # Build rollout function
        rollout_fn = create_rollout_fn(
            rollout_engine=None,  # will auto-create transformers engine
            tools=tools,
            max_steps=2,
            system_prompt=(
                "You are a helpful assistant. Use the echo_message tool to "
                "repeat the user's message back to them."
            ),
            model_path=model_name,
        )

        prompts = [
            "Please echo back: hello world",
            "Please echo back: testing remote tool",
        ]

        print(f"\n  Running rollout on {len(prompts)} prompts …")
        outputs = rollout_fn(prompts)
        print(f"  ✓  Rollout complete. Received {len(outputs)} completions.")

        for i, (prompt, output) in enumerate(zip(prompts, outputs, strict=False)):
            print(f"\n  Prompt {i+1}: {prompt!r}")
            print(f"  Output (truncated): {str(output)[:200]!r}")

    finally:
        handle.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    echo_url = os.environ.get("OPENENV_ECHO_URL", "http://localhost:8765")

    print("\nAgentTune × OpenEnv — Layer 1 Safety Demo")
    print("==========================================")

    demo_security_comparison()
    demo_echo_env(base_url=echo_url)
    demo_yaml_config()
    demo_live_rollout(base_url=echo_url)

    print("\n✓  Demo complete.\n")
