"""
agenttune.agentic.scenarios
===========================
Generic scenario generation for any tool-using agent.

Works with:
  - Plain Python callables  (def my_tool(x: int) -> str: ...)
  - BaseTool subclasses     (agenttune.agentic.tools.base.BaseTool)
  - Raw dicts               ({"name": ..., "description": ..., "parameters": ...})
  - MCP MCPTool objects     (art.mcp.types.MCPTool) — optional, no hard dep
  - MCP MCPResource objects (art.mcp.types.MCPResource) — optional, no hard dep

Generation backends (first match wins):
  1. Pre-built RolloutEngine  ->  vLLM / Transformers (local)
  2. rollout_backend kwarg    ->  builds engine via create_rollout_engine
  3. generator_model="claude-*"  ->  Anthropic SDK
  4. default                  ->  OpenAI-compatible (OpenAI, OpenRouter, custom URL)

Placement in project
--------------------
    src/agenttune/agentic/scenarios/   <- this package
        __init__.py
        normalizer.py
        collection.py
        generator.py

Usage
-----
    from agenttune.agentic.scenarios import generate_scenarios

    # plain functions
    scenarios = generate_scenarios([search, run_code], num_scenarios=12)

    # reuse a vLLM engine from training
    scenarios = generate_scenarios(tools, rollout_engine=my_engine)

    # get task strings ready for rollout_fn
    prompts = scenarios.filter_by_difficulty(min_difficulty=3).tasks()
"""

from .collection import Scenario, ScenarioCollection
from .generator import ScenarioGeneratorConfig, generate_scenarios
from .normalizer import ResourceInfo, ToolInfo, normalize_resources, normalize_tools

__all__ = [
    "generate_scenarios",
    "ScenarioGeneratorConfig",
    "normalize_tools",
    "normalize_resources",
    "ToolInfo",
    "ResourceInfo",
    "ScenarioCollection",
    "Scenario",
]
