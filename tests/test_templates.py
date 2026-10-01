"""Tests for template validation and execution."""

from pathlib import Path

import pytest

from agenttune.decide.config import ConfigLoader
from agenttune.decide.registry import TemplateRegistry


class TestTemplates:
    """Test cases for all 18 templates."""

    @pytest.fixture(scope="session")
    def templates_dir(self):
        """Get templates directory."""
        return Path(__file__).parent.parent / "src" / "agenttune" / "decide" / "templates"

    @pytest.fixture(scope="session")
    def registry(self, templates_dir):
        """Create registry and discover templates."""
        reg = TemplateRegistry()
        reg.discover(str(templates_dir))
        return reg

    def test_all_templates_load(self, registry, templates_dir):
        """Test all 18 templates load successfully."""
        templates = registry.list_all()

        # Should have at least the core templates
        assert len(templates) >= 1

        # Try to load each template
        for template in templates:
            template_id = template["id"]
            try:
                config = ConfigLoader.load(template_id, None)
                assert config is not None
                assert "id" in config
                assert "stages" in config
            except Exception as e:
                pytest.fail(f"Template {template_id} failed to load: {str(e)}")

    def test_no_schema_errors(self, registry):
        """Test templates have no schema errors."""
        templates = registry.list_all()

        for template in templates:
            template_id = template["id"]
            config = ConfigLoader.load(template_id, None)

            # Validate required fields
            assert "id" in config, f"{template_id}: missing 'id'"
            assert "name" in config, f"{template_id}: missing 'name'"
            assert "version" in config, f"{template_id}: missing 'version'"
            assert "stages" in config, f"{template_id}: missing 'stages'"

            # Validate stages
            stages = config.get("stages", [])
            assert len(stages) > 0, f"{template_id}: no stages"

            for idx, stage in enumerate(stages):
                assert "id" in stage, f"{template_id}: stage {idx} missing 'id'"
                assert "type" in stage, f"{template_id}: stage {idx} missing 'type'"

                # Validate stage type
                valid_types = [
                    "llm_call",
                    "llm_judge",
                    "rules",
                    "parallel",
                    "router",
                    "human_review",
                    "tool_call",
                    "output",
                ]
                assert (
                    stage["type"] in valid_types
                ), f"{template_id}: stage {stage['id']} has invalid type '{stage['type']}'"

    def test_no_orphaned_stages(self, registry):
        """Test templates have no unreachable stages."""
        templates = registry.list_all()

        for template in templates:
            template_id = template["id"]
            config = ConfigLoader.load(template_id, None)

            stages = config.get("stages", [])
            stage_ids = {stage["id"] for stage in stages}

            # Build reachability map from edges
            edges = config.get("edges", [])
            reachable = set()

            # Start from first stage (entry point)
            if stages:
                first_stage = stages[0]["id"]
                reachable.add(first_stage)

                # BFS to find all reachable stages
                queue = [first_stage]
                while queue:
                    current = queue.pop(0)
                    for edge in edges:
                        if edge.get("from") == current:
                            next_stage = edge.get("to")
                            if next_stage and next_stage not in ("__end__", "__start__"):
                                if next_stage not in reachable:
                                    reachable.add(next_stage)
                                    queue.append(next_stage)

                # Check for orphaned stages
                orphaned = stage_ids - reachable
                assert not orphaned, f"{template_id}: orphaned stages {orphaned}"

    def test_no_circular_edges(self, registry):
        """Test templates have no circular edges (except intentional iteration loops)."""
        templates = registry.list_all()

        for template in templates:
            template_id = template["id"]
            config = ConfigLoader.load(template_id, None)

            edges = config.get("edges", [])
            stages = {s["id"]: s for s in config.get("stages", [])}

            # Build adjacency graph excluding conditional loop-back edges
            graph = {}
            for edge in edges:
                from_stage = edge.get("from")
                to_stage = edge.get("to")

                if from_stage and to_stage not in ("__end__", "__start__"):
                    # Skip edges that are part of iteration loops (conditional routes back to same or earlier stage)
                    # These are intentional for refinement patterns
                    stage = stages.get(from_stage, {})

                    # Check if this edge is part of a loop that's intentional for iteration
                    is_iteration_loop = False

                    # 1. Check on_result conditions with max_iterations
                    on_result = stage.get("on_result", [])
                    if on_result:
                        for condition_entry in on_result:
                            if condition_entry.get("goto") == to_stage:
                                # Allow if source or target has max_iterations > 1
                                target_stage = stages.get(to_stage, {})
                                if (
                                    stage.get("max_iterations", 1) > 1
                                    or target_stage.get("max_iterations", 1) > 1
                                ):
                                    is_iteration_loop = True
                                    break

                    # 2. Check on_fail/on_failure routing (rules refinement loops)
                    if not is_iteration_loop:
                        rules = stage.get("rules", [])
                        if isinstance(rules, list):
                            for rule in rules:
                                if isinstance(rule, dict):
                                    on_fail = rule.get("on_fail") or rule.get("on_failure")
                                    if on_fail:
                                        target = (
                                            on_fail.get("goto")
                                            if isinstance(on_fail, dict)
                                            else on_fail
                                        )
                                        if target == to_stage:
                                            is_iteration_loop = True
                                            break

                    # 3. Check on_parse_error (extraction refinement loops)
                    if not is_iteration_loop and stage.get("on_parse_error") == to_stage:
                        is_iteration_loop = True

                    if not is_iteration_loop:
                        if from_stage not in graph:
                            graph[from_stage] = []
                        graph[from_stage].append(to_stage)

            # DFS to detect unconditional cycles
            def has_cycle(start, visited, rec_stack):
                visited.add(start)
                rec_stack.add(start)

                for neighbor in graph.get(start, []):  # noqa: B023
                    if neighbor not in visited:
                        if has_cycle(neighbor, visited, rec_stack):
                            return True
                    elif neighbor in rec_stack:
                        return True

                rec_stack.remove(start)
                return False

            # Check all nodes
            visited = set()
            for node in graph:
                if node not in visited:
                    rec_stack = set()
                    if has_cycle(node, visited, rec_stack):
                        pytest.fail(f"{template_id}: contains circular dependency")
