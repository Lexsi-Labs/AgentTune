"""Configuration loader with deep merge and validation."""

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """Load and merge YAML configuration with template extensions."""

    @staticmethod
    def _load_yaml(file_path: str) -> dict[str, Any]:
        """
        Load a YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            Parsed YAML as dictionary

        Raises:
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path) as f:
            content = yaml.safe_load(f)
            return content or {}

    @staticmethod
    def load(template_id: str, config_path: str | None = None) -> dict[str, Any]:
        """
        Load and merge template with global config.

        Args:
            template_id: Template identifier (e.g., "bfsi/kyc_triage")
            config_path: Path to global config.yaml (optional)

        Returns:
            Merged configuration dictionary

        Raises:
            FileNotFoundError: If config or template file not found
            ValueError: If configuration validation fails
        """
        # Load global config
        global_config = {}
        if config_path:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path) as f:
                global_config = yaml.safe_load(f) or {}

        # Resolve template path
        # Handle both "bfsi/kyc_triage" and full paths
        module_dir = Path(__file__).parent
        (module_dir / "templates").resolve()

        if "/" in template_id and not template_id.startswith("/"):
            # It's a relative path like "bfsi/kyc_triage"
            template_path = module_dir / "templates" / f"{template_id}.yaml"
        else:
            template_path = Path(template_id)

        # Guard against path traversal: reject paths with .. components
        if ".." in str(template_path):
            raise ValueError(
                f"Template path traversal detected: '{template_id}' contains '..' path components."
            )

        template_path.resolve()

        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        # Load template
        with open(template_path) as f:
            template_config = yaml.safe_load(f) or {}

        # Handle extends field recursively
        if "extends" in template_config:
            extends_path = Path(template_config["extends"])
            if not extends_path.is_absolute():
                # Resolve relative to template directory
                extends_path = template_path.parent / extends_path

            if extends_path.exists():
                with open(extends_path) as f:
                    extends_config = yaml.safe_load(f) or {}
                # Merge extends into global_config first
                global_config = ConfigLoader.deep_merge(extends_config, global_config)

        # Merge template into global config
        merged = ConfigLoader.deep_merge(global_config, template_config)

        # Validate
        ConfigLoader._validate(merged)

        # Generate edges from stage routing
        ConfigLoader._generate_edges(merged)

        return merged

    @staticmethod
    def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """
        Deep merge override dictionary into base dictionary.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration with override values taking precedence
        """
        result = base.copy()
        result.pop("extends", None)  # Remove extends from base if present

        for key, override_value in override.items():
            if key == "extends":
                continue  # Skip extends after loading

            if key not in result:
                result[key] = override_value
            elif isinstance(result[key], dict) and isinstance(override_value, dict):
                # Recursively merge nested dicts
                result[key] = ConfigLoader.deep_merge(result[key], override_value)
            elif isinstance(result[key], list) and isinstance(override_value, list):
                # Lists replace entirely
                result[key] = override_value
            else:
                # Scalars: override wins
                result[key] = override_value

        return result

    @staticmethod
    def _validate(config: dict[str, Any]) -> None:
        """
        Validate configuration against schema.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Check required top-level fields
        if "stages" not in config or not config["stages"]:
            raise ValueError("Template must define at least one stage")

        # Check each stage FIRST (before checking name/version) to provide better error messages
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

        for i, stage in enumerate(config["stages"]):
            if not isinstance(stage, dict):
                raise ValueError(f"Stage {i} is not a dictionary")

            if "id" not in stage or "type" not in stage:
                raise ValueError(f"Stage {i} missing id or type")

            if stage["type"] not in valid_types:
                raise ValueError(f"Unknown stage type: {stage['type']}")

        # Then check template-level metadata
        if "id" not in config:
            raise ValueError("Template must have an id")

        if "name" not in config:
            raise ValueError("Template must have a name")

        if "version" not in config:
            raise ValueError("Template must have a version")

    @staticmethod
    def _generate_edges(config: dict[str, Any]) -> None:
        """
        Generate edges from stage routing information.

        Creates edges list from 'next' field (deterministic), on_result conditions,
        and on_fail/on_failure fields (conditional routing).
        For stages without explicit routing, creates edges to the next stage in sequence
        (unless it's an output stage).
        Modifies config in-place.

        Args:
            config: Configuration dictionary to augment with edges
        """
        edges = []
        stages = config.get("stages", [])
        stage_ids = {s.get("id") for s in stages if s.get("id")}
        stage_list = [s for s in stages if s.get("id")]

        # Collect all edges from various routing fields
        for idx, stage in enumerate(stage_list):
            stage_id = stage.get("id")
            if not stage_id:
                continue

            has_explicit_routing = False

            # 1. Edges from explicit 'next' field
            if "next" in stage:
                next_stage = stage["next"]
                if next_stage in stage_ids:
                    edges.append({"from": stage_id, "to": next_stage})
                    has_explicit_routing = True

            # 2. Edges from on_result conditions (router, llm_judge with routing)
            on_result = stage.get("on_result", [])
            if isinstance(on_result, list) and on_result:
                has_explicit_routing = True
                for condition_path in on_result:
                    goto = condition_path.get("goto")
                    if goto and goto in stage_ids:
                        edges.append({"from": stage_id, "to": goto})

            # 2b. Edges from rules on_fail/on_failure (rules stage failure handling)
            rules = stage.get("rules", [])
            if isinstance(rules, list):
                for rule in rules:
                    if isinstance(rule, dict):
                        on_fail = rule.get("on_fail") or rule.get("on_failure")
                        if on_fail:
                            target = on_fail.get("goto") if isinstance(on_fail, dict) else on_fail
                            if target and target in stage_ids:
                                edges.append({"from": stage_id, "to": target})
                                has_explicit_routing = True

            # 3. Edges from on_fail/on_failure (rules failure handling)
            on_fail = stage.get("on_fail") or stage.get("on_failure")
            if on_fail:
                # Handle both string and dict formats
                target = on_fail.get("goto") if isinstance(on_fail, dict) else on_fail
                if target and target in stage_ids:
                    edges.append({"from": stage_id, "to": target})
                    has_explicit_routing = True

            # 4. Router 'default' field (default routing)
            # This is conditional routing, handled by graph_runner
            default = stage.get("default")
            if default:
                has_explicit_routing = True

            # 5. Edges from human_review routing (on_approved, on_denied, on_failed)
            for field in ["on_approved", "on_denied", "on_failed"]:
                target = stage.get(field)
                if target and target in stage_ids:
                    edges.append({"from": stage_id, "to": target})
                    has_explicit_routing = True

            # 6. Auto-route to next stage if no explicit routing (except for output stages)
            if not has_explicit_routing and stage.get("type") != "output":
                if idx + 1 < len(stage_list):
                    next_stage_id = stage_list[idx + 1].get("id")
                    if next_stage_id:
                        edges.append({"from": stage_id, "to": next_stage_id})

        # Merge with any explicitly defined edges (don't overwrite)
        existing_edges = config.get("edges", [])
        if existing_edges:
            edges.extend(existing_edges)

        # Remove duplicate edges
        unique_edges = []
        seen = set()
        for edge in edges:
            key = (edge["from"], edge["to"])
            if key not in seen:
                unique_edges.append(edge)
                seen.add(key)

        # If no edges generated, fall back to sequential
        if not unique_edges:
            stage_list = [s for s in stages if s.get("id")]
            for i, stage in enumerate(stage_list[:-1]):
                unique_edges.append({"from": stage["id"], "to": stage_list[i + 1]["id"]})

        config["edges"] = unique_edges
