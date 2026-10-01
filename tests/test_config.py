"""
Tests for ConfigLoader: YAML loading, merging, and validation.

Tests the deep merge strategy and config validation logic.
No API calls required.
"""

import tempfile
from pathlib import Path

import pytest

from agenttune.decide.config import ConfigLoader


class TestConfigLoaderBasics:
    """Test basic YAML loading and parsing."""

    def test_load_yaml_file(self):
        """Test loading a simple YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value\nnumber: 42")
            f.flush()
            result = ConfigLoader._load_yaml(f.name)
            assert result == {"key": "value", "number": 42}
            Path(f.name).unlink()

    def test_load_empty_yaml(self):
        """Test loading an empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            result = ConfigLoader._load_yaml(f.name)
            assert result == {}
            Path(f.name).unlink()

    def test_load_yaml_with_lists(self):
        """Test loading YAML with lists."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("items:\n  - item1\n  - item2\n")
            f.flush()
            result = ConfigLoader._load_yaml(f.name)
            assert result == {"items": ["item1", "item2"]}
            Path(f.name).unlink()

    def test_load_yaml_nested(self):
        """Test loading YAML with nested structures."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("parent:\n  child: value\n  nested:\n    deep: 123\n")
            f.flush()
            result = ConfigLoader._load_yaml(f.name)
            assert result == {"parent": {"child": "value", "nested": {"deep": 123}}}
            Path(f.name).unlink()

    def test_load_yaml_file_not_found(self):
        """Test loading a non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader._load_yaml("/nonexistent/path/config.yaml")


class TestDeepMerge:
    """Test the deep merge algorithm."""

    def test_merge_simple_scalars(self):
        """Test merging simple scalar values."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = ConfigLoader.deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {"parent": {"child1": "value1", "child2": "value2"}}
        override = {"parent": {"child2": "new_value2", "child3": "value3"}}
        result = ConfigLoader.deep_merge(base, override)
        assert result == {
            "parent": {"child1": "value1", "child2": "new_value2", "child3": "value3"}
        }

    def test_merge_lists_replace(self):
        """Test that lists are replaced entirely, not merged."""
        base = {"items": ["a", "b", "c"]}
        override = {"items": ["x", "y"]}
        result = ConfigLoader.deep_merge(base, override)
        assert result == {"items": ["x", "y"]}

    def test_merge_deeply_nested(self):
        """Test merging deeply nested structures."""
        base = {"level1": {"level2": {"level3": {"a": 1, "b": 2}}}}
        override = {"level1": {"level2": {"level3": {"b": 3, "c": 4}}}}
        result = ConfigLoader.deep_merge(base, override)
        assert result == {"level1": {"level2": {"level3": {"a": 1, "b": 3, "c": 4}}}}

    def test_merge_skip_extends_field(self):
        """Test that 'extends' field is skipped."""
        base = {"key": "base_value", "extends": "some/path"}
        override = {"extends": "other/path"}
        result = ConfigLoader.deep_merge(base, override)
        # extends should be skipped
        assert "extends" not in result
        assert result["key"] == "base_value"

    def test_merge_empty_dicts(self):
        """Test merging with empty dicts."""
        base = {}
        override = {"a": 1}
        result = ConfigLoader.deep_merge(base, override)
        assert result == {"a": 1}

    def test_merge_override_empty(self):
        """Test merging when override is empty."""
        base = {"a": 1, "b": 2}
        override = {}
        result = ConfigLoader.deep_merge(base, override)
        assert result == {"a": 1, "b": 2}


class TestConfigValidation:
    """Test config validation."""

    def test_validate_valid_config(self):
        """Test validation of a valid config."""
        config = {
            "id": "test/template",
            "version": "1.0.0",
            "name": "Test Template",
            "stages": [{"id": "stage1", "type": "llm_call"}],
        }
        # Should not raise
        ConfigLoader._validate(config)

    def test_validate_missing_stages(self):
        """Test validation fails when stages missing."""
        config = {"id": "test/template", "version": "1.0.0", "name": "Test Template"}
        with pytest.raises(ValueError, match="must define at least one stage"):
            ConfigLoader._validate(config)

    def test_validate_empty_stages(self):
        """Test validation fails when stages is empty."""
        config = {"id": "test/template", "version": "1.0.0", "name": "Test Template", "stages": []}
        with pytest.raises(ValueError, match="must define at least one stage"):
            ConfigLoader._validate(config)

    def test_validate_stage_missing_id(self):
        """Test validation fails when stage missing id."""
        config = {
            "id": "test/template",
            "version": "1.0.0",
            "name": "Test Template",
            "stages": [{"type": "llm_call"}],
        }
        with pytest.raises(ValueError, match="missing id or type"):
            ConfigLoader._validate(config)

    def test_validate_stage_missing_type(self):
        """Test validation fails when stage missing type."""
        config = {
            "id": "test/template",
            "version": "1.0.0",
            "name": "Test Template",
            "stages": [{"id": "stage1"}],
        }
        with pytest.raises(ValueError, match="missing id or type"):
            ConfigLoader._validate(config)

    def test_validate_invalid_stage_type(self):
        """Test validation fails with invalid stage type."""
        config = {
            "id": "test/template",
            "version": "1.0.0",
            "name": "Test Template",
            "stages": [{"id": "stage1", "type": "invalid_type"}],
        }
        with pytest.raises(ValueError, match="Unknown stage type"):
            ConfigLoader._validate(config)

    def test_validate_valid_stage_types(self):
        """Test validation passes for all valid stage types."""
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
        for stage_type in valid_types:
            config = {
                "id": "test/template",
                "version": "1.0.0",
                "name": "Test Template",
                "stages": [{"id": "stage1", "type": stage_type}],
            }
            ConfigLoader._validate(config)  # Should not raise


class TestConfigIntegration:
    """Integration tests for config loading and merging."""

    def test_load_and_merge_with_files(self):
        """Test loading and merging two YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create global config
            global_config_path = Path(tmpdir) / "config.yaml"
            global_config_path.write_text(
                """
api_keys:
  anthropic: sk-test
default_model: claude-haiku-4-5
max_total_steps: 50
stages: []
"""
            )

            # Create template
            template_path = Path(tmpdir) / "template.yaml"
            template_path.write_text(
                """
name: Test Template
id: test/template
version: 1.0.0
default_model: claude-opus-4-1
stages:
  - id: stage1
    type: llm_call
"""
            )

            # Merge manually for testing
            base = ConfigLoader._load_yaml(str(global_config_path))
            override = ConfigLoader._load_yaml(str(template_path))
            result = ConfigLoader.deep_merge(base, override)

            assert result["default_model"] == "claude-opus-4-1"
            assert result["api_keys"]["anthropic"] == "sk-test"
            assert len(result["stages"]) == 1
            assert result["stages"][0]["id"] == "stage1"

    def test_config_with_destinations(self):
        """Test merging configs with destination settings."""
        base = {
            "destinations": {
                "file": {"enabled": True, "path": "./decisions.jsonl"},
                "postgres": {"enabled": False},
            }
        }
        override = {"destinations": {"postgres": {"enabled": True, "table": "kyc_decisions"}}}
        result = ConfigLoader.deep_merge(base, override)

        # postgres should be merged and override postgres value
        assert result["destinations"]["postgres"]["enabled"] is True
        assert result["destinations"]["postgres"]["table"] == "kyc_decisions"
        # file should be preserved
        assert result["destinations"]["file"]["enabled"] is True
