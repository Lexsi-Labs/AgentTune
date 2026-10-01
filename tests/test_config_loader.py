"""Tests for ConfigLoader."""

import pytest

from agenttune.decide.config import ConfigLoader


class TestDeepMerge:
    """Test deep_merge functionality."""

    def test_deep_merge_simple(self):
        """Test merging simple dicts."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = ConfigLoader.deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self):
        """Test merging nested dicts."""
        base = {"parent": {"child1": 1, "child2": 2}}
        override = {"parent": {"child2": 3}}
        result = ConfigLoader.deep_merge(base, override)
        assert result == {"parent": {"child1": 1, "child2": 3}}

    def test_deep_merge_lists_replace(self):
        """Test that lists are replaced."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = ConfigLoader.deep_merge(base, override)
        assert result == {"items": [4, 5]}

    def test_deep_merge_skip_extends(self):
        """Test that extends field is skipped."""
        base = {"key": "value"}
        override = {"extends": "path", "other": "data"}
        result = ConfigLoader.deep_merge(base, override)
        assert "extends" not in result
        assert result == {"key": "value", "other": "data"}


class TestValidation:
    """Test configuration validation."""

    def test_validation_valid_config(self):
        """Test validation passes for valid config."""
        config = {
            "id": "test/template",
            "name": "Test",
            "version": "1.0.0",
            "stages": [{"id": "s1", "type": "llm_call"}],
        }
        ConfigLoader._validate(config)  # Should not raise

    def test_validation_missing_stages(self):
        """Test validation fails without stages."""
        config = {"id": "test", "name": "Test", "version": "1.0.0"}
        with pytest.raises(ValueError, match="at least one stage"):
            ConfigLoader._validate(config)

    def test_validation_missing_id(self):
        """Test validation fails without id."""
        config = {"name": "Test", "version": "1.0.0", "stages": [{"id": "s1", "type": "llm_call"}]}
        with pytest.raises(ValueError, match="id"):
            ConfigLoader._validate(config)

    def test_validation_invalid_stage_type(self):
        """Test validation fails with invalid stage type."""
        config = {
            "id": "test",
            "name": "Test",
            "version": "1.0.0",
            "stages": [{"id": "s1", "type": "invalid"}],
        }
        with pytest.raises(ValueError, match="Unknown stage type"):
            ConfigLoader._validate(config)
