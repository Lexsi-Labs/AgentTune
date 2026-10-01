"""Template discovery and registry."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class TemplateRegistry:
    """
    Discover and manage available templates.

    Scans template directory and indexes templates by ID.
    """

    def __init__(self) -> None:
        """Initialize template registry."""
        self._templates: dict[str, dict[str, Any]] = {}

    def discover(self, templates_dir: str) -> None:
        """
        Auto-discover all templates in directory.

        Args:
            templates_dir: Path to templates directory
        """
        templates_path = Path(templates_dir)
        if not templates_path.exists():
            return

        # Scan all YAML files recursively
        for yaml_file in templates_path.glob("**/*.yaml"):
            try:
                with open(yaml_file) as f:
                    content = yaml.safe_load(f)

                if content and "id" in content:
                    template_id = content["id"]
                    # Extract metadata
                    metadata = {
                        "id": template_id,
                        "name": content.get("name", ""),
                        "version": content.get("version", ""),
                        "description": content.get("description", ""),
                        "tags": content.get("tags", []),
                        "compliance_note": content.get("compliance_note", ""),
                        "category": self._extract_category(template_id),
                        "path": str(yaml_file),
                    }
                    self._templates[template_id] = metadata
            except Exception as e:
                logger.warning(f"Warning: Failed to load template {yaml_file}: {str(e)}")

    def list_all(self) -> list[dict[str, Any]]:
        """
        List all discovered templates.

        Returns:
            List of template metadata dictionaries
        """
        return list(self._templates.values())

    def get(self, template_id: str) -> dict[str, Any] | None:
        """
        Get template metadata by ID.

        Args:
            template_id: Template identifier

        Returns:
            Template metadata or None if not found
        """
        return self._templates.get(template_id)

    def list_by_category(self, category: str) -> list[dict[str, Any]]:
        """
        List templates by category.

        Args:
            category: Category name (e.g., "bfsi", "generic")

        Returns:
            List of template metadata in category
        """
        return [t for t in self._templates.values() if t.get("category") == category]

    @staticmethod
    def _extract_category(template_id: str) -> str:
        """Extract category from template ID (e.g., 'bfsi' from 'bfsi/kyc_triage')."""
        parts = template_id.split("/")
        return parts[0] if parts else ""
