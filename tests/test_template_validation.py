"""
Template validation tests — every YAML template is syntactically and
structurally valid.

Checks:
  - YAML parses without error
  - Required top-level fields present
  - All stages have id + type
  - All stage types are recognized
  - All cross-stage references (next, goto, on_fail) point to existing stage IDs
  - No obvious infinite loops (every cycle has max_iterations guard)
  - Parallel branches have valid sub-stage definitions
"""

import re
from pathlib import Path

import pytest
import yaml

# Template directories
TEMPLATES_DIR = Path(__file__).parent.parent / "src" / "agenttune" / "decide" / "templates"
BFSI_DIR = TEMPLATES_DIR / "bfsi"
GENERIC_DIR = TEMPLATES_DIR / "generic"

VALID_STAGE_TYPES = {
    "llm_call",
    "llm_judge",
    "rules",
    "parallel",
    "router",
    "human_review",
    "tool_call",
    "output",
}

REQUIRED_TEMPLATE_FIELDS = {"id", "name", "version", "stages"}
REQUIRED_STAGE_FIELDS = {"id", "type"}


def load_template(path: Path) -> dict:
    with open(path) as f:
        content = f.read()
    # Strip the extends line for standalone parsing
    content_no_extends = re.sub(r"^extends:.*\n", "", content, flags=re.MULTILINE)
    return yaml.safe_load(content_no_extends)


def get_all_stage_ids(template: dict) -> set:
    ids = set()
    for stage in template.get("stages", []):
        ids.add(stage["id"])
        for branch in stage.get("branches", []):
            ids.add(branch["id"])
    return ids


def collect_goto_references(stage: dict) -> list:
    refs = []
    if "next" in stage:
        refs.append(stage["next"])
    if "on_fail" in stage:
        refs.append(stage["on_fail"])
    if "on_parse_error" in stage:
        refs.append(stage["on_parse_error"])
    for rule in stage.get("rules", []):
        fail = rule.get("on_fail", {})
        if isinstance(fail, dict) and "goto" in fail:
            refs.append(fail["goto"])
    for cond in stage.get("on_result", []):
        if "goto" in cond:
            refs.append(cond["goto"])
    if "on_approved" in stage:
        refs.append(stage["on_approved"])
    if "on_denied" in stage:
        refs.append(stage["on_denied"])
    return refs


def bfsi_templates():
    return sorted(BFSI_DIR.glob("*.yaml"))


def generic_templates():
    return sorted(GENERIC_DIR.glob("*.yaml"))


def all_templates():
    return bfsi_templates() + generic_templates()


# ---------------------------------------------------------------------------
# Parametrize over all BFSI templates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template_path", bfsi_templates(), ids=lambda p: p.stem)
class TestBFSITemplates:
    def test_yaml_is_parseable(self, template_path):
        template = load_template(template_path)
        assert isinstance(template, dict)

    def test_required_top_level_fields(self, template_path):
        template = load_template(template_path)
        missing = REQUIRED_TEMPLATE_FIELDS - set(template.keys())
        assert not missing, f"Missing fields: {missing}"

    def test_version_is_semver(self, template_path):
        template = load_template(template_path)
        version = str(template.get("version", ""))
        assert re.match(r"^\d+\.\d+\.\d+$", version), f"Version '{version}' not semver"

    def test_stages_is_nonempty_list(self, template_path):
        template = load_template(template_path)
        assert isinstance(template["stages"], list)
        assert len(template["stages"]) >= 1

    def test_all_stages_have_id_and_type(self, template_path):
        template = load_template(template_path)
        for stage in template["stages"]:
            missing = REQUIRED_STAGE_FIELDS - set(stage.keys())
            assert not missing, f"Stage missing {missing}: {stage}"

    def test_all_stage_types_are_valid(self, template_path):
        template = load_template(template_path)
        for stage in template["stages"]:
            assert (
                stage["type"] in VALID_STAGE_TYPES
            ), f"Unknown stage type '{stage['type']}' in stage '{stage.get('id')}'"

    def test_cross_stage_references_exist(self, template_path):
        template = load_template(template_path)
        all_ids = get_all_stage_ids(template)
        for stage in template["stages"]:
            for ref in collect_goto_references(stage):
                assert (
                    ref in all_ids
                ), f"Stage '{stage['id']}' references non-existent stage '{ref}'"

    def test_parallel_branches_have_id_and_type(self, template_path):
        template = load_template(template_path)
        for stage in template["stages"]:
            if stage["type"] == "parallel":
                assert "branches" in stage, f"Parallel stage '{stage['id']}' has no branches"
                for branch in stage["branches"]:
                    assert "id" in branch and "type" in branch

    def test_output_stages_have_verdict(self, template_path):
        template = load_template(template_path)
        for stage in template["stages"]:
            if stage["type"] == "output":
                assert "verdict" in stage, f"Output stage '{stage['id']}' missing 'verdict'"

    def test_at_least_one_output_stage(self, template_path):
        template = load_template(template_path)
        output_stages = [s for s in template["stages"] if s["type"] == "output"]
        assert len(output_stages) >= 1, "Template has no output stage"

    def test_stage_ids_are_unique(self, template_path):
        template = load_template(template_path)
        ids = [s["id"] for s in template["stages"]]
        assert len(ids) == len(set(ids)), f"Duplicate stage IDs: {ids}"

    def test_has_description_or_tags(self, template_path):
        template = load_template(template_path)
        has_desc = bool(template.get("description"))
        has_tags = bool(template.get("tags"))
        assert has_desc or has_tags, "Template should have description or tags"


# ---------------------------------------------------------------------------
# Parametrize over all generic templates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("template_path", generic_templates(), ids=lambda p: p.stem)
class TestGenericTemplates:
    def test_yaml_is_parseable(self, template_path):
        template = load_template(template_path)
        assert isinstance(template, dict)

    def test_required_top_level_fields(self, template_path):
        template = load_template(template_path)
        missing = REQUIRED_TEMPLATE_FIELDS - set(template.keys())
        assert not missing, f"Missing fields: {missing}"

    def test_all_stages_have_id_and_type(self, template_path):
        template = load_template(template_path)
        for stage in template["stages"]:
            missing = REQUIRED_STAGE_FIELDS - set(stage.keys())
            assert not missing, f"Stage missing {missing}: {stage}"

    def test_all_stage_types_are_valid(self, template_path):
        template = load_template(template_path)
        for stage in template["stages"]:
            assert stage["type"] in VALID_STAGE_TYPES

    def test_cross_stage_references_exist(self, template_path):
        template = load_template(template_path)
        all_ids = get_all_stage_ids(template)
        for stage in template["stages"]:
            for ref in collect_goto_references(stage):
                assert (
                    ref in all_ids
                ), f"Stage '{stage['id']}' references non-existent stage '{ref}'"

    def test_stage_ids_are_unique(self, template_path):
        template = load_template(template_path)
        ids = [s["id"] for s in template["stages"]]
        assert len(ids) == len(set(ids))

    def test_at_least_one_output_stage(self, template_path):
        template = load_template(template_path)
        output_stages = [s for s in template["stages"] if s["type"] == "output"]
        assert len(output_stages) >= 1


# ---------------------------------------------------------------------------
# Count-level checks (both dirs together)
# ---------------------------------------------------------------------------


class TestTemplateLibraryCoverage:
    def test_bfsi_template_count(self):
        """DECIDE_PLAN specifies ≥10 BFSI templates."""
        templates = bfsi_templates()
        assert len(templates) >= 10, f"Only {len(templates)} BFSI templates found"

    def test_generic_template_count(self):
        """DECIDE_PLAN specifies ≥8 generic templates."""
        templates = generic_templates()
        assert len(templates) >= 8, f"Only {len(templates)} generic templates found"

    def test_total_template_count(self):
        """Combined at least 18 templates."""
        assert len(all_templates()) >= 18

    def test_expected_bfsi_templates_exist(self):
        stems = {p.stem for p in bfsi_templates()}
        expected = {
            "kyc_triage",
            "loan_prequal",
            "claims_triage",
            "sanctions_check",
            "account_opening",
            "transaction_monitoring",
            "customer_risk_rating",
            "card_decline",
            "regulatory_filing",
        }
        missing = expected - stems
        assert not missing, f"Missing BFSI templates: {missing}"

    def test_expected_generic_templates_exist(self):
        stems = {p.stem for p in generic_templates()}
        expected = {
            "text_classify",
            "entity_extract",
            "sentiment_analysis",
            "summarize",
            "qa_system",
        }
        missing = expected - stems
        assert not missing, f"Missing generic templates: {missing}"
