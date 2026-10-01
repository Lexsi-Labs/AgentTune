import io
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

# CI runs bare `pytest`, which puts tests/ (not the repo root) on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.release import (  # noqa: E402
    Presence,
    ReleaseAction,
    ReleaseError,
    ReleaseState,
    bootstrap_version,
    classify_release_state,
    decide_action,
    main,
    next_patch,
    parse_version,
    validate_distributions,
)


@pytest.mark.parametrize(
    ("current", "expected"),
    [("0.1.0", "0.1.1"), ("1.9.99", "1.9.100"), ("10.0.8", "10.0.9")],
)
def test_next_patch_increments_only_patch(current, expected):
    assert next_patch(current) == expected


def test_successive_merges_advance_patch_releases():
    latest = "1.1.0"
    for expected in ("1.1.1", "1.1.2", "1.1.3"):
        requested = next_patch(latest)
        assert requested == expected
        decision = decide_action(trigger="normal", version=requested, state=ReleaseState.READY)
        assert decision.action is ReleaseAction.RELEASE
        assert decision.version == expected
        latest = requested


@pytest.mark.parametrize(
    "invalid",
    ["v1.2.3", "1.2", "1.2.3.4", "01.2.3", "1.02.3", "1.2.03", "1.2.3rc1", "1.2.3+meta", ""],
)
def test_plain_semver_rejects_invalid_versions(invalid):
    with pytest.raises(ReleaseError):
        parse_version(invalid)


def test_repo_bootstrap_version_is_scm_fallback():
    # First release on a repo with no GitHub release yet.
    assert bootstrap_version(REPO_ROOT) == "1.1.0"
    state = classify_release_state(Presence(False, False, False))
    decision = decide_action(trigger="normal", version=bootstrap_version(REPO_ROOT), state=state)
    assert (decision.action, decision.version) == (ReleaseAction.RELEASE, "1.1.0")


def test_bootstrap_version_requires_plain_fallback(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.setuptools_scm]\n")
    with pytest.raises(ReleaseError, match="fallback_version"):
        bootstrap_version(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[tool.setuptools_scm]\nfallback_version = "1.1"\n')
    with pytest.raises(ReleaseError):
        bootstrap_version(tmp_path)


@pytest.mark.parametrize(
    ("presence", "expected"),
    [
        (Presence(False, False, False), ReleaseState.READY),
        (Presence(True, False, False), ReleaseState.RESUMABLE),
        (Presence(True, True, False), ReleaseState.RESUMABLE),
        (Presence(True, True, True), ReleaseState.COMPLETE),
        (Presence(False, True, False), ReleaseState.CONFLICT),
        (Presence(False, False, True), ReleaseState.CONFLICT),
        (Presence(True, False, True), ReleaseState.CONFLICT),
    ],
)
def test_release_completeness_states(presence, expected):
    assert classify_release_state(presence) is expected


def test_mismatched_or_lightweight_tag_is_conflicting():
    presence = Presence(True, False, False)
    assert classify_release_state(presence, tag_is_annotated=False) is ReleaseState.CONFLICT
    assert classify_release_state(presence, tag_matches_source=False) is ReleaseState.CONFLICT
    assert (
        classify_release_state(Presence(True, True, False), release_matches_tag=False)
        is ReleaseState.CONFLICT
    )


def test_resumable_release_is_released_again():
    decision = decide_action(trigger="normal", version="1.1.0", state=ReleaseState.RESUMABLE)
    assert decision.action is ReleaseAction.RELEASE


def test_complete_release_is_a_noop():
    # A re-run on the commit that is already the latest release must not cut a new one.
    decision = decide_action(trigger="normal", version="1.1.0", state=ReleaseState.COMPLETE)
    assert decision.action is ReleaseAction.NOOP


def test_conflicting_state_fails_decision():
    with pytest.raises(ReleaseError, match="manual repair"):
        decide_action(trigger="dispatch", version="1.1.0", state=ReleaseState.CONFLICT)


def test_plan_cli_emits_json(capsys):
    argv = ["plan", "--trigger", "normal", "--version", "1.1.1"]
    main([*argv, "--tag", "false", "--release", "false", "--pypi", "false"])
    assert '"action": "release"' in capsys.readouterr().out
    main([*argv, "--tag", "true", "--release", "true", "--pypi", "true"])
    assert '"action": "noop"' in capsys.readouterr().out


def _write_distributions(root: Path, version: str, name: str = "agenttune") -> list[Path]:
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n"
    wheel = root / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{name}-{version}.dist-info/METADATA", metadata)
    sdist = root / f"{name}-{version}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        payload = metadata.encode()
        info = tarfile.TarInfo(f"{name}-{version}/PKG-INFO")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        nested = tarfile.TarInfo(f"{name}-{version}/src/{name}.egg-info/PKG-INFO")
        nested.size = len(payload)
        archive.addfile(nested, io.BytesIO(payload))
    return [wheel, sdist]


def test_distribution_version_must_equal_tag(tmp_path):
    dists = _write_distributions(tmp_path, "1.1.1")
    validate_distributions("1.1.1", dists)
    with pytest.raises(ReleaseError, match="expected exact version"):
        validate_distributions("1.1.2", dists)


def test_untagged_dev_build_is_rejected(tmp_path):
    # What setuptools_scm builds when the tag is missing or the tree is dirty.
    dists = _write_distributions(tmp_path, "1.1.2.dev0")
    with pytest.raises(ReleaseError, match="expected exact version"):
        validate_distributions("1.1.1", dists)


def test_distribution_name_must_match(tmp_path):
    dists = _write_distributions(tmp_path, "1.1.1", name="auditkit")
    with pytest.raises(ReleaseError, match="project name"):
        validate_distributions("1.1.1", dists)
