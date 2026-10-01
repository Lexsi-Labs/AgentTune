"""Pure, testable helpers for AgentTune patch releases.

The version is not stored in any tracked file: setuptools_scm derives it from
the git tag at build time. The release workflow tags the release commit
*before* building, so the built sdist/wheel carry exactly the tag's version,
and ``validate-dist`` proves it before anything is pushed or uploaded.

Network access and repository mutations stay in the workflow; it gathers the
tag / GitHub release / PyPI state and passes booleans to these helpers, which
keeps the safety rules easy to test.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import tomllib
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from email.parser import Parser
from enum import Enum
from pathlib import Path

SEMVER_PATTERN = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


class ReleaseError(ValueError):
    """Raised when version or release state is unsafe or inconsistent."""


class ReleaseState(str, Enum):
    READY = "ready"
    RESUMABLE = "resumable"
    COMPLETE = "complete"
    CONFLICT = "conflict"


class ReleaseAction(str, Enum):
    RELEASE = "release"
    NOOP = "noop"


@dataclass(frozen=True)
class Presence:
    """Whether an exact version exists in each external release system."""

    tag: bool
    github_release: bool
    pypi: bool


@dataclass(frozen=True)
class Decision:
    action: ReleaseAction
    version: str
    state: ReleaseState
    reason: str


def parse_version(value: str) -> tuple[int, int, int]:
    """Parse a plain MAJOR.MINOR.PATCH version, rejecting all extensions."""

    match = SEMVER_PATTERN.fullmatch(value)
    if not match:
        raise ReleaseError(
            f"Invalid version {value!r}; expected plain SemVer MAJOR.MINOR.PATCH "
            "with no prefix, suffix, build metadata, or leading zeroes."
        )
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def next_patch(value: str) -> str:
    major, minor, patch = parse_version(value)
    return f"{major}.{minor}.{patch + 1}"


def bootstrap_version(root: Path | str = Path(".")) -> str:
    """Version for the very first release: setuptools_scm's ``fallback_version``.

    Only used while the repository has no GitHub release yet; afterwards every
    release is the next patch of the latest one.
    """

    with open(Path(root) / "pyproject.toml", "rb") as handle:
        config = tomllib.load(handle)
    value = config.get("tool", {}).get("setuptools_scm", {}).get("fallback_version")
    if not isinstance(value, str):
        raise ReleaseError("pyproject.toml has no [tool.setuptools_scm] fallback_version.")
    parse_version(value)
    return value


def classify_release_state(
    presence: Presence,
    *,
    tag_is_annotated: bool = True,
    tag_matches_source: bool = True,
    release_matches_tag: bool = True,
) -> ReleaseState:
    """Classify exact-version state without making an external mutation."""

    if presence.tag and (not tag_is_annotated or not tag_matches_source):
        return ReleaseState.CONFLICT
    if presence.github_release and (not presence.tag or not release_matches_tag):
        return ReleaseState.CONFLICT
    if presence.pypi and not (presence.tag and presence.github_release):
        return ReleaseState.CONFLICT
    if presence.tag and presence.github_release and presence.pypi:
        return ReleaseState.COMPLETE
    if presence.tag or presence.github_release:
        return ReleaseState.RESUMABLE
    return ReleaseState.READY


def decide_action(*, trigger: str, version: str, state: ReleaseState) -> Decision:
    """Release the requested version once; completed versions are a no-op."""

    parse_version(version)
    if state is ReleaseState.CONFLICT:
        raise ReleaseError(
            f"Release {version} has conflicting tag/GitHub/PyPI state; manual repair is required."
        )
    if trigger not in {"normal", "dispatch"}:
        raise ReleaseError(f"Unknown release trigger: {trigger!r}")
    if state is ReleaseState.COMPLETE:
        return Decision(ReleaseAction.NOOP, version, state, "release already complete")
    return Decision(ReleaseAction.RELEASE, version, state, "create or resume release")


def _metadata_version(contents: str, filename: Path) -> tuple[str, str]:
    metadata = Parser().parsestr(contents)
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ReleaseError(f"Distribution metadata in {filename} lacks Name or Version.")
    return name, version


def distribution_metadata(path: Path) -> tuple[str, str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            ]
            if len(candidates) != 1:
                raise ReleaseError(
                    f"Expected one METADATA file in {path}, found {len(candidates)}."
                )
            contents = archive.read(candidates[0]).decode("utf-8")
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            candidates = [
                member
                for member in archive.getmembers()
                if member.name.endswith("/PKG-INFO") and member.name.count("/") == 1
            ]
            if len(candidates) != 1:
                raise ReleaseError(
                    f"Expected one PKG-INFO file in {path}, found {len(candidates)}."
                )
            extracted = archive.extractfile(candidates[0])
            if extracted is None:
                raise ReleaseError(f"Could not read PKG-INFO from {path}.")
            contents = extracted.read().decode("utf-8")
    else:
        raise ReleaseError(f"Unsupported distribution file: {path}")
    return _metadata_version(contents, path)


def validate_distributions(
    requested: str,
    paths: Iterable[Path],
    *,
    project_name: str = "agenttune",
) -> None:
    """Every built dist must be ``project_name`` at exactly ``requested`` (the tag)."""

    parse_version(requested)
    distributions = list(paths)
    if not distributions:
        raise ReleaseError("No distributions were supplied for validation.")

    def normalize(value: str) -> str:
        return re.sub(r"[-_.]+", "-", value).lower()

    for path in distributions:
        name, version = distribution_metadata(path)
        if normalize(name) != normalize(project_name):
            raise ReleaseError(
                f"Distribution {path} has project name {name!r}, expected {project_name!r}."
            )
        if version != requested:
            raise ReleaseError(
                f"Distribution {path} has version {version!r}, expected exact version {requested!r}."
            )


def _bool(value: str) -> bool:
    normalized = value.lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return normalized == "true"


def _json_decision(decision: Decision) -> str:
    return json.dumps(
        {
            "action": decision.action.value,
            "version": decision.version,
            "state": decision.state.value,
            "reason": decision.reason,
        },
        sort_keys=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("bootstrap-version")
    next_parser = subparsers.add_parser("next-version")
    next_parser.add_argument("latest")
    validate_parser = subparsers.add_parser("validate-version")
    validate_parser.add_argument("version")
    dist_parser = subparsers.add_parser("validate-dist")
    dist_parser.add_argument("version")
    dist_parser.add_argument("paths", nargs="+", type=Path)
    dist_parser.add_argument("--project-name", default="agenttune")

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--trigger", choices=("normal", "dispatch"), required=True)
    plan_parser.add_argument("--version", required=True)
    plan_parser.add_argument("--tag", type=_bool, required=True)
    plan_parser.add_argument("--release", type=_bool, required=True)
    plan_parser.add_argument("--pypi", type=_bool, required=True)
    plan_parser.add_argument("--tag-annotated", type=_bool, default=True)
    plan_parser.add_argument("--tag-matches-source", type=_bool, default=True)
    plan_parser.add_argument("--release-matches-tag", type=_bool, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "bootstrap-version":
        print(bootstrap_version())
    elif args.command == "next-version":
        print(next_patch(args.latest))
    elif args.command == "validate-version":
        parse_version(args.version)
        print(args.version)
    elif args.command == "validate-dist":
        validate_distributions(args.version, args.paths, project_name=args.project_name)
    elif args.command == "plan":
        state = classify_release_state(
            Presence(args.tag, args.release, args.pypi),
            tag_is_annotated=args.tag_annotated,
            tag_matches_source=args.tag_matches_source,
            release_matches_tag=args.release_matches_tag,
        )
        print(
            _json_decision(decide_action(trigger=args.trigger, version=args.version, state=state))
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as error:
        raise SystemExit(f"release error: {error}") from error
