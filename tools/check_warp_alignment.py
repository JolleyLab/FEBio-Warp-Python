"""Report where this repository's tooling differs from Warp's.

FEBio-Warp follows the conventions of Warp (https://github.com/NVIDIA/warp) for
its Python version, linting and formatting. This script compares the settings
here with those of the latest Warp release and prints a Markdown report. It
changes nothing. A scheduled workflow (.github/workflows/warp-alignment.yml)
runs it and opens an issue when something differs, so that updates are made
deliberately and in one pass. See docs/WARP_ALIGNMENT.md.

Usage:
    uv run tools/check_warp_alignment.py                  # compare with the latest Warp release
    uv run tools/check_warp_alignment.py --ref main       # compare with Warp's main branch
    uv run tools/check_warp_alignment.py --warp-dir ../warp   # compare with a local checkout

The script uses only the standard library. It exits with status 0 unless
``--strict`` is given, in which case it exits with status 1 when a tracked
setting differs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WARP_RAW_URL = "https://raw.githubusercontent.com/NVIDIA/warp/{ref}/{path}"
WARP_PYPI_URL = "https://pypi.org/pypi/warp-lang/json"

#: Hooks whose pinned versions are compared: name in the report, repository URL.
TRACKED_HOOKS = {
    "Ruff": "https://github.com/astral-sh/ruff-pre-commit",
    "uv": "https://github.com/astral-sh/uv-pre-commit",
    "typos": "https://github.com/crate-ci/typos",
}

#: Lint rules Warp selects that this project has decided not to adopt yet.
#: Add a rule here, with the reason, to stop it from being reported.
RULES_NOT_ADOPTED = {
    "S101": "assert is used in examples and tools here",
    "D200": "docstring style rules are not enforced yet",
    "D205": "docstring style rules are not enforced yet",
    "D400": "docstring style rules are not enforced yet",
    "D401": "docstring style rules are not enforced yet",
    "D403": "docstring style rules are not enforced yet",
}


@dataclass
class Settings:
    """The tooling settings read from one repository."""

    label: str
    python_version: str | None = None
    requires_python: str | None = None
    line_length: int | None = None
    lint_select: set[str] = field(default_factory=set)
    lint_ignore: set[str] = field(default_factory=set)
    hooks: dict[str, str] = field(default_factory=dict)
    warp_requirement: str | None = None


def parse_settings(label: str, pyproject_text: str, python_version_text: str, pre_commit_text: str) -> Settings:
    """Read the tracked settings from the text of three configuration files."""
    pyproject = tomllib.loads(pyproject_text)
    project = pyproject.get("project", {})
    ruff = pyproject.get("tool", {}).get("ruff", {})
    lint = ruff.get("lint", {})

    requirements = [*project.get("dependencies", [])]
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    warp_requirement = next((r for r in requirements if r.replace("_", "-").startswith("warp-lang")), None)

    return Settings(
        label=label,
        python_version=python_version_text.strip() or None,
        requires_python=project.get("requires-python"),
        line_length=ruff.get("line-length"),
        lint_select=set(lint.get("select", [])),
        lint_ignore=set(lint.get("ignore", [])),
        hooks=parse_hook_versions(pre_commit_text),
        warp_requirement=warp_requirement,
    )


def parse_hook_versions(pre_commit_text: str) -> dict[str, str]:
    """Return the pinned version of each tracked pre-commit hook.

    The file is read with a regular expression so that no YAML package is needed.
    """
    versions = {}
    for name, url in TRACKED_HOOKS.items():
        match = re.search(
            rf"repo:\s*{re.escape(url)}\s*\n(?:\s*#[^\n]*\n)*\s*rev:\s*['\"]?v?([^\s'\"]+)", pre_commit_text
        )
        if match:
            versions[name] = match.group(1)
    return versions


def read_local(root: Path, label: str) -> Settings:
    """Read the settings of a checkout on disk."""

    def text(name: str) -> str:
        path = root / name
        return path.read_text(encoding="utf-8") if path.exists() else ""

    return parse_settings(label, text("pyproject.toml"), text(".python-version"), text(".pre-commit-config.yaml"))


def fetch(url: str) -> str:
    """Download a text file."""
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def latest_warp_release() -> str:
    """Return the version number of the latest warp-lang release on PyPI."""
    return json.loads(fetch(WARP_PYPI_URL))["info"]["version"]


def read_warp(ref: str) -> Settings:
    """Read Warp's settings at a Git reference (a tag such as v1.19.0, or a branch)."""

    def text(name: str) -> str:
        return fetch(WARP_RAW_URL.format(ref=ref, path=name))

    return parse_settings(
        f"Warp {ref}", text("pyproject.toml"), text(".python-version"), text(".pre-commit-config.yaml")
    )


def pinned_warp_version(requirement: str | None) -> str | None:
    """Return the version from a requirement such as ``warp-lang==1.19.0`` or ``warp-lang>=1.19``."""
    if requirement is None:
        return None
    match = re.search(r"[=>~]=\s*([0-9][0-9A-Za-z.]*)", requirement)
    return match.group(1) if match else None


def version_key(version: str) -> tuple[int, ...]:
    """Turn ``1.19.0`` into ``(1, 19, 0)`` for comparison; non-numeric parts are dropped."""
    return tuple(int(part) for part in re.findall(r"\d+", version))


def compare(ours: Settings, warp: Settings, latest_release: str | None) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Compare two sets of settings.

    Returns:
        The tracked differences as ``(setting, here, Warp)`` rows, and a list of
        notes that are for information only.
    """
    differences: list[tuple[str, str, str]] = []
    notes: list[str] = []

    def check_equal(name: str, here: object, there: object) -> None:
        if there is not None and here != there:
            differences.append((name, str(here), str(there)))

    def check_version(name: str, here: str | None, there: str | None) -> None:
        """Report a difference only when Warp is ahead; being ahead of Warp is a note."""
        if there is None or here == there:
            return
        if here is None or version_key(there) > version_key(here):
            differences.append((name, str(here), there))
        else:
            notes.append(f"{name}: {here} here is ahead of {there} in {warp.label}.")

    check_version(
        "Python version used for development and CI (`.python-version`)", ours.python_version, warp.python_version
    )
    check_version("Oldest supported Python (`requires-python`)", ours.requires_python, warp.requires_python)
    check_equal("Line length", ours.line_length, warp.line_length)
    for name in TRACKED_HOOKS:
        check_version(f"{name} version in `.pre-commit-config.yaml`", ours.hooks.get(name), warp.hooks.get(name))

    missing = sorted(warp.lint_select - ours.lint_select - set(RULES_NOT_ADOPTED))
    if missing:
        differences.append(("Lint rules Warp selects that are not selected here", "not selected", ", ".join(missing)))

    our_version = pinned_warp_version(ours.warp_requirement)
    if latest_release and our_version and version_key(latest_release) > version_key(our_version):
        differences.append(("Warp release", f"{ours.warp_requirement}", f"{latest_release} is available"))

    extra_selected = sorted(ours.lint_select - warp.lint_select)
    if extra_selected:
        notes.append("Lint rules selected here but not in Warp: " + ", ".join(extra_selected) + ".")
    extra_ignored = sorted(ours.lint_ignore - warp.lint_ignore)
    if extra_ignored:
        notes.append(
            "Lint rules ignored here but not in Warp: "
            + ", ".join(extra_ignored)
            + ". These are listed with their reasons in `pyproject.toml` and are meant to shrink over time."
        )
    skipped = sorted(set(RULES_NOT_ADOPTED) & warp.lint_select)
    if skipped:
        notes.append(
            "Warp rules deliberately not adopted yet (see `RULES_NOT_ADOPTED` in this script): "
            + ", ".join(skipped)
            + "."
        )
    return differences, notes


def render(ours: Settings, warp: Settings, differences: list[tuple[str, str, str]], notes: list[str]) -> str:
    """Write the comparison as Markdown."""
    lines = [f"## Warp alignment: {ours.label} compared with {warp.label}", ""]
    if differences:
        lines += ["| Setting | Here | Warp |", "| --- | --- | --- |"]
        lines += [f"| {name} | {here} | {there} |" for name, here, there in differences]
        lines += ["", "Nothing was changed. See `docs/WARP_ALIGNMENT.md` for how to apply an update."]
    else:
        lines.append("No differences in the tracked settings.")
    if notes:
        lines += ["", "For information:", ""]
        lines += [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ref", help="Warp Git reference to compare with; the default is the latest release")
    parser.add_argument("--warp-dir", type=Path, help="compare with a local Warp checkout instead of downloading")
    parser.add_argument("--strict", action="store_true", help="exit with status 1 if a tracked setting differs")
    args = parser.parse_args()

    ours = read_local(REPO_ROOT, "this repository")
    latest_release = None
    if args.warp_dir:
        warp = read_local(args.warp_dir, f"Warp checkout at {args.warp_dir}")
    else:
        latest_release = latest_warp_release()
        warp = read_warp(args.ref or f"v{latest_release}")

    differences, notes = compare(ours, warp, latest_release)
    report = render(ours, warp, differences, notes)
    print(report)

    # In GitHub Actions, show the report on the run page and tell later steps the result.
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(report)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"differences={len(differences)}\n")
    return 1 if (args.strict and differences) else 0


if __name__ == "__main__":
    sys.exit(main())
