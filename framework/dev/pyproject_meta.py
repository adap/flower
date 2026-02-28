#!/usr/bin/env python3
"""Read and update [project] metadata in pyproject.toml."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


class ProjectSectionError(RuntimeError):
    """Raised when [project] metadata is missing or malformed."""


def _load_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def _find_project_bounds(lines: list[str]) -> tuple[int, int]:
    start = -1
    for idx, line in enumerate(lines):
        if line.strip() == "[project]":
            start = idx
            break

    if start == -1:
        raise ProjectSectionError("Missing [project] section in pyproject.toml")

    end = len(lines)
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            end = idx
            break

    return start, end


def _get_project_value(lines: list[str], key: str) -> str:
    start, end = _find_project_bounds(lines)
    pattern = re.compile(rf'^\s*{re.escape(key)}\s*=\s*"([^"]+)"')

    for idx in range(start + 1, end):
        match = pattern.match(lines[idx])
        if match:
            return match.group(1)

    raise ProjectSectionError(f'Missing [project].{key} in pyproject.toml')


def _set_project_value(lines: list[str], key: str, value: str) -> None:
    start, end = _find_project_bounds(lines)
    pattern = re.compile(rf'^(\s*{re.escape(key)}\s*=\s*")([^"]*)(".*)$')

    for idx in range(start + 1, end):
        match = pattern.match(lines[idx])
        if match:
            lines[idx] = f"{match.group(1)}{value}{match.group(3)}\n"
            return

    raise ProjectSectionError(f'Missing [project].{key} in pyproject.toml')


def _cmd_get_name(path: Path) -> int:
    print(_get_project_value(_load_lines(path), "name"))
    return 0


def _cmd_get_version(path: Path) -> int:
    print(_get_project_value(_load_lines(path), "version"))
    return 0


def _cmd_set_nightly(path: Path) -> int:
    lines = _load_lines(path)

    name = _get_project_value(lines, "name")
    version = _get_project_value(lines, "version")

    nightly_name = name if name.endswith("-nightly") else f"{name}-nightly"
    base_version = version.split(".dev", maxsplit=1)[0]
    nightly_version = f"{base_version}.dev{datetime.utcnow().strftime('%Y%m%d')}"

    _set_project_value(lines, "name", nightly_name)
    _set_project_value(lines, "version", nightly_version)

    path.write_text("".join(lines), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_get_name = sub.add_parser("get-name", help="Print [project].name")
    p_get_name.add_argument("pyproject", type=Path)

    p_get_version = sub.add_parser("get-version", help="Print [project].version")
    p_get_version.add_argument("pyproject", type=Path)

    p_set_nightly = sub.add_parser(
        "set-nightly-name-version",
        help="Set [project].name and [project].version for nightly publishing",
    )
    p_set_nightly.add_argument("pyproject", type=Path)

    args = parser.parse_args()

    try:
        if args.command == "get-name":
            return _cmd_get_name(args.pyproject)
        if args.command == "get-version":
            return _cmd_get_version(args.pyproject)
        if args.command == "set-nightly-name-version":
            return _cmd_set_nightly(args.pyproject)
    except ProjectSectionError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    print(f"Unsupported command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
