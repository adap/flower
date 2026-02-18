#!/usr/bin/env python3

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from packaging.version import InvalidVersion, Version

MIN_VERSION = Version("1.8.0")
MINOR_LABEL_FROM = Version("1.21.0")


def _display_version(version: Version) -> str:
    if version >= MINOR_LABEL_FROM:
        return f"v{version.release[0]}.{version.release[1]}.x"
    return f"v{version}"


def _collect_versions() -> list[dict[str, str]]:
    tags = subprocess.run(
        ["git", "tag", "-l", "v*.*.*", "framework-v*.*.*"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    versions: list[Version] = []
    for tag in tags:
        try:
            version = Version(tag.split("v")[-1])
        except InvalidVersion:
            continue
        if version < MIN_VERSION:
            continue
        versions.append(version)

    versions.sort(reverse=True)

    version_items = [{"name": "main"}]
    added_names = {"main"}
    for version in versions:
        name = _display_version(version)
        if name in added_names:
            continue
        version_items.append({"name": name})
        added_names.add(name)

    return version_items


def _load_announcement(ui_config_path: Path) -> dict[str, object]:
    if not ui_config_path.exists():
        return {"enabled": False, "html": ""}

    with ui_config_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    announcement = payload.get("announcement", {}) if isinstance(payload, dict) else {}
    if not isinstance(announcement, dict):
        return {"enabled": False, "html": ""}

    enabled = bool(announcement.get("enabled"))
    html = str(announcement.get("html", "")).strip()

    return {
        "enabled": enabled and bool(html),
        "html": html,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate runtime docs UI metadata")
    parser.add_argument(
        "--ui-config",
        type=Path,
        required=True,
        help="Path to docs UI config JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for generated runtime metadata JSON",
    )
    args = parser.parse_args()

    runtime_payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "versions": _collect_versions(),
        "announcement": _load_announcement(args.ui_config),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(runtime_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
