"""Utility used to bump the version of the package."""

import argparse
import re
from pathlib import Path


REPLACE_CURR_VERSION = {
    "doc/source/conf.py": [
        ".. |stable_flwr_version| replace:: {version}",
    ],
    "examples/*/pyproject.toml": [
        "flwr[simulation]=={version}",
        "flwr[simulation]>={version}",
    ],
    "src/py/flwr/cli/new/templates/app/pyproject.*.toml.tpl": [
        "flwr[simulation]>={version}",
    ],
}

REPLACE_NEXT_VERSION = {
    "pyproject.toml": ['version = "{version}"'],
    "doc/source/conf.py": [
        'release = "{version}"',
    ],
    "examples/doc/source/conf.py": ['release = "{version}"'],
    "baselines/doc/source/conf.py": ['release = "{version}"'],
}


def _get_next_version(curr_version, increment):
    """Calculate the next version based on the type of release."""
    major, minor, patch_version = map(int, curr_version.split("."))
    if increment == "patch":
        patch_version += 1
    elif increment == "minor":
        minor += 1
        patch_version = 0
    elif increment == "major":
        major += 1
        minor = 0
        patch_version = 0
    else:
        raise ValueError(
            "Invalid increment type. Must be 'major', 'minor', or 'patch'."
        )
    return f"{major}.{minor}.{patch_version}"


def _update_versions(file_patterns, replace_strings, new_version, check):
    """Update the version strings in the specified files."""
    for pattern in file_patterns:
        files = list(Path(__file__).parents[1].glob(pattern))
        for file_path in files:
            if not file_path.is_file():
                continue
            content = file_path.read_text()
            original_content = content
            for s in replace_strings:
                # Construct regex pattern to match any version number in the string
                escaped_s = re.escape(s).replace(r"\{version\}", r"(\d+\.\d+\.\d+)")
                regex_pattern = re.compile(escaped_s)
                content = regex_pattern.sub(s.format(version=new_version), content)
            if content != original_content:
                if check:
                    raise ValueError(f"The version in {file_path} seems incorrect")
                file_path.write_text(content)
                print(f"Updated {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility used to bump the version of the package."
    )
    parser.add_argument("current_version", help="Current version of the package.")
    parser.add_argument(
        "--check", action="store_true", help="Fails if any file would be modified."
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--patch", action="store_true", help="Increment the patch version."
    )
    group.add_argument(
        "--major", action="store_true", help="Increment the major version."
    )
    args = parser.parse_args()

    curr_version = args.current_version

    # Determine the type of version increment
    if args.major:
        increment = "major"
    elif args.patch:
        increment = "patch"
    else:
        increment = "minor"

    next_version = _get_next_version(curr_version, increment)

    # Update files with next version
    for file_pattern, strings in REPLACE_NEXT_VERSION.items():
        _update_versions([file_pattern], strings, next_version, args.check)

    # Update files with current version
    for file_pattern, strings in REPLACE_CURR_VERSION.items():
        _update_versions([file_pattern], strings, curr_version, args.check)
