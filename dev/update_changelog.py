# mypy: ignore-errors
# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Update the changelog using PR titles."""


import pathlib
import re

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from datetime import date
from sys import argv
from typing import Optional

from github import Github
from github.PullRequest import PullRequest
from github.Repository import Repository
from github.Tag import Tag

REPO_NAME = "adap/flower"
CHANGELOG_FILE = "framework/docs/source/ref-changelog.md"
CHANGELOG_SECTION_HEADER = "### Changelog entry"

# Load the TOML configuration
with (pathlib.Path(__file__).parent.resolve() / "changelog_config.toml").open(
    "rb"
) as file:
    CONFIG = tomllib.load(file)

# Extract types, project, and scope from the config
TYPES = "|".join(CONFIG["type"])
PROJECTS = "|".join(CONFIG["project"]) + "|\\*"
SCOPE = CONFIG["scope"]
ALLOWED_VERBS = CONFIG["allowed_verbs"]

# Construct the pattern
PATTERN_TEMPLATE = CONFIG["pattern_template"]
PATTERN = PATTERN_TEMPLATE.format(types=TYPES, projects=PROJECTS, scope=SCOPE)


def _get_latest_tag(gh_api: Github) -> tuple[Repository, Optional[Tag]]:
    """Retrieve the latest tag from the GitHub repository."""
    repo = gh_api.get_repo(REPO_NAME)
    tags = repo.get_tags()
    return repo, tags[0] if tags.totalCount > 0 else None


def _add_shortlog(new_version: str, shortlog: str) -> None:
    """Update the markdown file with the new version or update existing logs."""
    token = f"<!---TOKEN_{new_version}-->"
    entry = (
        "\n### Thanks to our contributors\n\n"
        "We would like to give our special thanks to all the contributors "
        "who made the new version of Flower possible "
        f"(in `git shortlog` order):\n\n{shortlog} {token}"
    )
    current_date = date.today()

    with open(CHANGELOG_FILE, encoding="utf-8") as file:
        content = file.readlines()

    token_exists = any(token in line for line in content)

    with open(CHANGELOG_FILE, "w", encoding="utf-8") as file:
        for line in content:
            if token in line:
                token_exists = True
                file.write(line)
            elif "## Unreleased" in line and not token_exists:
                # Add the new entry under "## Unreleased"
                file.write(f"## {new_version} ({current_date})\n{entry}\n")
                token_exists = True
            else:
                file.write(line)


def _get_pull_requests_since_tag(
    repo: Repository, tag: Tag
) -> tuple[str, set[PullRequest]]:
    """Get a list of pull requests merged into the main branch since a given tag."""
    commit_shas = set()
    contributors = set()
    prs = set()

    for commit in repo.compare(tag.commit.sha, "main").commits:
        commit_shas.add(commit.sha)
        if commit.author.name is None:
            continue
        if "[bot]" in commit.author.name:
            continue
        contributors.add(commit.author.name)

    for pr_info in repo.get_pulls(
        state="closed", sort="created", direction="desc", base="main"
    ):
        if pr_info.merge_commit_sha in commit_shas:
            prs.add(pr_info)
        if len(prs) == len(commit_shas):
            break

    shortlog = ", ".join([f"`{name}`" for name in sorted(contributors)])
    return shortlog, prs


def _format_pr_reference(title: str, number: int, url: str) -> str:
    """Format a pull request reference as a markdown list item."""
    parts = title.strip().replace("*", "").split("`")
    formatted_parts = []

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Even index parts are normal text, ensure we do not add extra bold if empty
            if part.strip():
                formatted_parts.append(f"**{part.strip()}**")
            else:
                formatted_parts.append("")
        else:
            # Odd index parts are inline code
            formatted_parts.append(f"`{part.strip()}`")

    # Join parts with spaces but avoid extra spaces
    formatted_title = " ".join(filter(None, formatted_parts))
    return f"- {formatted_title} ([#{number}]({url}))"


def _extract_changelog_entry(
    pr_info: PullRequest,
) -> dict[str, str]:
    """Extract the changelog entry from a pull request's body."""
    # Use regex search to find matches
    match = re.search(PATTERN, pr_info.title)
    if match:
        # Extract components from the regex groups
        pr_type = match.group(1)
        pr_project = match.group(2)
        pr_scope = match.group(3)  # Correctly capture optional sub-scope
        pr_subject = match.group(
            4
        )  # Capture subject starting with uppercase and no terminal period
        return {
            "type": pr_type,
            "project": pr_project,
            "scope": pr_scope,
            "subject": pr_subject,
        }

    return {
        "type": "unknown",
        "project": "unknown",
        "scope": "unknown",
        "subject": "unknown",
    }


def _update_changelog(prs: set[PullRequest]) -> bool:
    """Update the changelog file with entries from provided pull requests."""
    breaking_changes = False
    unknown_changes = False

    with open(CHANGELOG_FILE, "r+", encoding="utf-8") as file:
        content = file.read()
        unreleased_index = content.find("## Unreleased")

        if unreleased_index == -1:
            print("Unreleased header not found in the changelog.")
            return False

        # Find the end of the Unreleased section
        next_header_index = content.find("## ", unreleased_index + 1)
        next_header_index = (
            next_header_index if next_header_index != -1 else len(content)
        )

        for pr_info in prs:
            parsed_title = _extract_changelog_entry(pr_info)

            # Skip if PR should be skipped or already in changelog
            if (
                parsed_title.get("scope", "unknown") == "skip"
                or f"#{pr_info.number}]" in content
            ):
                continue

            pr_type = parsed_title.get("type", "unknown")
            if pr_type == "feat":
                insert_content_index = content.find("### What", unreleased_index + 1)
            elif pr_type == "docs":
                insert_content_index = content.find(
                    "### Documentation improvements", unreleased_index + 1
                )
            elif pr_type == "break":
                breaking_changes = True
                insert_content_index = content.find(
                    "### Incompatible changes", unreleased_index + 1
                )
            elif pr_type in {"ci", "fix", "refactor"}:
                insert_content_index = content.find(
                    "### Other changes", unreleased_index + 1
                )
            else:
                unknown_changes = True
                insert_content_index = unreleased_index

            pr_reference = _format_pr_reference(
                pr_info.title, pr_info.number, pr_info.html_url
            )

            content = _insert_entry_no_desc(
                content,
                pr_reference,
                insert_content_index,
            )

            next_header_index = content.find("## ", unreleased_index + 1)
            next_header_index = (
                next_header_index if next_header_index != -1 else len(content)
            )

        if unknown_changes:
            content = _insert_entry_no_desc(
                content,
                "### Unknown changes",
                unreleased_index,
            )

        if not breaking_changes:
            content = _insert_entry_no_desc(
                content,
                "None",
                content.find("### Incompatible changes", unreleased_index + 1),
            )

        # Finalize content update
        file.seek(0)
        file.write(content)
        file.truncate()
    return True


def _insert_entry_no_desc(
    content: str, pr_reference: str, unreleased_index: int
) -> str:
    """Insert a changelog entry for a pull request with no specific description."""
    insert_index = content.find("\n", unreleased_index) + 1
    content = (
        content[:insert_index] + "\n" + pr_reference + "\n" + content[insert_index:]
    )
    return content


def _bump_minor_version(tag: Tag) -> Optional[str]:
    """Bump the minor version of the tag."""
    match = re.match(r"v(\d+)\.(\d+)\.(\d+)", tag.name)
    if match is None:
        return None
    major, minor, _ = [int(x) for x in match.groups()]
    # Increment the minor version and reset patch version
    new_version = f"v{major}.{minor + 1}.0"
    return new_version


def main() -> None:
    """Update changelog using the descriptions of PRs since the latest tag."""
    # Initialize GitHub Client with provided token (as argument)
    gh_api = Github(argv[1])
    repo, latest_tag = _get_latest_tag(gh_api)
    if not latest_tag:
        print("No tags found in the repository.")
        return

    shortlog, prs = _get_pull_requests_since_tag(repo, latest_tag)
    if _update_changelog(prs):
        new_version = _bump_minor_version(latest_tag)
        if not new_version:
            print("Wrong tag format.")
            return
        _add_shortlog(new_version, shortlog)
        print("Changelog updated succesfully.")


if __name__ == "__main__":
    main()
