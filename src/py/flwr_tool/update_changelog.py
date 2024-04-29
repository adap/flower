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
"""This module is used to update the changelog."""


import re
from datetime import date
from sys import argv
from typing import Dict, Optional, Set, Tuple

from github import Github
from github.Tag import Tag
from github.Repository import Repository
from github.PullRequest import PullRequest

REPO_NAME = "adap/flower"
CHANGELOG_FILE = "doc/source/ref-changelog.md"
CHANGELOG_SECTION_HEADER = "### Changelog entry"
TYPES_PATTERN = r"(build|ci|docs|feat|fix|perf|refactor|style|test)"
SCOPES_PATTERN = r"(framework|datasets|examples|baselines|sdk|skip)"


def _get_latest_tag(gh_api: Github) -> Tuple[Repository, Optional[Tag]]:
    """Retrieve the latest tag from the GitHub repository."""
    repo = gh_api.get_repo(REPO_NAME)
    tags = repo.get_tags()
    return repo, tags[0] if tags.totalCount > 0 else None


def _add_shorlog(new_version: str, shortlog: str) -> None:
    """Update the markdown file with the new version information or update existing logs."""
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

    with open(CHANGELOG_FILE, "w", encoding="utf-8") as file:
        for line in content:
            if token in line:
                file.write(f"{shortlog} {token}\n")
            elif "## Unreleased" in line:
                file.write(f"## {new_version} ({current_date})\n{entry}\n")
            else:
                file.write(line)


def _get_pull_requests_since_tag(
    repo: Repository, tag: Tag
) -> Tuple[str, Set[PullRequest]]:
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
    return f"- **{title.replace('*', '')}** ([#{number}]({url}))"


def _extract_changelog_entry(
    pr_info: PullRequest,
) -> Tuple[Optional[str], Dict[str, str]]:
    """Extract the changelog entry from a pull request's body."""
    pattern = (
        rf"^({TYPES_PATTERN})\(({SCOPES_PATTERN})(?::([^:]+))?\): "
        r"([A-Z][^\.\n]*(?:\.(?=[^\.\n]))*[^\.\n]*)$"
    )

    # Use regex search to find matches
    match = re.search(pattern, pr_info.title)
    if match:
        # Extract components from the regex groups
        pr_type = match.group(1)
        pr_scope = match.group(3)
        pr_sub_scope = match.group(5)  # Correctly capture optional sub-scope
        pr_subject = match.group(
            6
        )  # Capture subject starting with uppercase and no terminal period
        return None, {
            "type": pr_type,
            "scope": pr_scope,
            "sub_scope": pr_sub_scope,
            "subject": pr_subject,
        }

    if not pr_info.body:
        return None, {
            "type": "unknown",
            "scope": "unknown",
            "sub_scope": "unknown",
            "subject": "unknown",
        }

    entry_match = re.search(
        f"{CHANGELOG_SECTION_HEADER}(.+?)(?=##|$)", pr_info.body, re.DOTALL
    )
    if not entry_match:
        return None, {
            "type": "unknown",
            "scope": "unknown",
            "sub_scope": "unknown",
            "subject": "unknown",
        }

    entry_text = entry_match.group(1).strip()

    # Remove markdown comments
    entry_text = re.sub(r"<!--.*?-->", "", entry_text, flags=re.DOTALL).strip()

    token_markers = {
        "general": "<general>",
        "skip": "<skip>",
        "baselines": "<baselines>",
        "examples": "<examples>",
        "sdk": "<sdk>",
        "simulations": "<simulations>",
    }

    # Find the token based on the presence of its marker in entry_text
    token = next(
        (token for token, marker in token_markers.items() if marker in entry_text),
        "unknown",
    )

    return entry_text, {
        "type": "unknown",
        "scope": token,
        "sub_scope": "unknown",
        "subject": "unknown",
    }


def _update_changelog(prs: Set[PullRequest]) -> None:
    """Update the changelog file with entries from provided pull requests."""
    with open(CHANGELOG_FILE, "r+", encoding="utf-8") as file:
        content = file.read()
        unreleased_index = content.find("## Unreleased")

        if unreleased_index == -1:
            print("Unreleased header not found in the changelog.")
            return

        # Find the end of the Unreleased section
        next_header_index = content.find("## ", unreleased_index + 1)
        next_header_index = (
            next_header_index if next_header_index != -1 else len(content)
        )

        content_index = content.find("### What", unreleased_index + 1)

        for pr_info in prs:
            pr_entry_text, parsed_title = _extract_changelog_entry(pr_info)

            if "scope" not in parsed_title:
                continue

            category = parsed_title["scope"]

            # Skip if PR should be skipped or already in changelog
            if category == "skip" or f"#{pr_info.number}]" in content:
                continue

            pr_reference = _format_pr_reference(
                pr_info.title, pr_info.number, pr_info.html_url
            )

            # Process based on category
            if category in [
                "framework",
                "general",
                "baselines",
                "examples",
                "sdk",
                "simulations",
            ]:
                entry_title = _get_category_title(category)
                content = _update_entry(
                    content,
                    entry_title,
                    pr_info,
                    content_index,
                    next_header_index,
                )

            elif pr_entry_text:
                content = _insert_new_entry(
                    content, pr_info, pr_reference, pr_entry_text, content_index
                )

            else:
                content = _insert_entry_no_desc(content, pr_reference, content_index)

            next_header_index = content.find("## ", unreleased_index + 1)
            next_header_index = (
                next_header_index if next_header_index != -1 else len(content)
            )

        # Finalize content update
        file.seek(0)
        file.write(content)
        file.truncate()


def _get_category_title(category: str) -> str:
    """Get the title of a changelog section based on its category."""
    headers = {
        "framework": "Framework improvements",
        "general": "General improvements",
        "baselines": "General updates to Flower Baselines",
        "examples": "General updates to Flower Examples",
        "sdk": "General updates to Flower SDKs",
        "simulations": "General updates to Flower Simulations",
    }
    return headers.get(category, "")


def _update_entry(
    content: str,
    category_title: str,
    pr_info: PullRequest,
    unreleased_index: int,
    next_header_index: int,
) -> str:
    """Update a specific section in the changelog content."""
    if (
        section_index := content.find(
            category_title, unreleased_index, next_header_index
        )
    ) != -1:
        newline_index = content.find("\n", section_index)
        closing_parenthesis_index = content.rfind(")", unreleased_index, newline_index)
        updated_entry = f", [{pr_info.number}]({pr_info.html_url})"
        content = (
            content[:closing_parenthesis_index]
            + updated_entry
            + content[closing_parenthesis_index:]
        )
    else:
        new_section = (
            f"\n- **{category_title}** ([#{pr_info.number}]({pr_info.html_url}))\n"
        )
        insert_index = content.find("\n", unreleased_index) + 1
        content = content[:insert_index] + new_section + content[insert_index:]
    return content


def _insert_new_entry(
    content: str,
    pr_info: PullRequest,
    pr_reference: str,
    pr_entry_text: str,
    unreleased_index: int,
) -> str:
    """Insert a new entry into the changelog."""
    if (existing_entry_start := content.find(pr_entry_text)) != -1:
        pr_ref_end = content.rfind("\n", 0, existing_entry_start)
        updated_entry = (
            f"{content[pr_ref_end]}\n, [{pr_info.number}]({pr_info.html_url})"
        )
        content = content[:pr_ref_end] + updated_entry + content[existing_entry_start:]
    else:
        insert_index = content.find("\n", unreleased_index) + 1

        # Split the pr_entry_text into paragraphs
        paragraphs = pr_entry_text.split("\n")

        # Indent each paragraph
        indented_paragraphs = [
            "    " + paragraph if paragraph else paragraph for paragraph in paragraphs
        ]

        # Join the paragraphs back together, ensuring each is separated by a newline
        indented_pr_entry_text = "\n".join(indented_paragraphs)

        content = (
            content[:insert_index]
            + "\n"
            + pr_reference
            + "\n\n"
            + indented_pr_entry_text
            + "\n"
            + content[insert_index:]
        )
    return content


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
    _update_changelog(prs)

    new_version = _bump_minor_version(latest_tag)
    if not new_version:
        print("Wrong tag format.")
        return

    _add_shorlog(new_version, shortlog)

    print("Changelog updated succesfully.")


if __name__ == "__main__":
    main()
