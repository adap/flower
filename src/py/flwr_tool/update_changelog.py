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
from sys import argv

from github import Github

REPO_NAME = "adap/flower"
CHANGELOG_FILE = "doc/source/ref-changelog.md"
CHANGELOG_SECTION_HEADER = "### Changelog entry"


def _get_latest_tag(gh_api):
    """Retrieve the latest tag from the GitHub repository."""
    repo = gh_api.get_repo(REPO_NAME)
    tags = repo.get_tags()
    return tags[0] if tags.totalCount > 0 else None


def _get_pull_requests_since_tag(gh_api, tag):
    """Get a list of pull requests merged into the main branch since a given tag."""
    repo = gh_api.get_repo(REPO_NAME)
    commits = {commit.sha for commit in repo.compare(tag.commit.sha, "main").commits}
    prs = set()
    for pr_info in repo.get_pulls(
        state="closed", sort="created", direction="desc", base="main"
    ):
        if pr_info.merge_commit_sha in commits:
            prs.add(pr_info)
        if len(prs) == len(commits):
            break
    return prs


def _format_pr_reference(title, number, url):
    """Format a pull request reference as a markdown list item."""
    return f"- **{title.replace('*', '')}** ([#{number}]({url}))"


def _extract_changelog_entry(pr_info):
    """Extract the changelog entry from a pull request's body."""
    if not pr_info.body:
        return None, "general"

    entry_match = re.search(
        f"{CHANGELOG_SECTION_HEADER}(.+?)(?=##|$)", pr_info.body, re.DOTALL
    )
    if not entry_match:
        return None, None

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
        (token for token, marker in token_markers.items() if marker in entry_text), None
    )

    return entry_text, token


def _update_changelog(prs):
    """Update the changelog file with entries from provided pull requests."""
    with open(CHANGELOG_FILE, "r+", encoding="utf-8") as file:
        content = file.read()
        unreleased_index = content.find("## Unreleased")

        if unreleased_index == -1:
            print("Unreleased header not found in the changelog.")
            return

        # Find the end of the Unreleased section
        next_header_index = content.find("##", unreleased_index + 1)
        next_header_index = (
            next_header_index if next_header_index != -1 else len(content)
        )

        for pr_info in prs:
            pr_entry_text, category = _extract_changelog_entry(pr_info)

            # Skip if PR should be skipped or already in changelog
            if category == "skip" or f"#{pr_info.number}]" in content:
                continue

            pr_reference = _format_pr_reference(
                pr_info.title, pr_info.number, pr_info.html_url
            )

            # Process based on category
            if category in ["general", "baselines", "examples", "sdk", "simulations"]:
                entry_title = _get_category_title(category)
                content = _update_entry(
                    content,
                    entry_title,
                    pr_info,
                    unreleased_index,
                    next_header_index,
                )

            elif pr_entry_text:
                content = _insert_new_entry(
                    content, pr_info, pr_reference, pr_entry_text, unreleased_index
                )

            else:
                content = _insert_entry_no_desc(content, pr_reference, unreleased_index)

            next_header_index = content.find("##", unreleased_index + 1)
            next_header_index = (
                next_header_index if next_header_index != -1 else len(content)
            )

        # Finalize content update
        file.seek(0)
        file.write(content)
        file.truncate()

    print("Changelog updated.")


def _get_category_title(category):
    """Get the title of a changelog section based on its category."""
    headers = {
        "general": "General improvements",
        "baselines": "General updates to Flower Baselines",
        "examples": "General updates to Flower Examples",
        "sdk": "General updates to Flower SDKs",
        "simulations": "General updates to Flower Simulations",
    }
    return headers.get(category, "")


def _update_entry(
    content, category_title, pr_info, unreleased_index, next_header_index
):
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


def _insert_new_entry(content, pr_info, pr_reference, pr_entry_text, unreleased_index):
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


def _insert_entry_no_desc(content, pr_reference, unreleased_index):
    """Insert a changelog entry for a pull request with no specific description."""
    insert_index = content.find("\n", unreleased_index) + 1
    content = (
        content[:insert_index] + "\n" + pr_reference + "\n" + content[insert_index:]
    )
    return content


def main():
    """Update changelog using the descriptions of PRs since the latest tag."""
    # Initialize GitHub Client with provided token (as argument)
    gh_api = Github(argv[1])
    latest_tag = _get_latest_tag(gh_api)
    if not latest_tag:
        print("No tags found in the repository.")
        return

    prs = _get_pull_requests_since_tag(gh_api, latest_tag)
    _update_changelog(prs)


if __name__ == "__main__":
    main()
