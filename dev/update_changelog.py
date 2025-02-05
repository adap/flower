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
import subprocess
from concurrent.futures import ThreadPoolExecutor
import time

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
from github.Commit import Commit
from github.Tag import Tag
from github.NamedUser import NamedUser

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

PR_TYPE_TO_SECTION = {
    "feat": "### New features",
    "docs": "### Documentation improvements",
    "break": "### Incompatible changes",
    "ci": "### Other changes",
    "fix": "### Other changes",
    "refactor": "### Other changes",
}


def _get_latest_tag(gh_api: Github) -> tuple[Repository, Optional[str]]:
    """Retrieve the latest tag from the GitHub repository."""
    repo = gh_api.get_repo(REPO_NAME)
    latest_tag = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"], 
        stdout=subprocess.PIPE, 
        text=True
    ).stdout.strip()
    return repo, latest_tag


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


def _git_commits_since_tag(repo: Repository, tag: str) -> set[Commit]:
    """Get a set of commits since a given tag."""
    # Get SHA hashes of commits since the tag
    result = subprocess.run(
        ["git", "log", "--pretty=format:%H", f"{tag}..origin/main"],
        stdout=subprocess.PIPE,
        text=True,
    )
    shas = set(result.stdout.splitlines())
    
    # Fetch GitHub commits based on the SHA hashes
    with ThreadPoolExecutor(max_workers=15) as executor:
        commits = list(executor.map(repo.get_commit, shas))

    return commits


def _get_contributors_from_commits(api: Github, commits: set[Commit]) -> set[str]:
    """Get a set of contributors from a set of commits."""
    # Get authors and co-authors from the commits
    authors: set[NamedUser] = set()
    coauthor_names: set[str] = set()
    coauthor_pattern = r"Co-authored-by:\s*(.+?)\s*<"
    start = time.time()
    # authors = {author for author in authors if author.name and "[bot]" not in author.name}

    def retrieve(commit: Commit) -> None:
        if commit.author.name is None:
            return
        if "[bot]" in commit.author.name:
            return
        authors.add(commit.author)
        print("A: ", time.time() - start)
        # Find co-authors in the commit message
        if matches := re.findall(coauthor_pattern, commit.commit.message):
            coauthor_names.update(name for name in matches)
        print("B: ", time.time() - start)

    with ThreadPoolExecutor(max_workers=15) as executor:
        executor.map(retrieve, commits)
    
    print("Get info from commits:", time.time() - start)

    # Remove repeated usernames
    contributors = set(author.name for author in authors if author.name)
    coauthor_names.difference_update(contributors)
    coauthor_names.difference_update(author.login for author in authors)

    # Get full names of the GitHub usernames
    print("Coauthors", coauthor_names)
    with ThreadPoolExecutor(max_workers=5) as executor:
        names = list(executor.map(lambda x: api.get_user(x).name, coauthor_names))
    contributors.update(name for name in names if name)
    return contributors


def _get_pull_requests_since_tag(
    api: Github, repo: Repository, tag: str
) -> tuple[str, set[PullRequest]]:
    """Get a list of pull requests merged into the main branch since a given tag."""
    prs = set()

    start = time.time()
    commits = _git_commits_since_tag(repo, tag)
    print("Time to get commits: ", time.time() - start)
    
    start = time.time()
    contributors = _get_contributors_from_commits(api, commits)
    print("Time to get contributors: ", time.time() - start)

    start = time.time()
    commit_shas = {commit.sha for commit in commits}
    for pr_info in repo.get_pulls(
        state="closed", sort="updated", direction="desc", base="main"
    ):
        if pr_info.merge_commit_sha in commit_shas:
            prs.add(pr_info)
        if len(prs) == len(commit_shas):
            break
    print("Time to get PRs: ", time.time() - start)

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


def _update_changelog(prs: set[PullRequest], tag: str) -> bool:
    """Update the changelog file with entries from provided pull requests."""
    with open(CHANGELOG_FILE, "r+", encoding="utf-8") as file:
        content = file.read()
        unreleased_index = content.find("## Unreleased")

        # Find the end of the Unreleased section
        end_index = content.find(f"## {tag}", unreleased_index + 1)

        if unreleased_index == -1:
            print("Unreleased header not found in the changelog.")
            return False

        for pr_info in prs:
            print("End index", end_index)
            parsed_title = _extract_changelog_entry(pr_info)

            # Skip if the PR is already in changelog
            if f"#{pr_info.number}]" in content:
                continue

            # Find section to insert
            pr_type = parsed_title.get("type", "unknown")
            section = PR_TYPE_TO_SECTION.get(pr_type, "### Unknown changes")
            insert_index = content.find(section, unreleased_index, end_index)

            # Add section if not exist
            if insert_index == -1:
                content = _insert_entry_no_desc(
                    content,
                    section,
                    unreleased_index,
                )
                insert_index = content.find(section, unreleased_index, end_index)

            pr_reference = _format_pr_reference(
                pr_info.title, pr_info.number, pr_info.html_url
            )
            content = _insert_entry_no_desc(
                content,
                pr_reference,
                insert_index,
            )

            # Find the end of the Unreleased section
            end_index = content.find(f"## {tag}", end_index)

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


def _bump_minor_version(tag: str) -> Optional[str]:
    """Bump the minor version of the tag."""
    match = re.match(r"v(\d+)\.(\d+)\.(\d+)", tag)
    if match is None:
        return None
    major, minor, _ = [int(x) for x in match.groups()]
    # Increment the minor version and reset patch version
    new_version = f"v{major}.{minor + 1}.0"
    return new_version


def _fetch_origin() -> None:
    """Fetch the latest changes from the origin."""
    subprocess.run(["git", "fetch", "origin"])


def main() -> None:
    """Update changelog using the descriptions of PRs since the latest tag."""
    # Initialize GitHub Client with provided token (as argument)
    gh_api = Github(argv[1])
    
    # Fetch the latest changes from the origin
    _fetch_origin()

    start = time.time()
    repo, latest_tag = _get_latest_tag(gh_api)
    if not latest_tag:
        return

    shortlog, prs = _get_pull_requests_since_tag(gh_api, repo, latest_tag)

    start = time.time()
    if _update_changelog(prs, latest_tag):
        new_version = _bump_minor_version(latest_tag)
        if not new_version:
            print("Wrong tag format.")
            return
        _add_shortlog(new_version, shortlog)
        print("Changelog updated succesfully.")


if __name__ == "__main__":
    main()
