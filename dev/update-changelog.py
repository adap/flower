import re
from sys import argv
from github import Github

# Constants
REPO_NAME = "adap/flower"  # Replace with your GitHub repo
CHANGELOG_FILE = "doc/source/ref-changelog.md"
CHANGELOG_SECTION_HEADER = "## Changelog entry"


def get_latest_tag(g):
    repo = g.get_repo(REPO_NAME)
    tags = repo.get_tags()
    return tags[0] if tags.totalCount > 0 else None


def get_pull_requests_since_tag(g, tag):
    repo = g.get_repo(REPO_NAME)
    commits = set(
        [commit.sha for commit in repo.compare(tag.commit.sha, "main").commits]
    )
    prs = set()
    for pr in repo.get_pulls(
        state="closed", sort="created", direction="desc", base="main"
    ):
        if pr.merge_commit_sha in commits:
            prs.add(pr)
        if len(prs) == len(commits):
            break
    return prs


def format_pr_reference(title, number, url):
    return f"- **{title}** ([#{number}]({url}))"


def extract_changelog_entry(pr):
    # Extract the changelog entry
    entry_match = re.search(
        f"{CHANGELOG_SECTION_HEADER}(.+?)(?=##|$)", pr.body, re.DOTALL
    )
    if not entry_match:
        return None, "general"

    entry_text = entry_match.group(1).strip()

    # Remove markdown comments
    entry_text = re.sub(r"<!--.*?-->", "", entry_text, flags=re.DOTALL).strip()

    if "<general>" in entry_text:
        return entry_text, "general"
    if "<skip>" in entry_text:
        return entry_text, "skip"
    if "<baselines>" in entry_text:
        return entry_text, "baselines"
    if "<examples>" in entry_text:
        return entry_text, "examples"
    if "<sdk>" in entry_text:
        return entry_text, "sdk"
    if "<simulations>" in entry_text:
        return entry_text, "simulations"

    return entry_text, None


def update_changelog(prs):
    with open(CHANGELOG_FILE, "r+") as file:
        content = file.read()
        unreleased_index = content.find("## Unreleased")

        if unreleased_index == -1:
            print("Unreleased header not found in the changelog.")
            return

        # Find the end of the Unreleased section
        next_header_index = content.find("##", unreleased_index + 1)
        if next_header_index == -1:
            next_header_index = len(content)

        for pr in prs:
            pr_entry_text, token = extract_changelog_entry(pr)

            # Check if the PR number is already in the changelog
            if token == "skip" or f"#{pr.number}]" in content:
                continue

            pr_reference = format_pr_reference(pr.title, pr.number, pr.html_url)

            if token == "general":
                general_index = content.find(
                    "General improvements", unreleased_index, next_header_index
                )
                if general_index != -1:
                    # Find the closing parenthesis before the newline
                    newline_index = content.find("\n", general_index)
                    closing_parenthesis_index = content.rfind(
                        ")", unreleased_index, newline_index
                    )
                    updated_entry = f", [{pr.number}]({pr.html_url})"
                    content = (
                        content[:closing_parenthesis_index]
                        + updated_entry
                        + content[closing_parenthesis_index:]
                    )
                else:
                    # Create a new 'General improvements' section
                    new_section = f"\n- **General improvements** ([#{pr.number}]({pr.html_url}))\n"
                    insert_index = content.find("\n", unreleased_index) + 1
                    content = (
                        content[:insert_index] + new_section + content[insert_index:]
                    )

                # Update next_header_index if necessary
                next_header_index = content.find("##", unreleased_index + 1)
                if next_header_index == -1:
                    next_header_index = len(content)

                continue

            if token == "baselines":
                general_index = content.find(
                    "General updates to Flower Baselines",
                    unreleased_index,
                    next_header_index,
                )
                if general_index != -1:
                    # Find the closing parenthesis before the newline
                    newline_index = content.find("\n", general_index)
                    closing_parenthesis_index = content.rfind(
                        ")", unreleased_index, newline_index
                    )
                    updated_entry = f", [{pr.number}]({pr.html_url})"
                    content = (
                        content[:closing_parenthesis_index]
                        + updated_entry
                        + content[closing_parenthesis_index:]
                    )
                else:
                    # Create a new 'General improvements' section
                    new_section = f"\n- **General updates to Flower Baselines** ([#{pr.number}]({pr.html_url}))\n"
                    insert_index = content.find("\n", unreleased_index) + 1
                    content = (
                        content[:insert_index] + new_section + content[insert_index:]
                    )

                # Update next_header_index if necessary
                next_header_index = content.find("##", unreleased_index + 1)
                if next_header_index == -1:
                    next_header_index = len(content)

                continue

            if token == "examples":
                general_index = content.find(
                    "General updates to Flower Examples",
                    unreleased_index,
                    next_header_index,
                )
                if general_index != -1:
                    # Find the closing parenthesis before the newline
                    newline_index = content.find("\n", general_index)
                    closing_parenthesis_index = content.rfind(
                        ")", unreleased_index, newline_index
                    )
                    updated_entry = f", [{pr.number}]({pr.html_url})"
                    content = (
                        content[:closing_parenthesis_index]
                        + updated_entry
                        + content[closing_parenthesis_index:]
                    )
                else:
                    # Create a new 'General improvements' section
                    new_section = f"\n- **General updates to Flower Examples** ([#{pr.number}]({pr.html_url}))\n"
                    insert_index = content.find("\n", unreleased_index) + 1
                    content = (
                        content[:insert_index] + new_section + content[insert_index:]
                    )

                # Update next_header_index if necessary
                next_header_index = content.find("##", unreleased_index + 1)
                if next_header_index == -1:
                    next_header_index = len(content)

                continue

            if token == "sdk":
                general_index = content.find(
                    "General updates to Flower SDKs",
                    unreleased_index,
                    next_header_index,
                )
                if general_index != -1:
                    # Find the closing parenthesis before the newline
                    newline_index = content.find("\n", general_index)
                    closing_parenthesis_index = content.rfind(
                        ")", unreleased_index, newline_index
                    )
                    updated_entry = f", [{pr.number}]({pr.html_url})"
                    content = (
                        content[:closing_parenthesis_index]
                        + updated_entry
                        + content[closing_parenthesis_index:]
                    )
                else:
                    # Create a new 'General improvements' section
                    new_section = f"\n- **General updates to Flower SDKs** ([#{pr.number}]({pr.html_url}))\n"
                    insert_index = content.find("\n", unreleased_index) + 1
                    content = (
                        content[:insert_index] + new_section + content[insert_index:]
                    )

                # Update next_header_index if necessary
                next_header_index = content.find("##", unreleased_index + 1)
                if next_header_index == -1:
                    next_header_index = len(content)

                continue

            if token == "simulations":
                general_index = content.find(
                    "General updates to Flower Simulations",
                    unreleased_index,
                    next_header_index,
                )
                if general_index != -1:
                    # Find the closing parenthesis before the newline
                    newline_index = content.find("\n", general_index)
                    closing_parenthesis_index = content.rfind(
                        ")", unreleased_index, newline_index
                    )
                    updated_entry = f", [{pr.number}]({pr.html_url})"
                    content = (
                        content[:closing_parenthesis_index]
                        + updated_entry
                        + content[closing_parenthesis_index:]
                    )
                else:
                    # Create a new 'General improvements' section
                    new_section = f"\n- **General updates to Flower Simulations** ([#{pr.number}]({pr.html_url}))\n"
                    insert_index = content.find("\n", unreleased_index) + 1
                    content = (
                        content[:insert_index] + new_section + content[insert_index:]
                    )

                # Update next_header_index if necessary
                next_header_index = content.find("##", unreleased_index + 1)
                if next_header_index == -1:
                    next_header_index = len(content)

                continue

            # If entry text length is greater than 0, check for existing entry
            if pr_entry_text:
                existing_entry_start = content.find(pr_entry_text)
                if existing_entry_start != -1:
                    # Find the end of the PR reference line
                    pr_ref_end = content.rfind("\n", 0, existing_entry_start)
                    updated_entry = (
                        f"{content[pr_ref_end]}\n, [{pr.number}]({pr.html_url})"
                    )
                    content = (
                        content[:pr_ref_end]
                        + updated_entry
                        + content[existing_entry_start:]
                    )
                else:
                    # Insert new entry
                    insert_index = content.find("\n", unreleased_index) + 1
                    content = (
                        content[:insert_index]
                        + pr_reference
                        + "\n  "
                        + pr_entry_text
                        + "\n"
                        + content[insert_index:]
                    )
            else:
                # Append PR reference for PRs with no entry text
                insert_index = content.find("\n", unreleased_index) + 1
                content = (
                    content[:insert_index]
                    + "\n"
                    + pr_reference
                    + "\n"
                    + content[insert_index:]
                )

        file.seek(0)
        file.write(content)
        file.truncate()

    print("Changelog updated.")


def main(g):
    latest_tag = get_latest_tag(g)
    if not latest_tag:
        print("No tags found in the repository.")
        return

    prs = get_pull_requests_since_tag(g, latest_tag)
    update_changelog(prs)


if __name__ == "__main__":
    # Initialize GitHub Client
    g = Github(argv[1])
    main(g)
