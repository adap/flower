"""Assigns a reviewer to a PR based on its label."""

import json
import random
import subprocess
import sys


def _get_pr_labels(event_path):
    with open(event_path, encoding="utf-8") as f:
        event_data = json.load(f)
    labels = [label["name"] for label in event_data["pull_request"]["labels"]]
    return labels


def _parse_codereviewers(file_path):
    reviewers_map = {}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            label = parts[0]
            reviewers = [part.lstrip("@") for part in parts[1:]]
            reviewers_map[label] = reviewers
    return reviewers_map


def _assign_reviewer(label, reviewers, repo, pr_number):
    if not reviewers:
        print(f"No reviewers to assign for label {label}")
        return
    random_reviewer = random.choice(reviewers)
    try:
        subprocess.run(
            [
                "gh",
                "api",
                "--method",
                "POST",
                "-H",
                "'Accept: application/vnd.github+json'",
                "-H",
                "'X-GitHub-Api-Version: 2022-11-28'",
                f"/repos/{repo}/pulls/{pr_number}/requested_reviewers",
                "-f",
                f"'reviewers[]={random_reviewer}'",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error assigning reviewer @{random_reviewer} for label {label}: {e}")
        sys.exit(1)
    else:
        print(f"Assigned reviewer @{random_reviewer} for label {label}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python assign_reviewers.py <CODEREVIEWERS file path>"
            "<REPO> <PR_NUMBER> <GH_EVENT file path>"
        )
        sys.exit(1)

    codereviewers_path = sys.argv[1]
    repo = sys.argv[2]
    pr_number = sys.argv[3]
    event_path = sys.argv[4]

    pr_labels = _get_pr_labels(event_path)
    reviewers_map = _parse_codereviewers(codereviewers_path)

    for label in pr_labels:
        if label in reviewers_map:
            _assign_reviewer(label, reviewers_map[label], repo, pr_number)
        else:
            print(f"No reviewers found for label {label}")
