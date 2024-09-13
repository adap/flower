"""Provide useful common functions."""

import subprocess


def get_git_root():
    """Obtain the root of the git repo."""
    return (
        subprocess.Popen(
            ["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE
        )
        .communicate()[0]
        .rstrip()
        .decode("utf-8")
    )
