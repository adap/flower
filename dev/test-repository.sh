#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Fetch latest main
git fetch --quiet origin main

# Condition evaluated to true if change detected.
if [ ! `git diff --quiet origin/main -- doc/source/ref-changelog.md` ] 
then
    echo "doc/source/ref-changelog.md was correctly updated."
else
    echo "doc/source/ref-changelog.md must be updated"
    exit 1
fi
