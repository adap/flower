#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Testing the changelog
git fetch --quiet origin main
CHANGED=`git diff --name-only origin/main -- doc/source/ref-changelog.md`

if [ -z "$CHANGED" ]
then
    echo "doc/source/ref-changelog.md was correctly updated."
else
    echo "doc/source/ref-changelog.md must be updated"
    exit 1
fi
