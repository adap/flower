#!/bin/bash

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Extract the latest release notes from the changelog, which starts at the line containing 
# the latest version tag and ends one line before the previous version tag.
tags=$(git tag --sort=-v:refname)
new_version=$(echo "$tags" | sed -n '1p')
old_version=$(echo "$tags" | sed -n '2p')

awk '{sub(/<!--.*-->/, ""); print}' doc/source/ref-changelog.md | awk -v start="$new_version" -v end="$old_version" '
    $0 ~ start {flag=1; next}
    $0 ~ end {flag=0}
    flag && !printed && /^$/ {next}  # skip the first blank line
    flag && !printed {printed=1}
    flag'
