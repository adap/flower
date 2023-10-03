#!/bin/bash

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

tags=$(git tag --sort=-creatordate)
new_version=$(echo "$tags" | sed -n '1p')
old_version=$(echo "$tags" | sed -n '2p')

awk -v start="$new_version" -v end="$old_version" '
    $0 ~ start {flag=1; next}
    $0 ~ end {flag=0}
    flag && !printed && /^$/ {next}  # skip the first blank line
    flag && !printed {printed=1}
    flag' doc/source/ref-changelog.md
