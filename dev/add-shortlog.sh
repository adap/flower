#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

tags=$(git tag --sort=-creatordate)
new_version=$(echo "$tags" | sed -n '1p')
old_version=$(echo "$tags" | sed -n '2p')

shortlog=$(git shortlog "$old_version".."$new_version" -s | grep -vEi '(\(|\[)bot(\)|\])' | awk '{printf "%s\`%s %s\`",sep,$2,$3; sep=", "} END{print ""}')
thanks="\n### Thanks to our contributors\n\nWe would like to give our special thanks to all the contributors who made the new version of Flower possible (in \`git shortlog\` order):\n\n$shortlog"

awk -v version="$new_version" -v text="$thanks" \
    '{print} $0 ~ "## " version {print text}' doc/source/ref-changelog.md > temp.md && mv temp.md doc/source/ref-changelog.md

