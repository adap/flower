#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

tags=$(git tag --sort=-v:refname)
new_version=$(echo "$tags" | sed -n '1p')
old_version=$(echo "$tags" | sed -n '2p')

shortlog=$(git shortlog "$old_version".."$new_version" -s | grep -vEi '(\(|\[)bot(\)|\])' | awk '{name = substr($0, index($0, $2)); printf "%s`%s`", sep, name; sep=", "} END {print ""}')

token="<!---TOKEN_$new_version-->"
thanks="\n### Thanks to our contributors\n\nWe would like to give our special thanks to all the contributors who made the new version of Flower possible (in \`git shortlog\` order):\n\n$shortlog $token"

# Check if the token exists in the markdown file
if ! grep -q "$token" doc/source/ref-changelog.md; then
    # If the token does not exist in the markdown file, append the new content after the version
    awk -v version="$new_version" -v text="$thanks" \
        '{print} $0 ~ "## " version {print text}' doc/source/ref-changelog.md > temp.md && mv temp.md doc/source/ref-changelog.md
else
    # If the token exists, replace the line containing the token with the new shortlog
    awk -v token="$token" -v newlog="$shortlog $token" '{ if ($0 ~ token) print newlog; else print $0 }' doc/source/ref-changelog.md > temp.md && mv temp.md doc/source/ref-changelog.md
fi
