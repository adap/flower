#!/bin/bash

old_version=$2
new_version=$3

changes=$(awk "/$new_version/{flag=1;next}/$old_version/{flag=0}flag" doc/source/ref-changelog.md)
shortlog=$(git shortlog "$old_version".."$new_version" -s | grep -vEi '(\(|\[)bot(\)|\])' | awk '{printf "%s%s %s",sep,$2,$3; sep=", "} END{print ""}')
thanks="### Thanks to our contributors\nWe would like to give our special thanks to all the contributors who made the new version of Flower possible (in git shortlog order):\n\n$shortlog\n"

complete="$thanks$changes"

echo complete
