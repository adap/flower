#!/bin/bash
set -e -o pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"/../

NIGHTLY_DATE=$(date '+%Y%m%d')
a=(${1//./ }) # replace points, split into array
((a[1]++))    # increment minor
NIGHTLY_VERSION="${a[0]}.${a[1]}.${a[2]}"

jq -c '.readme_updates[]' <<<"$2" | while read -r i; do
    README_PATH=$(echo "$i" | jq -r .path)
    UPDATE=$(echo "$i" | jq -r .update)
    sed "/<!-- version_latest -->/a $UPDATE" -i "$README_PATH"
    sed '/<!-- version_nightly -->/{n;s/.*/- `nightly`, `<version>.dev<YYYYMMDD>` e.g. `'$NIGHTLY_VERSION'.dev'$NIGHTLY_DATE'`/}' -i "$README_PATH"
done
