#!/bin/sh
set -e

if [ -n "$1" ]; then
  DOC_VERSION="$1"
fi

if [ -z "$DOC_VERSION" ]; then
  echo "DOC_VERSION is required (e.g. main or 1.26)" >&2
  exit 1
fi

# Move to the docs directory
cd "$(git rev-parse --show-toplevel)/framework/docs"

# Clean previous builds
rm -rf build

# Get a list of languages based on the folders in locales
languages="en"
for lang_dir in locales/*; do
  if [ -d "$lang_dir" ]; then
    languages="$languages $(basename "$lang_dir")"
  fi
done

current_version="$DOC_VERSION"
export current_version

for current_language in $languages; do
  export current_language
  sphinx-build -b html source/ build/html/${current_version}/${current_language} -A lang=True -D language=${current_language}
done
