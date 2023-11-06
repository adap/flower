#!/bin/bash
# Generating the docs, rename and move the files such that the meet the convention used in Flower.
# Note that it involves two runs of sphinx-build that are necessary.
# The first run generates the .rst files (and the html files that are discarded)
# The second time it is run after the files are renamed and moved to the correct place. It generates the final htmls.

set -e

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )"  >/dev/null 2>&1 && pwd )"/../doc

# Remove the old docs from source/ref-api
REF_API_DIR="source/ref-api"
if [[ -d "$REF_API_DIR" ]]; then
  rm -r "${REF_API_DIR}"
fi

# Remove the old html files
rm -r build

# Generate new rst files
sphinx-build -M html source build
