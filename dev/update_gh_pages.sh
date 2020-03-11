#!/bin/bash
# Rebuild the docs, commit and push the changes if any to GitHub

set -e
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Cleanup previous .tmp directory if present
rm -rf .tmp

# Update docs
cd $ROOT/doc
make html

# Checkout gh-pages in .tmp dir
cd $ROOT
mkdir .tmp
cp -r .git .tmp/
cd .tmp
git checkout gh-pages
git pull

# Copy new updates docs into .tmp
cd $ROOT
cp -r doc/build/html/* .tmp/

# Commit new content
cd $ROOT/.tmp
git add .
git commit -m "Update gh-pages"
git push

# Cleanup
cd $ROOT
rm -rf .tmp
