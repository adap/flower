#!/bin/bash
set -euo pipefail

cd framework

files=(
  "pyproject.toml"
  "devtool/pyproject.toml"
)

comment_taplo() {
  sed -i "s/^\(\s*taplo\s*=.*\)$/#\1/" "$1"
}

uncomment_taplo() {
  sed -i "s/^#\(\s*taplo\s*=.*\)/\1/" "$1"
}

# Comment out taplo from pyproject.toml and devtool/pyproject.toml.
# This prevents version conflicts with taplo built from maturin and
# is only required for devcontainer builds.
for f in "${files[@]}"; do
  comment_taplo "$f"
done

# Prevent Poetry from creating a venv, since we are using a DevContainer and
# therefore don't care if dependencies are installed into your system
# environment.
sudo python -m poetry config virtualenvs.create false
sudo python -m poetry install --all-extras

# Restore taplo lines in "files"
for f in "${files[@]}"; do
  uncomment_taplo "$f"
done
