#!/bin/bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"/../

# Setup environment variables for development
./devtool/setup-envs.sh

# Remove caches
./dev/rm-caches.sh

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

# Use `uv` to install project dependencies from lockfile
uv sync --frozen --all-extras --all-groups
