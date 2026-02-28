#!/bin/bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"/../

# Setup environment variables for development
./devtool/setup-envs.sh

# Remove caches
./dev/rm-caches.sh

# Upgrade/install specific versions of `pip`, `setuptools`, and `poetry`
python -m pip install -U pip==26.0.1
python -m pip install -U setuptools==82.0.0
python -m pip install -U poetry==2.3.2

# Use `poetry` to install project dependencies
poetry install --all-extras
