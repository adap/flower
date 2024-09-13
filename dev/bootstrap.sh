#!/bin/bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"/../

# Setup environment variables for development
./dev/setup-envs.sh

# Remove caches
./dev/rm-caches.sh

# Upgrade/install spcific versions of `pip`, `setuptools`, and `poetry`
python -m pip install -U pip==24.1.2
python -m pip install -U setuptools==70.3.0
python -m pip install -U poetry==1.7.1

# Use `poetry` to install project dependencies
python -m poetry install --all-extras
