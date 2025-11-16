#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Remove caches
./dev/rm-caches.sh

# Upgrade/install spcific versions of `pip`, `setuptools`, and `poetry`
python -m pip install -U pip==25.3
python -m pip install -U setuptools==80.9.0
python -m pip install -U poetry==2.1.3

# Use `poetry` to install project dependencies
python -m poetry install
