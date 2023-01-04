#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Remove caches
./dev/rm-caches.sh

# Upgrade/install spcific versions of `pip`, `setuptools`, and `poetry`
python -m pip install -U pip==22.3.1
python -m pip install -U setuptools==65.6.3
python -m pip install -U poetry==1.1.15

# Use `poetry` to install project dependencies
python -m poetry install
