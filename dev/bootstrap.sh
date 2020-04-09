#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Upgrade/install spcific versions of `pip`, `setuptools`, and `poetry`
python -m pip install -U pip==20.0.2
python -m pip install -U setuptools==45.2.0
python -m pip install -U poetry==1.0.5

# Use `poetry` to install project dependencies
python -m poetry install --extras "examples-tensorflow ops"
