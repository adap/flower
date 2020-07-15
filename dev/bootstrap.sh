#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Remove caches
./dev/rm-caches.sh

# Upgrade/install spcific versions of `pip`, `setuptools`, and `poetry`
python -m pip install -U pip==20.1.1
python -m pip install -U setuptools==47.3.1
python -m pip install -U poetry==1.0.9

# Use `poetry` to install project dependencies
python -m poetry install \
  --extras "baseline" \
  --extras "examples-pytorch" \
  --extras "examples-tensorflow" \
  --extras "http-logger" \
  --extras "ops"

# Temporary workaround (Poetry 1.0.9 cannot install TensorFlow 2.2.0)
python -m pip install -U tensorflow-cpu==2.2.0
