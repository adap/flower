#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Remove caches
./dev/rm-caches.sh

# Upgrade/install spcific versions of `pip`, `setuptools`, and `poetry`
python -m pip install -U pip==20.1.1
python -m pip install -U setuptools==46.3.1
python -m pip install -U poetry==1.0.5

# Use `poetry` to install project dependencies
python -m poetry install --extras "benchmark examples-tensorflow ops http-logger"
# Temporary workaround (Poetry 1.0.5 cannot install TensorFlow 2.1.0)
python -m pip install -U tensorflow-cpu==2.1.0
