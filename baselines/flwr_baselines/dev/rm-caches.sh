#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

find flwr_baselines -type d -name __pycache__ -exec rm -r {} \+
rm -rf .mypy_cache
rm -rf .pytest_cache
