#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

version=${1:-3.10.19}

pyenv uninstall -f baselines-$version
