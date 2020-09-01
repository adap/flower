#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

pyenv virtualenv 3.7.9 flower-3.7.9
echo flower-3.7.9 > .python-version
