#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

version=${1:-3.7.12}

# Check if the directory does not exist and if so, install python
[[ ! -d $PYENV_ROOT/versions/$version ]] && pyenv install $version

pyenv virtualenv $version flower-$version
echo flower-$version > .python-version
