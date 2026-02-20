#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

version=${1:-3.10.19}

# Delete caches, venv, and lock file
./dev/rm-caches.sh
./devtool/venv-delete.sh $version
[ ! -e poetry.lock ] || rm poetry.lock

# Recreate
./devtool/venv-create.sh $version
./devtool/bootstrap.sh
