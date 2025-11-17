#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

version=${1:-3.10.19}

# Delete caches, venv, and lock file
./dev/rm-caches.sh
./dev/venv-delete.sh $version
[ ! -e poetry.lock ] || rm poetry.lock

# Recreate
./dev/venv-create.sh $version
./dev/bootstrap.sh
