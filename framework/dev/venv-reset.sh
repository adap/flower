#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../../

version=${1:-3.9.20}

# Delete caches, venv, and lock file
./framework/dev/rm-caches.sh
./framework/dev/venv-delete.sh $version
[ ! -e poetry.lock ] || rm poetry.lock

# Recreate
./framework/dev/venv-create.sh $version
./framework/dev/bootstrap.sh
