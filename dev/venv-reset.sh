#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Delete caches, venv, and lock file
./dev/rm-caches.sh
./dev/venv-delete.sh
[ ! -e poetry.lock ] || rm poetry.lock

# Recreate
./dev/venv-create.sh
./dev/bootstrap.sh
