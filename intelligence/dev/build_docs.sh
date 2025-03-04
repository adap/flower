#!/usr/bin/env bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

cd ts && \
  pnpm build:docs --readme none --name "TypeScript API" && \
  cd ..

cd docs

make html
