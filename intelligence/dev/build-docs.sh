#!/usr/bin/env bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

cd ts && \
  pnpm build:docs --readme none --name "TypeScript API" && \
  cd ..

cd docs

{
  echo ''
  echo '```{toctree}'
  echo ':hidden:'
  echo ':maxdepth: 2'
  echo ':glob:'
  echo ''
  echo '*/*'
  echo '```'
} | tee -a source/ts-api-ref/index.md

make html
