#!/usr/bin/env bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../../

docstrfmt intelligence/docs/source -x intelligence/docs/source/_templates/autosummary
