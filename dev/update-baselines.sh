#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

ROOT=`pwd`
INDEX=$ROOT/baselines/doc/source/index.rst

initial_text=$(cat <<-END
.. toctree::
  :maxdepth: 1
  :caption: References
END
)

echo "$initial_text" >> $INDEX

rm -f "baselines/doc/source/*.md"

cd baselines/
for d in $(printf '%s\n' */ | sort -V); do
  baseline=${d%/}
  # For each baseline, copy the README into the source of the Baselines docs
  ! [[ "$baseline" =~ ^(baseline_template|dev|doc|flwr_baselines)$ ]] && cp $baseline/README.md $ROOT/baselines/doc/source/$baseline.md 2>&1 >/dev/null
  # For each baseline, insert the name of the baseline into the index file
  ! [[ "$baseline" =~ ^(baseline_template|dev|doc|flwr_baselines)$ ]] && ! grep "$baseline" $INDEX && echo "  $baseline" >> $INDEX
done
