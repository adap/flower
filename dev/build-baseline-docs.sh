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

! grep ":caption: References" $INDEX && echo "$initial_text" >> $INDEX && echo "" >> $INDEX

rm -f "baselines/doc/source/*.md"

cd $ROOT/baselines/

images_arr=()

for d in $(printf '%s\n' */ | sort -V); do

  # Select directories
  baseline=${d%/}

  if ! [[ "$baseline" =~ ^(baseline_template|dev|doc|flwr_baselines)$ ]]; then

    # For each baseline, copy the README into the source of the Baselines docs
    cp $baseline/README.md $ROOT/baselines/doc/source/$baseline.md 2>&1 >/dev/null

    # Copy the images to the same folder in source
    image_path=$(cd $baseline && find  . -type f -regex ".*\.png" | cut -c 3-)
    image_dir=$(dirname $image_path)

    mkdir -p $ROOT/baselines/doc/source/$image_dir && cp $baseline/$image_path $_

    images_arr+=("$ROOT/baselines/doc/source/$image_path")

    if [[ $(grep -L "$baseline" $INDEX) ]]; then

      # For each baseline, insert the name of the baseline into the index file
      echo "  $baseline" >> $INDEX

    fi
  fi
done

cd $ROOT/baselines/doc
make html

# Restore everything back to the initial state
git restore source/
rm source/*.md
for image in "${images_arr[@]}"; do
  rm image
done
