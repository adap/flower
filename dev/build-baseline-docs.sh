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

table_body="\\
.. list-table:: \\
   :widths: 15 15 50\\
   :header-rows: 1\\
  \\
   * - Method\\
     - Dataset\\
     - Tags\\
   .. BASELINES_TABLE_ENTRY\\
  "


function add_table_entry ()
{
  # extract lines from markdown file between --- and ---, preserving newlines and store in variable called metadata
  metadata=$(awk '/^---$/{flag=1; next} flag; /^---$/{exit}' $1/README.md)

  # get text after "title:" in metadata using sed
  title=$(echo "$metadata" | sed -n 's/title: //p')

  # get text after "url:" in metadata using sed
  url=$(echo "$metadata" | sed -n 's/url: //p')

  # get text after "labels:" in metadata using sed
  labels=$(echo "$metadata" | sed -n 's/labels: //p' | sed 's/\[//g; s/\]//g')

  # get text after "dataset:" in metadata using sed
  dataset=$(echo "$metadata" | sed -n 's/dataset: //p' | sed 's/\[//g; s/\]//g')

  table_entry="\\
   * - \`$1 <$1.html>\`_\\
     - $dataset\\
     - $labels\\
    \\
.. BASELINES_TABLE_ENTRY\
  "
}


# Create Sphinx table block and header
! sed -i '' -e "s/.. BASELINES_TABLE_ANCHOR/$table_body/" $INDEX

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

    # for each image found, copy
    for img in $image_path
    do
      image_dir=$(dirname $img)

      mkdir -p $ROOT/baselines/doc/source/$image_dir && cp $baseline/$img $_
      images_arr+=("$ROOT/baselines/doc/source/$img")
    done

    if [[ $(grep -L "$baseline" $INDEX) ]]; then

      # For each baseline, insert the name of the baseline into the index file
      echo "  $baseline" >> $INDEX

      # Add entry to the table
      add_table_entry $baseline
      ! sed -i '' -e "s/.. BASELINES_TABLE_ENTRY/$table_entry/" $INDEX

    fi
  fi
done

cd $ROOT/baselines/doc
make html

# Restore everything back to the initial state
git restore source/
rm source/*.md
for image in "${images_arr[@]}"; do
  rm $image
done
