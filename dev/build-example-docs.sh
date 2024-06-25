#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

ROOT=`pwd`
INDEX=$ROOT/examples/doc/source/index.rst
INSERT_LINE=6

initial_text=$(cat <<-END
.. toctree::
  :maxdepth: 1
  :caption: References
END
)

table_body="\\
.. list-table:: \\
   :widths: 50 15 15 15\\
   :header-rows: 1\\
  \\
   * - Title \\
     - Framework\\
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

  framework=$(echo "$metadata" | sed -n 's/framework: //p' | sed 's/\[//g; s/\]//g')

  table_entry="\\
   * - \`$title <$1.html>\`_\\
     - $framework\\
     - $dataset\\
     - $labels\\
    \\
.. BASELINES_TABLE_ENTRY\
  "
}

copy_markdown_files () {
  for file in $1/*.md; do
    # Copy the README into the source of the Example docs as the name of the example
    if [[ $(basename "$file") = "README.md" ]]; then
      cp $file $ROOT/examples/doc/source/$1.md 2>&1 >/dev/null
    else
      # If the example contains other markdown files, copy them to the source of the Example docs
      cp $file $ROOT/examples/doc/source/$(basename "$file") 2>&1 >/dev/null
    fi
  done
}

add_gh_button () {
  gh_text="[<img src=\"_static/view-gh.png\" alt=\"View on GitHub\" width=\"200\"/>](https://github.com/adap/flower/blob/main/examples/$1)"
  readme_file="$ROOT/examples/doc/source/$1.md"

  if ! grep -Fq "$gh_text" "$readme_file"; then
    awk -v text="$gh_text" '
    /^# / && !found {
      print $0 "\n" text;
      found=1;
      next;
    }
    { print }
    ' "$readme_file" > tmpfile && mv tmpfile "$readme_file"
  fi
}

copy_images () {
  if [ -d "$1/_static" ]; then
    cp $1/_static/**.{jpg,png,jpeg} $ROOT/examples/doc/source/_static/ 2>/dev/null || true
  fi
}

add_to_index () {
  echo "  $1" >> $INDEX
}

add_single_entry () {
  # Copy markdown files to correct folder
  copy_markdown_files $1

  # Add button linked to GitHub
  add_gh_button $1
  
  # Copy all images of the _static folder into the examples
  # docs static folder
  copy_images $1

  # Insert the name of the example into the index file
  add_to_index $1
}

add_all_entries () {
  cd $ROOT/examples
  # Iterate through each folder in examples/
  for d in $(printf '%s\n' */ | sort -V); do
    # Add entry based on the name of the folder
    example=${d%/}

    if [[ $example != doc ]]; then
      add_single_entry $example
    fi
  done
}

# Clean up before starting
rm -f $ROOT/examples/doc/source/*.md
rm -f $INDEX

# Create empty index file
touch $INDEX

echo "Flower Examples Documentation" >> $INDEX
echo "-----------------------------" >> $INDEX
echo "" >> $INDEX
echo ".. BASELINES_TABLE_ANCHOR" >> $INDEX

! sed -i '' -e "s/.. BASELINES_TABLE_ANCHOR/$table_body/" $INDEX

cd $ROOT/examples
# Iterate through each folder in examples/
for d in $(printf '%s\n' */ | sort -V); do
  # Add entry based on the name of the folder
  example=${d%/}

  if [[ $example != doc ]]; then

    # Copy markdown files to correct folder
    copy_markdown_files $example

    # Add button linked to GitHub
    add_gh_button $example
    
    # Copy all images of the _static folder into the examples
    # docs static folder
    copy_images $example

    add_table_entry $example
    ! sed -i '' -e "s/.. BASELINES_TABLE_ENTRY/$table_entry/" $INDEX
  fi
done

echo "" >> $INDEX
echo "$initial_text" >> $INDEX
echo "" >> $INDEX

add_all_entries

echo "" >> $INDEX

