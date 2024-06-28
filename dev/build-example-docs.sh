#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

ROOT=`pwd`
INDEX=$ROOT/examples/doc/source/index.rst
INSERT_LINE=6

table_text=$(cat <<-END
.. toctree::
  :maxdepth: 1
  :caption: Projects
END
)

function add_table_entry ()
{
  local example="$1"
  local label="$2"
  local table_var="$3"

  # extract lines from markdown file between --- and ---, preserving newlines and store in variable called metadata
  metadata=$(awk '/^---$/{flag=1; next} flag; /^---$/{exit}' $example/README.md)

  # get text after "title:" in metadata using sed
  title=$(echo "$metadata" | sed -n 's/title: //p')

  # get text after "labels:" in metadata using sed
  labels=$(echo "$metadata" | sed -n 's/labels: //p' | sed 's/\[//g; s/\]//g')

  # get text after "dataset:" in metadata using sed
  dataset=$(echo "$metadata" | sed -n 's/dataset: //p' | sed 's/\[//g; s/\]//g')

  framework=$(echo "$metadata" | sed -n 's/framework: //p' | sed 's/\[//g; s/\]//g')

  table_entry="   * - \`$title <$example.html>\`_ \n     - $framework \n     - $dataset \n     - $labels\n\n"

  if [[ "$labels" == *"$label"* ]]; then
    eval "$table_var+=\$table_entry"
    return 0
  fi
  return 1
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

initial_text=$(cat <<-END
Flower Examples Documentation
-----------------------------

Welcome to Flower Examples' documentation. \`Flower <https://flower.ai>\`_ is a friendly federated learning framework.


Join the Flower Community
-------------------------

The Flower Community is growing quickly - we're a friendly group of researchers, engineers, students, professionals, academics, and other enthusiasts.

.. button-link:: https://flower.ai/join-slack
    :color: primary
    :shadow:

    Join us on Slack


Quickstart Examples
-------------------

Flower Quickstart Examples are a collection of demo project that show how you can use Flower in combination with other existing frameworks or technologies.

END
)

echo "$initial_text" >> $INDEX

# Table headers
quickstart_table="\n.. list-table::\n   :widths: 50 15 15 15\n   :header-rows: 1\n\n   * - Title\n     - Framework\n     - Dataset\n     - Tags\n\n"
comprehensive_table="\n.. list-table::\n   :widths: 50 15 15 15\n   :header-rows: 1\n\n   * - Title\n     - Framework\n     - Dataset\n     - Tags\n\n"
advanced_table="\n.. list-table::\n   :widths: 50 15 15 15\n   :header-rows: 1\n\n   * - Title\n     - Framework\n     - Dataset\n     - Tags\n\n"
other_table="\n.. list-table::\n   :widths: 50 15 15 15\n   :header-rows: 1\n\n   * - Title\n     - Framework\n     - Dataset\n     - Tags\n\n"

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

    # Add entry to the appropriate table
    if ! add_table_entry $example "quickstart" quickstart_table; then
      if ! add_table_entry $example "comprehensive" comprehensive_table; then
        if ! add_table_entry $example "advanced" advanced_table; then
          add_table_entry $example "" other_table
        fi
      fi
    fi
  fi
done

# Add the tables to the index
echo -e "$quickstart_table" >> $INDEX

tmp_text=$(cat <<-END
Comprehensive Examples
----------------------

Comprehensive example allow us to explore certain topics more in-depth and are often associated with a simpler, less detailed, example.

END
)
echo -e "$tmp_text" >> $INDEX

echo -e "$comprehensive_table" >> $INDEX

tmp_text=$(cat <<-END
Advanced Examples
-----------------

Advanced Examples are mostly for users that are both familiar with Federated Learning but also somewhat familiar with Flower\'s main features.

END
)
echo -e "$tmp_text" >> $INDEX

echo -e "$advanced_table" >> $INDEX

tmp_text=$(cat <<-END
Other Examples
--------------

Flower Examples are a collection of example projects written with Flower that explore different domains and features. You can check which examples already exist and/or contribute your own example.

END
)
echo -e "$tmp_text" >> $INDEX

echo -e "$other_table" >> $INDEX

echo "" >> $INDEX
echo "$table_text" >> $INDEX
echo "" >> $INDEX

add_all_entries

echo "" >> $INDEX

