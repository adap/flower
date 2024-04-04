#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

ROOT=`pwd`
INDEX=$ROOT/examples/README.md
INSERT_LINE=6

rm -f $INDEX
touch $INDEX

echo "# Flower Examples Documentation" >> $INDEX
echo "" >> $INDEX
echo "\`\`\`{toctree}" >> $INDEX
echo "---" >> $INDEX
echo "maxdepth: 1" >> $INDEX
echo "---" >> $INDEX

rm -f examples/doc/source/*.md

cd examples/
for d in $(printf '%s\n' */ | sort -V); do
  example=${d%/}

  if [[ $example != doc ]]; then

    for file in $example/*.md; do
      # For each example, copy the README into the source of the Example docs
      if [[ $(basename "$file") = "README.md" ]]; then
        cp $file $ROOT/examples/doc/source/$example.md 2>&1 >/dev/null
      else
        # If the example contains other markdown files, copy them to the source of the Example docs
        cp $file $ROOT/examples/doc/source/$(basename "$file") 2>&1 >/dev/null
      fi
    done

    gh_text="[<img src=\"_static/view-gh.png\" alt=\"View on GitHub\" width=\"200\"/>](https://github.com/adap/flower/blob/main/examples/$example)"
    readme_file="$ROOT/examples/doc/source/$example.md"

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
    
    # For each example, copy all images of the _static folder into the examples
    # docs static folder
    [ -d "$example/_static" ] && {
      cp $example/_static/**.{jpg,png,jpeg} $ROOT/examples/doc/source/_static/ 2>/dev/null || true
    }
    # For each example, insert the name of the example into the index file
    (echo $INSERT_LINE; echo a; echo $example; echo .; echo wq) | ed $INDEX 2>&1 >/dev/null

  fi
done

echo "\`\`\`" >> $INDEX

cp $INDEX $ROOT/examples/doc/source/index.md
