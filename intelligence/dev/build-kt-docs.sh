#!/usr/bin/env bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

cd kt && \
  ./gradlew dokkaGfm && \
  mkdir -p ../docs/source/kt-api-ref && \
  cp -r flwr/build/dokka/dokkaGfm/flwr/ai.flower.intelligence/* ../docs/source/kt-api-ref/ && \
  cd ../docs/source/kt-api-ref && \

  find . -type f -name "*.md" | while read -r file; do
    # Step 1: Replace leading //[flwr](...)/[ai.flower.intelligence](...) with [Kotlin API](../index.md)
    sed -E 's|^\/\/\[flwr\]\([^)]+\)/\[ai\.flower\.intelligence\]\([^)]+\)|[Kotlin API](../index.md)|' "$file" > "$file.tmp" && mv "$file.tmp" "$file"

    # Step 2: Remove [androidJvm]<br>
    sed -E 's/\[androidJvm\]<br>//g' "$file" > "$file.tmp" && mv "$file.tmp" "$file"

    # Step 3: Remove entire lines that are [androidJvm]\ 
    sed -E '/^\[androidJvm\\\]$/d' "$file" > "$file.tmp" && mv "$file.tmp" "$file"

    # Step 4: Remove all remaining instances of the word "androidJvm"
    sed -E 's/androidJvm//g' "$file" > "$file.tmp" && mv "$file.tmp" "$file"

    # Step 5: Remove empty leftover brackets like []\
    sed -E 's/\[\]\\//g' "$file" > "$file.tmp" && mv "$file.tmp" "$file"

    echo "Stripped androidJvm markers from: $file"
  done && \
  # Absolute or relative path to the specific file
  TARGET_FILE="index.md" && \

  # Step 1: Replace heading
  sed -E 's/^# Package-level declarations$/# Kotlin API/' "$TARGET_FILE" > "$TARGET_FILE.tmp" && mv "$TARGET_FILE.tmp" "$TARGET_FILE" && \

  # Step 2: Convert markdown table to bullet list (inside '## Types' section)
  awk '
    BEGIN { in_types=0 }
    /^## Types/ { print; in_types=1; next }
    in_types && /^$/ { print ""; next }
    in_types && /^\| *Name *\|/ { next }
    in_types && /^\| *---/ { next }
    in_types && /^\|/ {
      # Extract first link only from the row
      line = $0
      start = index(line, "[")
      end = index(line, ")")
      if (start > 0 && end > start) {
        link_text = substr(line, start + 1, index(line, "]") - start - 1)
        link_url = substr(line, index(line, "(") + 1, end - index(line, "(") - 1)
        print "- [" link_text "](" link_url ")"
      }
      next
    }
    { in_types=0; print }
  ' "$TARGET_FILE" > "$TARGET_FILE.tmp" && mv "$TARGET_FILE.tmp" "$TARGET_FILE" && \

  echo "Transformed: $TARGET_FILE" && \
  cd ../..

{
  echo ''
  echo '```{toctree}'
  echo ':hidden:'
  echo ':maxdepth: 2'
  echo ':glob:'
  echo ''
  echo '*/*'
  echo '```'
} | tee -a source/kt-api-ref/index.md
