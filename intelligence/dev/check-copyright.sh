#!/usr/bin/env bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../


EXPECTED_NOTICE=$(cat << 'EOF'
// Copyright 2025 Flower Labs GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
EOF
)

cd ts

# Check if the script is run with the "--fix" option.
FIX_MODE=false
if [ "$1" == "--fix" ]; then
  FIX_MODE=true
fi

EXIT_STATUS=0

# Helper function to check if the file already has the full notice.
has_complete_notice() {
  header=$(head -n 14 "$1")
  if [ "$header" = "$EXPECTED_NOTICE" ]; then
    return 0
  else
    return 1
  fi
}

# Process all TypeScript files in the src/ directory.
while IFS= read -r -d '' file; do
  if has_complete_notice "$file"; then
    continue
  fi
  
  if [ "$FIX_MODE" = true ]; then
    # Read the first line of the file.
    first_line=$(head -n 1 "$file")
    correct_first_line='// Copyright 2025 Flower Labs GmbH. All Rights Reserved.'
    
    if [ "$first_line" != "$correct_first_line" ]; then
      # The file doesn't even start with the first line, so prepend the full notice.
      new_content="${EXPECTED_NOTICE}"$'\n\n'"$(cat "$file")"
    else
      # The file starts with the correct first line but is missing the rest.
      # Instead of replacing the first 14 lines, we insert the missing lines (lines 2-14)
      # right after the first line.
      missing_notice=$(echo "$EXPECTED_NOTICE" | tail -n +2)
      rest=$(tail -n +2 "$file")
      new_content="${first_line}"$'\n'"${missing_notice}"$'\n'"${rest}"
    fi

    echo "$new_content" > "$file"
    echo "Fixed: $file"
  else
    echo "Missing or incomplete copyright notice in: $file"
    EXIT_STATUS=1
  fi
done < <(find src -type f -name '*.ts' -print0)

exit $EXIT_STATUS
