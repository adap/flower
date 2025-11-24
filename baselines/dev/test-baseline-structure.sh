#!/bin/bash

set -e
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <baseline-name=directory-of-the-baseline>"
    exit 1
fi
baseline_dir=$1
baseline_path=$(pwd)/$1
echo "Testing baseline under the path: $baseline_path"

# Specify the exceptions to the structure requirement
declare -a structure_exceptions=()

# List of require to check
declare -a required_files=("client_app.py" "dataset.py" "model.py" "server_app.py" "strategy.py" "utils.py")

# Check and pass the test if the baseline directory is in the list of exceptions
for exception in "${structure_exceptions[@]}"; do
  if [[ "$baseline_dir" == "$exception" ]]; then
    exit 0
  fi
done

# If the baseline directory is not in the list of exceptions
for file in "${required_files[@]}"; do
  if [ -f "$baseline_path/$baseline_dir/$file" ]; then
    echo "$file exists."
  else
    echo "$file does not exist."
    exit 1
  fi
done

echo "The structure of $baseline_dir is correct."
