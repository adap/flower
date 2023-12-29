#!/bin/bash

# This script duplicates the `baseline_template` directory and changes its name
# to the one you specify when running this script. That name is also used to
# rename the subdirectory inside your new baseline directory as well as to set
# the Python package name that Poetry will build

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

template="baseline_template"
name=$1

# copying directory
echo "Copying '$template' and renaming it to '$name'"
cp -r $template $name

# renaming sub-directory
echo "Renaming sub-directory as '$name'"
mv $name/$template $name/$name

# adjusting package name in pyproject.toml
cd $name
if [[ "$OSTYPE" == "darwin"* ]]; then
  sed -i '' -e "s/<BASELINE_NAME>/$name/" pyproject.toml
else
  sed -i -e "s/<BASELINE_NAME>/$name/" pyproject.toml
fi

echo "!!! Your directory for your baseline '$name' is ready."
