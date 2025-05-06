#!/usr/bin/env bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# Check if exactly one argument is provided and is one of the allowed values.
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 [patch|minor|major]"
  exit 1
fi

version_type="$1"

if [[ "$version_type" != "patch" && "$version_type" != "minor" && "$version_type" != "major" ]]; then
  echo "Invalid version type: $version_type"
  echo "Usage: $0 [patch|minor|major]"
  exit 1
fi

# Bump version in the ts package using pnpm
cd ts

# Get the old version from package.json
old_version=$(node -p "require('./package.json').version")
echo "Old version: $old_version"

pnpm version "$version_type"

# Get the new version from package.json
new_version=$(node -p "require('./package.json').version")
echo "New version: $new_version"

sed -i.bak "s/release = \".*\"/release = \"$new_version\"/g" ../docs/source/conf.py

# Update the version in src/constants.ts
sed -i.bak "s/export const VERSION = '.*';/export const VERSION = '$new_version';/g" src/constants.ts

# Update the version in src/flowerintelligence.test.ts
sed -i.bak "s/VERSION: '.*',/VERSION: '$new_version',/g" src/flowerintelligence.test.ts

# Remove the temporary backup files created by sed
rm src/constants.ts.bak src/flowerintelligence.test.ts.bak ../docs/source/conf.py.bak

# Update all examples/*/package.json files to set "@flwr/flwr" to the old version.
for pkg in examples/*/package.json; do
  echo "Updating $pkg with version ^$old_version"
  sed -i.bak "s/\"@flwr\/flwr\": \"\^[0-9]*\.[0-9]*\.[0-9]*\"/\"@flwr\/flwr\": \"^$old_version\"/g" "$pkg"
  rm "$pkg.bak"
done

cd ..

echo "Version updated successfully!"
