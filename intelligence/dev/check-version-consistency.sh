#!/usr/bin/env bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../../

PACKAGE_JSON="./intelligence/ts/package.json"
GRADLE_PROPERTIES="./intelligence/kt/gradle.properties"

package_version=$(node -p "require('$PACKAGE_JSON').version")
gradle_version=$(grep '^VERSION_NAME=' "$GRADLE_PROPERTIES" | cut -d '=' -f2 | xargs)

echo "package.json version: $package_version"
echo "gradle.properties VERSION_NAME: $gradle_version"

if [ "$package_version" != "$gradle_version" ]; then
  echo "❌ Version mismatch!"
  exit 1
else
  echo "✅ Versions match."
fi
