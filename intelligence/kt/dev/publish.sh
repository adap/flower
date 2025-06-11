#!/usr/bin/env bash
set -e

# === Go to project root ===
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../

# === CONFIGURATION ===
PROPERTIES_FILE="./gradle.properties"

getProp() {
  grep "^$1=" "$PROPERTIES_FILE" | cut -d'=' -f2 | tr -d '\r'
}

GROUP_ID=$(getProp "GROUP_ID")
ARTIFACT_ID=$(getProp "ARTIFACT_ID")
VERSION=$(getProp "VERSION_NAME")

CENTRAL_API_URL="https://central.sonatype.com/api/v1/publisher/upload?publishingType=AUTOMATIC"
ZIP_FILE="${ARTIFACT_ID}-${VERSION}.zip"
REPO_DIR="$HOME/.m2/repository/$(echo $GROUP_ID | tr '.' '/')/$ARTIFACT_ID/$VERSION"

# === AUTH ===
if [[ -z "$MVN_CENTRAL_USER" || -z "$MVN_CENTRAL_PASSWORD" ]]; then
  echo "❌ Error: MVN_CENTRAL_USER and/or MVN_CENTRAL_PASSWORD not set."
  exit 1
fi

AUTH_TOKEN=$(echo -n "${MVN_CENTRAL_USER}:${MVN_CENTRAL_PASSWORD}" | base64)

# === BUILD AND SIGN ===
echo "Building and signing AAR..."
./gradlew clean :flwr:publishToMavenLocal

if [[ ! -d "$REPO_DIR" ]]; then
  echo "❌ Failed to locate Maven artifact output at $REPO_DIR"
  exit 1
fi

# === GENERATE CHECKSUMS ===
echo "Generating .md5 and .sha1 checksums..."
pushd "$REPO_DIR" > /dev/null

for f in *.{aar,pom,jar,module}; do
  [[ -f "$f" ]] || continue
  if [[ ! -f "$f.md5" ]]; then
    if command -v md5sum &> /dev/null; then
      md5sum "$f" | awk '{print $1}' > "$f.md5"
    else
      md5 -q "$f" > "$f.md5"
    fi
  fi
  if [[ ! -f "$f.sha1" ]]; then
    if command -v sha1sum &> /dev/null; then
      sha1sum "$f" | awk '{print $1}' > "$f.sha1"
    else
      shasum -a 1 "$f" | awk '{print $1}' > "$f.sha1"
    fi
  fi
done
popd > /dev/null

# === CREATE ZIP ===
echo "Packaging ZIP..."
REPO_ROOT="$HOME/.m2/repository"
RELATIVE_PATH="$(echo "$GROUP_ID" | tr '.' '/')/$ARTIFACT_ID/$VERSION"
pushd "$REPO_ROOT" > /dev/null
zip -r "$OLDPWD/$ZIP_FILE" "$RELATIVE_PATH"

popd > /dev/null

# === PUBLISH ===
echo "Uploading to Maven Central (AUTOMATIC)..."
RESPONSE=$(curl -s -X POST "$CENTRAL_API_URL" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -F "bundle=@$ZIP_FILE;type=application/octet-stream")

echo "RAW RESPONSE:"
echo "$RESPONSE"
