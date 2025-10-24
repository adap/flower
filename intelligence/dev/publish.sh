#!/usr/bin/env bash
set -euo pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../../

# --- CONFIG ---
REPO="adap/flower"
TS_PATH="intelligence/ts/package.json"
KT_PATH="intelligence/kt/gradle.properties"
CODEOWNERS_FILE=".github/CODEOWNERS"
RELEASE_PREFIX="intelligence/v"

# --- UTILITY FUNCTIONS ---
get_ts_version() {
  jq -r '.version' "$TS_PATH"
}

get_kt_version() {
  grep '^VERSION_NAME=' "$KT_PATH" | cut -d'=' -f2
}

get_latest_tag() {
  git fetch --tags >/dev/null 2>&1
  git tag --list "${RELEASE_PREFIX}*" | sort -V | tail -n 1 | sed "s|${RELEASE_PREFIX}||"
}

confirm() {
  read -r -p "$1 [y/N] " response
  [[ "$response" =~ ^[Yy]$ ]]
}

check_admin_rights() {
  local user perm
  user=$(gh api user --jq '.login' 2>/dev/null || echo "")
  if [[ -z "$user" ]]; then
    echo "⚠️ Could not determine GitHub username. Try running: gh auth login"
    return 1
  fi
  echo "👤 Logged in as @$user"
  perm=$(gh api "repos/$REPO/collaborators/$user/permission" --jq '.permission' 2>/dev/null || echo "none")
  if [[ "$perm" == "admin" ]]; then
    echo "✅ User @$user has admin access to $REPO"
    return 0
  else
    echo "❌ User @$user does not have admin rights to $REPO (permission: $perm)"
    return 1
  fi
}

# --- MAIN LOGIC ---

echo "🔍 Checking current SDK versions..."
TS_VERSION=$(get_ts_version)
KT_VERSION=$(get_kt_version)

if [ "$TS_VERSION" != "$KT_VERSION" ]; then
  echo "❌ TypeScript and Kotlin versions mismatch!"
  echo "   TypeScript: $TS_VERSION"
  echo "   Kotlin: $KT_VERSION"
  exit 1
fi

NEW_VERSION="$TS_VERSION"
echo "✅ Both SDKs at version $NEW_VERSION"

LATEST_TAG=$(get_latest_tag)
echo "🔖 Latest tag: v$LATEST_TAG"

# Compare versions to avoid skipping
if [[ $(printf '%s\n' "$LATEST_TAG" "$NEW_VERSION" | sort -V | uniq | wc -l) -eq 1 ]]; then
  echo "❌ Version $NEW_VERSION is already tagged."
  exit 1
elif [[ $(printf '%s\n' "$LATEST_TAG" "$NEW_VERSION" | sort -V | tail -n 1) != "$NEW_VERSION" ]]; then
  echo "❌ Version $NEW_VERSION is not greater than $LATEST_TAG."
  exit 1
elif [[ $(printf '%s\n' "$LATEST_TAG" "$NEW_VERSION" | sort -V | head -n 1) != "$LATEST_TAG" ]]; then
  echo "⚠️ Warning: You might be skipping a version ($LATEST_TAG → $NEW_VERSION)."
  confirm "Do you really want to proceed?" || exit 1
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "(Dry run) Would tag ${RELEASE_PREFIX}${NEW_VERSION}"
  exit 0
fi

# Admin permission check
echo "🔐 Verifying GitHub permissions..."
check_admin_rights || exit 1

# Confirm before tagging
echo "About to tag and push:"
echo "   Branch: main"
echo "   Tag: ${RELEASE_PREFIX}${NEW_VERSION}"
confirm "Proceed with tagging and pushing?" || exit 1

# Perform the release
echo "🚀 Performing release..."
git switch main
git pull
git tag "${RELEASE_PREFIX}${NEW_VERSION}"
git push origin "${RELEASE_PREFIX}${NEW_VERSION}"
echo "✅ Successfully pushed tag intelligence/v${NEW_VERSION}"
