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

parse_semver() {
  # echo "<major> <minor> <patch>" or fail
  local v="$1"
  if [[ "$v" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]}"
  else
    echo "❌ Invalid semver: $v" >&2
    return 1
  fi
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

# echo "🔎 Verifying branch..."
# CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
# if [[ "$CURRENT_BRANCH" != "main" ]]; then
#   echo "❌ You are on branch '$CURRENT_BRANCH', not 'main'."
#   echo "   Please switch to main before running this script."
#   exit 1
# fi
#
# echo "🔄 Fetching latest main commits from origin..."
# git fetch origin main >/dev/null 2>&1

# --- Let user pick a commit from origin/main ---
echo ""
echo "📜 Latest commits on origin/main:"
COMMITS=$(git log origin/main --pretty=format:"%h %s" -n 10)
echo "$COMMITS" | nl -w2 -s'. '

echo ""
read -r -p "Select a commit number to tag (default: 1 for latest): " choice
choice=${choice:-1}

SELECTED_HASH=$(echo "$COMMITS" | sed -n "${choice}p" | awk '{print $1}')
if [[ -z "$SELECTED_HASH" ]]; then
  echo "❌ Invalid selection."
  exit 1
fi
SELECTED_COMMIT_MSG=$(git log -1 --pretty=format:"%s" "$SELECTED_HASH")

echo "🪶 Selected commit: $SELECTED_HASH - $SELECTED_COMMIT_MSG"
confirm "Check out this commit to perform validation?" || exit 1

# --- Checkout the commit temporarily ---
echo "📦 Checking out $SELECTED_HASH..."
git checkout --quiet "$SELECTED_HASH"

# --- Perform your checks here ---
echo "🔍 Running validation on commit $SELECTED_HASH..."
TS_VERSION=$(get_ts_version)
KT_VERSION=$(get_kt_version)

if [ "$TS_VERSION" != "$KT_VERSION" ]; then
  echo "❌ TypeScript and Kotlin versions mismatch!"
  echo "   TypeScript: $TS_VERSION"
  echo "   Kotlin: $KT_VERSION"
  git checkout --quiet main
  exit 1
fi

NEW_VERSION="$TS_VERSION"
echo "✅ Both SDKs at version $NEW_VERSION"

LATEST_TAG=$(get_latest_tag)
echo "🔖 Latest tag: v$LATEST_TAG"

# Compare versions
if [[ $(printf '%s\n' "$LATEST_TAG" "$NEW_VERSION" | sort -V | uniq | wc -l) -eq 1 ]]; then
  echo "❌ Version $NEW_VERSION is already tagged."
  git checkout --quiet main
  exit 1
elif [[ $(printf '%s\n' "$LATEST_TAG" "$NEW_VERSION" | sort -V | tail -n 1) != "$NEW_VERSION" ]]; then
  echo "❌ Version $NEW_VERSION is not greater than $LATEST_TAG."
  git checkout --quiet main
  exit 1
elif [[ $(printf '%s\n' "$LATEST_TAG" "$NEW_VERSION" | sort -V | head -n 1) != "$LATEST_TAG" ]]; then
  echo "⚠️  Warning: You might be skipping a version ($LATEST_TAG → $NEW_VERSION)."
  confirm "Do you really want to proceed?" || { git checkout --quiet main; exit 1; }
fi

# Fine-grained “skip” detection
read -r LMAJ LMIN LPAT < <(parse_semver "$LATEST_TAG")
read -r NMAJ NMIN NPAT < <(parse_semver "$NEW_VERSION")

warn_skip() {
  echo "⚠️  Warning: This looks like a skipped version ($LATEST_TAG → $NEW_VERSION)."
  confirm "Do you really want to proceed?" || { git checkout --quiet main; exit 1; }
}

echo "${LMAJ}.${LMIN}.${LPAT} → ${NMAJ}.${NMIN}.${NPAT}"

# Same major/minor → expect patch+1
if (( NMAJ == LMAJ && NMIN == LMIN )); then
  if   (( NPAT == LPAT + 1 )); then : # OK patch bump
  elif (( NPAT >  LPAT + 1 )); then warn_skip
  else
    echo "❌ Patch version must increase (expected ${LMAJ}.${LMIN}.$((LPAT+1)))."
    git checkout --quiet main
    exit 1
  fi

# Same major, next minor → expect .0 patch
elif (( NMAJ == LMAJ && NMIN == LMIN + 1 )); then
  if (( NPAT == 0 )); then : # OK minor bump
  else
    warn_skip
  fi

# Next major → expect .0.0
elif (( NMAJ == LMAJ + 1 )); then
  if (( NMIN == 0 && NPAT == 0 )); then : # OK major bump
  else
    warn_skip
  fi

# Anything else is a bigger jump → warn
else
  warn_skip
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "(Dry run) Would tag ${RELEASE_PREFIX}${NEW_VERSION}"
  git checkout --quiet main
  exit 0
fi

# Admin permission check echo "🔐 Verifying GitHub permissions..." 
check_admin_rights || { git checkout --quiet main; exit 1; }

# --- Confirm tagging ---
confirm "Proceed to tag ${RELEASE_PREFIX}${NEW_VERSION} at this commit?" || {
  git checkout --quiet main
  exit 1
}

# --- Tag and push ---
echo "🚀 Tagging release..."
git tag -a "${RELEASE_PREFIX}${NEW_VERSION}" "$SELECTED_HASH" -m "Release $NEW_VERSION: $SELECTED_COMMIT_MSG"
git push origin "${RELEASE_PREFIX}${NEW_VERSION}"
echo "✅ Successfully pushed tag intelligence/v${NEW_VERSION} at commit $SELECTED_HASH"

# --- Restore main ---
echo "↩️ Restoring main branch..."
git checkout --quiet main
echo "✅ Back on main"
