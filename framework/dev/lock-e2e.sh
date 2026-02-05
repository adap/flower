#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Override with UV_LOCK_PYTHON if you want a different interpreter path.
python_path="${UV_LOCK_PYTHON:-}"

if [[ -z "$python_path" ]]; then
  if [[ -x "$HOME/.pyenv/versions/3.10.19/bin/python" ]]; then
    python_path="$HOME/.pyenv/versions/3.10.19/bin/python"
  elif command -v python3.10 >/dev/null 2>&1; then
    python_path="$(command -v python3.10)"
  else
    echo "Python 3.10 is required to lock E2E dependencies. Set UV_LOCK_PYTHON." >&2
    exit 1
  fi
fi

while IFS= read -r -d '' pyproject; do
  dir="$(dirname "$pyproject")"
  if [[ "$dir" == "$root_dir/e2e" ]]; then
    continue
  fi
  echo "Locking $dir"
  (cd "$dir" && uv lock --python "$python_path" --no-managed-python --no-python-downloads)
done < <(find "$root_dir/e2e" -maxdepth 2 -name pyproject.toml -print0)
