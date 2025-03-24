#!/bin/bash

# Ensure errors are caught
set -e

# Activate pyenv and the virtual environment
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Activate the virtualenv
pyenv activate flower-3.10.13

# Run the chat command with all passed args
mlx_lm.chat "$@"
