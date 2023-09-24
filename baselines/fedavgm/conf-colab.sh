#!/bin/bash
# Shellscript to configure the environment on the Google Colab terminal

# fix issue with ctypes on Colab instance
apt-get update
apt-get install -y libffi-dev

# Install pyenv
curl https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# this version is specific to the FedAvgM baseline
pyenv install 3.10.6
pyenv global 3.10.6

# install Poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"

# install and set environment with Poetry
poetry install
poetry shell
