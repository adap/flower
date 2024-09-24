# Setting up your RaspberryPi

> This guide assumes you have a fresh install of Ubuntu Server (either 22.04 or 23.10) and that you have successfully `ssh`-ed into your device.

## Setting up your device for Python developemnet

We are going to use [`pyenv`](https://github.com/pyenv/pyenv) to manage different Python versions and to create an environment. First, we need to install some system dependencies

```bash
sudo apt update
# the last package is needed for whisper
sudo apt install build-essential zlib1g-dev libssl-dev libsqlite3-dev libreadline-dev libbz2-dev libffi-dev liblzma-dev libsndfile1
```

Create Python environment with `pyenv`:

```bash

# Ensure you have installed pyenv, else do the below:
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Install python 3.9+
pyenv install 3.9.17

# Install pyenv virtual env plugin
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
# Restart your shell
exec "$SHELL"

# Create the environment
pyenv virtualenv 3.9.17 flower-whisperer
```

## Installing the dependencies for Whisper+Flower

With our environmnet ready, let's install the dependencies. Please note that at the time of writing, PyTorch 2.0+ won't work properly on `aarm64`. Because of this, we'll be using an earlier version of this package.

```bash
# activate your environment
pyenv activate flower-whisperer

# install pytorch (RPi aren't ready for PyTorch 2.0+ apparently...)
pip install torch==1.13.1
# install rest of requirerments
pip install -r requirements.txt
```
