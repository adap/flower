#!/bin/bash
set -e

if [ ! -d $HOME/.pyenv ]
then
    # Install pyenv with the virtualenv plugin
    curl https://pyenv.run | bash 

    # To add the config to the right file (depends on the shell used)
    rcfile=$HOME/.$(basename $SHELL)rc

    # Add $PYENV_ROOT environmnet variable
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> $rcfile
    # Add pyenv to $PATH
    echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> $rcfile

    # Init pyenv with the shell
    echo 'eval "$(pyenv init -)"' >> $rcfile
    echo 'eval "$(pyenv virtualenv-init -)"' >> $rcfile
    
    exec "$SHELL"
else
    # If pyenv is already installed, check for a newer version

    # If the pyenv-update plugin isn't installed do the update manually
    if [ ! -d $HOME/.pyenv/plugins/pyenv-update ]
    then
        git -C $HOME/.pyenv pull
        git -C $HOME/.pyenv/plugins/pyenv-virtualenv pull
    else
        pyenv update
    fi
fi

export PYENV_ROOT="$HOME/.pyenv"
# Create the virtual environment for Flower
$( dirname "${BASH_SOURCE[0]}" )/venv-create.sh

# Install the dependencies inside the virtual environment
$( dirname "${BASH_SOURCE[0]}" )/bootstrap.sh
