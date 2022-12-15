#!/bin/bash
set -e

# To install pyenv and virtualenv plugin
function install_pyenv(){
        curl https://pyenv.run | bash
}

if [ ! -d $HOME/.pyenv ]
then
    # Install pyenv with the virtualenv plugin
    echo 'Installing pyenv...'
    install_pyenv &>/dev/null

    # To add the config to the right file (depends on the shell used)
    rcfile=$HOME/.$(basename $SHELL)rc

    # Add $PYENV_ROOT environmnet variable
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> $rcfile
    # Add pyenv to $PATH
    echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> $rcfile

    # Init pyenv with the shell
    echo 'eval "$(pyenv init -)"' >> $rcfile
    echo 'eval "$(pyenv virtualenv-init -)"' >> $rcfile
else
	[[ ! $PYENV_ROOT ]] && echo "You must restart your shell for env variables to be set" && exit

    # If pyenv is already installed, check for a newer version
    echo 'Pyenv already installed, updating it...'

    # If the pyenv-update plugin isn't installed do the update manually
    if [ ! -d $HOME/.pyenv/plugins/pyenv-update ]
    then
        git -C $HOME/.pyenv pull &>/dev/null
        git -C $HOME/.pyenv/plugins/pyenv-virtualenv pull &>/dev/null
    else
        pyenv update &>/dev/null
    fi
fi

# Create the virtual environment for Flower baselines
function create_venv(){
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        $( dirname "${BASH_SOURCE[0]}" )/venv-create.sh
}
echo 'Creating the virtual environment for Flower baselines...'
create_venv &>/dev/null

echo "$(tput bold)Virtual env baselines-3.7.12 created, you must now run baselines/dev/bootstrap.sh to install all dependencies.$(tput sgr0)"

exec "$SHELL"

