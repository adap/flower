Run examples on AWS
========================

Intro
-----

This guide will change significantly in the future as we will make it easier
to run on AWS without the need to actually do all these steps manually. As thats
still a bit in the future follow this guide for now:

Setup
-----

1. Start an instance with an Ubuntu 18.04 AMI on AWS and ssh into it.
2. Execute the following commands to setup the project.

.. code-block:: bash

    sudo apt update && sudo apt -y upgrade

    sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

    curl https://pyenv.run | bash

    git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv

    pyenv install 3.7.6

    # Install
    git clone https://github.com/adap/flower.git

    cd flower
    ./dev/venv-create.sh
    ./dev/bootstrap.sh

3. If you want to run on multiple instances you will need to replicate this setup on multiple instances.

Outlook
-------
We are working on automating this process so in the near future you will only need to setup your local AWS
cli correctly and the rest will happen automatically.
