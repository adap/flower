Use Baselines
=============

.. warning::
  We are changing the way we structure the Flower baselines. While we complete the transition to the new format, you can still find the existing baselines and use them: `baselines (old) <https://github.com/adap/flower/tree/main/baselines/flwr_baselines>`_.
  Currently, you can make use of baselines for `FedAvg <https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/fedavg_mnist>`_, `FedOpt <https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/adaptive_federated_optimization>`_,  and `LEAF-FEMNIST <https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist>`_.

  The documentation below has been updated to reflect the new way of using Flower baselines.

Structure
---------

All baselines are available in the directory `baselines <https://github.com/adap/flower/blob/main/baselines>`_ with each baseline directory being fully self-contained in terms of source code. In addition, each baseline uses its very own Python environment as designed by the contributors of such baseline in order to replicate the experiments in the paper. Each baseline directory contains the following structure: 

.. code-block:: shell

    flower/baselines/<baseline_name>/
                          ├── README.md
                          ├── pyproject.toml
                          └── <baseline_name>
                                      ├── *.py # several .py files including main.py and __init__.py
                                      └── conf
                                            └── *.yaml # one or more Hydra config files

Please note that some baselines might include additional files (e.g. a :code:`requirements.txt`) or a hierarchy of :code:`.yaml` files for `Hydra <https://hydra.cc/>`_.


Setting up your machine
-----------------------

.. note::
  Flower baselines are designed to run on Ubuntu 22.04. While a GPU is not required to run the baselines, some of the more computationally demanding ones do benefit from GPU acceleration.

Common to all baselines is `Poetry <https://python-poetry.org/docs/>`_, a tool to manage Python dependencies. Baselines also make use of `Pyenv <https://github.com/pyenv/pyenv>`_. You'll need to install both on your system before running a baseline. What follows is a step-by-step guide on getting :code:`pyenv` and :code:`Poetry` installed on your system.

Let's begin by installing :code:`pyenv`. We'll be following the standard procedure. Please refer to the `pyenv docs <https://github.com/pyenv/pyenv#installation>`_ for alternative ways of installing it.

.. code-block:: bash

  # first install a few packages needed later for pyenv
  sudo apt install build-essential zlib1g-dev libssl-dev libsqlite3-dev \
            libreadline-dev libbz2-dev libffi-dev liblzma-dev

  # now clone pyenv into your home directory (this is the default way of installing pyenv)
  git clone https://github.com/pyenv/pyenv.git ~/.pyenv

  # Then add pyenv to your path by adding the below to your .bashrc/.zshrc
  export PYENV_ROOT="$HOME/.pyenv"
  command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init -)"

Verify your installation by opening a new terminal and

.. code-block:: bash

  # check python versions available
  pyenv versions
  # * system (...)  # <-- it should just show one

Then you can proceed and install any version of Python. Most baselines currently use Python 3.10.6, so we'll be installing that one.

.. code-block:: bash

  pyenv install 3.10.6
  # this will take a little while
  # once done, you should see that that version is available
  pyenv versions
  # system
  # * 3.10.6  # <-- you just installed this

Now that we have :code:`pyenv` installed, we are ready to install :code:`poetry`. Installing Poetry can be done from a single command:

.. code-block:: bash

  curl -sSL https://install.python-poetry.org | python3 -

  # add to path by putting this line at the end of your .zshrc/.bashrc
  export PATH="$HOME/.local/bin:$PATH"


To install Poetry from source, to customise your installation, or to further integrate Poetry with your shell after installation, please check `the Poetry documentation <https://python-poetry.org/docs/#installation>`_.

Using a Flower Baseline
-----------------------

To use Flower Baselines you need first to install :code:`pyenv` and :code:`Poetry`, then:

1. Clone the flower repository

.. code-block:: bash

  git clone https://github.com/adap/flower.git && cd flower

2. Navigate inside the directory of the baseline you'd like to run
3. Follow the :code:`[Environment Setup]` instructions in the :code:`README.md`. In most cases this will require you to just do:

.. code-block:: bash

    poetry install

4. Run the baseline as indicated in the :code:`[Running the Experiments]` section in the :code:`README.md` or in the `[Expected Results]` section to reproduce the experiments in the paper.
