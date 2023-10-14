Use Baselines
=============

.. warning::
  We are changing the way we structure the Flower baselines. While we complete the transition to the new format, you can still find the existing baselines and use them: `baselines (old) <https://github.com/adap/flower/tree/main/baselines/flwr_baselines>`_.
  Currently, you can make use of baselines for `FedAvg <https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/fedavg_mnist>`_, `FedProx <https://github.com/adap/flower/tree/main/baselines/fedprox>`_, `FedOpt <https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/adaptive_federated_optimization>`_, `FedBN <https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/fedbn/convergence_rate>`_, and `LEAF-FEMNIST <https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/leaf/femnist>`_.

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


Using a Baseline
----------------

Common to all baselines is `Poetry <https://python-poetry.org/docs/>`_, a tool to manage Python dependencies. You'll need to install it on your system before running a baseline. For Linux and macOS, installing Poetry can be done from a single command:

.. code-block:: bash

  curl -sSL https://install.python-poetry.org | python3 -

To install Poetry on a different OS, to customise your installation, or to further integrate Poetry with your shell after installation, please check `the Poetry documentation <https://python-poetry.org/docs/#installation>`_.


1. Navigate inside the directory of the baseline you'd like to run
2. Follow the :code:`[Environment Setup]` instructions in the :code:`README.md`. In most cases this will require you to just do:

.. code-block:: bash

    poetry install

3. Run the baseline as indicated in the :code:`[Running the Experiments]` section in the :code:`README.md` 
