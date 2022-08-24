Using Baselines
===============


Structure
---------

All baselines are available in the directory `baselines <https://github.com/adap/flower/blob/main/baselines>`_. This directory has two different files:

.. code-block:: shell

    - pyproject.toml
    - requirements.txt

Both files contain all the information about required Python packages (libraries, frameworks, ...) and their versions. You can install each library separately by using :code: `pip install` or you can use Poetry and run code:`poetry install` in the directory where you find the :code:`pyproject.toml` file. After installing all requirements, you can start to run your baseline.

Go to the baseline that you want to execute. The directories and files are structured so that you can first find the paper with their unique identifier such that, for example, :code:`FedProx` refers to the paper "Federated Optimization in Heterogeneous Networks". The :code:`fedprox` section contains all available experiments from that paper.

.. code-block:: shell   

    flower/baselines/flwr_baselines/publications/
    |---name_of_publication/
    |---|---experiment1/
    |---|---experiment2/

The experiment area contains a :code:`README.md` covering the corresponding paper, its abstract, and goal as well as a detailed description of how to run the baseline. Please use the :code:`README.md` to see how to execute each individual baseline.


Available Baselines
-------------------

The following table lists all currently available baselines and the corresponding papers. If you want to add a new baseline or experiment, please check the `Contributing Baselines <https://flower.dev/docs/contributing-baselines.html>`_ section. 

.. list-table::
    :widths: 20 30 50
    :header-rows: 1

    * - Paper
      - Experiment
      - Directory 
    * - `FedBN <https://arxiv.org/pdf/2102.07623.pdf>`_
      - convergence rate
      - :code:`flower/baselines/flwr_baselines/publications/fedbn/convergence_rate`
    * - `FedOpt <https://arxiv.org/pdf/2003.00295.pdf>`_
      - sparse gradient task
      - :code:`flower/baselines/flwr_baselines/publications/fedopt/sparse_gradient_task`
    * - `FedProx <https://arxiv.org/pdf/1812.06127.pdf>`_
      - system heterogeneity
      - :code:`flower/baselines/flwr_baselines/publications/fedprox/system_heterogeneity`
