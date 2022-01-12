Executing Baselines
===================

Structure
---------

All baselines are available in the directory `flower/baselines/flwr_baselines <https://github.com/adap/flower/blob/main/baselines>`_. This directory has two different files:

.. code-block:: shell

    - pyproject.toml
    - requirements.txt

Both files contain all the information about required libraries and their versions. 
You can install each library separately by using :code: `pip install` or you can use poetry and run code:`poetry install` in the directory where you find the :code:`pyproject.toml` file. After installing all requirements you can start to run your baseline.
Go to the baseline that you want to execute. The file system is structured that you can first find the paper with their unique identifier such as FedProx represents the paper "Federated Optimization in Heterogeneous Networks". The :code:`fedprox` section contains all available experiments that the paper is covering.   

.. code-block:: shell   

    flower/baselines/flwr_baselines/publications/
    |---name_of_publication/
    |---|---experiment1/
    |---|---experiment2/

The experiment area contains a :code:`README.md` covering the correspondent paper, its abstract, and goal as well as a detailed description of how to run the baseline.
Please use the :code:`README.md` to execute the baseline.  

Available Baselines
-------------------

The following table lists all baselines and their papers. If you want to add another baseline or experiment, please check the `Contribute Baseline <https://flower.dev/docs/contribute-baseline.html>`_ section. 

.. list-table::
    :widths: 20 30 50
    :header-rows: 1

    * - Paper
      - Experiment
      - Directory 
    * - `FedBN <https://arxiv.org/pdf/2102.07623.pdf>`_
      - convergence rate
      - flower/baselines/flwr_baselines/publications/ fedbn/convergence_rate
    * - `FedOpt <https://arxiv.org/pdf/2003.00295.pdf>`_
      - sparse gradient task
      - flower/baselines/flwr_baselines/publications/ fedopt/sparse_gradient_task
    * - `FedProx <https://arxiv.org/pdf/1812.06127.pdf>`_
      - system heterogeneity
      - flower/baselines/flwr_baselines/publications/ fedprox/system_heterogeneity
