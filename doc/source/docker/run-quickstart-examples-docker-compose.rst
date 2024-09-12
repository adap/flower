Run Flower Quickstart Examples with Docker Compose
==================================================

Flower provides a set of `quickstart examples <https://github.com/adap/flower/tree/main/examples>`_
to help you get started with the framework.

These examples are designed to demonstrate the capabilities of Flower and can be run
using Docker Compose.

.. important::

   Some quickstart examples may have limitations or requirements that prevent them from running
   on every environment. For more information, please see `Limitation`_.

Prerequisites
-------------

Before you start, make sure that:

- The ``flwr`` CLI is :doc:`installed <../how-to-install-flower>` locally.
- The Docker daemon is running.
- Docker Compose is `installed <https://docs.docker.com/compose/install/>`_.
- Two open terminal windows.

**Why Two Terminals?**

It is recommended to use two terminals to run the quickstart examples. This allows you to keep the
``docker compose`` commands in one terminal and the ``flwr`` commands in the other, avoiding the
need to switch between directories.

Step 1: Start the Docker Services (Terminal 1)
----------------------------------------------

#. Clone the Flower Repository:

   .. code-block:: bash

      $ git clone --depth=1 https://github.com/adap/flower.git


#. Change the directory to the location of the Docker Compose file:

   .. code-block:: bash

      $ cd flower/src/docker/complete/

#. Set the ``PROJECT_DIR`` environment variable to the path of the quickstart example you want to
   run. The path should be relative to the location of the Docker Compose files:

   .. code-block:: bash

      $ export PROJECT_DIR=../../../examples/quickstart-sklearn-tabular

#. Build and start the services using the following command:

   .. code-block:: bash

      $ docker compose -f compose.yml up --build -d

#. Follow the logs of the SuperExec service:

   .. code-block:: bash

      $ docker compose logs superexec -f

Step 2: Run the Quickstart Example (Terminal 2)
-----------------------------------------------

#. Change the directory to the quickstart example:

   .. code-block:: bash

      $ cd flower/examples/quickstart-sklearn-tabular

#. Append the following lines to the end of the ``pyproject.toml`` file and save it:

   .. code-block:: toml
      :caption: pyproject.toml

      [tool.flwr.federations.docker-compose]
      address  =  "127.0.0.1:9093"
      insecure  = true

#. Run the example:

   .. code-block:: bash

      $ flwr run . docker-compose

That is all it takes! You can monitor the progress of the run in the first terminal.

Run a Different Quickstart Example
----------------------------------

#. To run a different quickstart example, such as ``quickstart-pytorch``, switch back to the first
   terminal and update the ``PROJECT_DIR`` environment variable:

   .. code-block:: bash
      :caption: Terminal 1

      $ export PROJECT_DIR=../../../examples/quickstart-pytorch

#. Rebuild and restart the services and follow the SuperExec logs:

   .. code-block:: bash
      :caption: Terminal 1

      $ docker compose -f compose.yml up --build -d --force-recreate
      $ docker compose logs superexec -f

#. Switch back to the second terminal and change the directory to the new quickstart
   example you like to run:

   .. code-block:: bash
      :caption: Terminal 2

      $ cd ../quickstart-pytorch

#. Append the following lines to the end of the ``pyproject.toml`` file and save it:

   .. code-block:: toml
      :caption: pyproject.toml

      [tool.flwr.federations.docker-compose]
      address  =  "127.0.0.1:9093"
      insecure  = true

#. Run the example:

   .. code-block:: bash
      :caption: Terminal 2

      $ flwr run . docker-compose

Clean Up
--------

Remove all services and volumes:

.. code-block:: bash

   $ docker compose down -v

Limitation
----------

.. list-table::
   :header-rows: 1

   * - Quickstart Example
     - Limitation/Requirements
   * - quickstart-fastai
     - None
   * - examples/quickstart-huggingface
     - For CPU-only environments, it requires at least 32GB of memory.
   * - quickstart-jax
     - The example has not yet been updated to work with the latest ``flwr`` version.
   * - quickstart-mlcube
     - The example has not yet been updated to work with the latest ``flwr`` version.
   * - quickstart-mlx
     - `Requires to run on macOS with Apple Silicon <https://ml-explore.github.io/mlx/build/html/install.html#python-installation>`_.
   * - quickstart-monai
     - None
   * - quickstart-pandas
     - The example has not yet been updated to work with the latest ``flwr`` version.
   * - quickstart-pytorch-lightning
     - Requires an older pip version that is not supported by the Flower Docker images.
   * - quickstart-pytorch
     - None
   * - quickstart-sklearn-tabular
     - None
   * - quickstart-tabnet
     - The example has not yet been updated to work with the latest ``flwr`` version.
   * - quickstart-tensorflow
     - Only runs on AMD64.
