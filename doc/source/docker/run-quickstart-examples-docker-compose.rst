Run Flower Quickstart Examples with Docker Compose
==================================================

Flower provides a set of `quickstart examples
<https://github.com/adap/flower/tree/main/examples>`_ to help you get started with the
framework. These examples are designed to demonstrate the capabilities of Flower and by
default run using the Simulation Engine. This guide demonstrates how to run them using
Flower's Deployment Engine via Docker Compose.

.. important::

    Some quickstart examples may have limitations or requirements that prevent them from
    running on every environment. For more information, please see Limitations_.

Prerequisites
-------------

Before you start, make sure that:

- The ``flwr`` CLI is :doc:`installed <../how-to-install-flower>` locally.
- The Docker daemon is running.
- Docker Compose V2 is `installed <https://docs.docker.com/compose/install/>`_.

Run the Quickstart Example
--------------------------

1. Clone the quickstart example you like to run. For example, ``quickstart-pytorch``:

   .. code-block:: bash

       $ git clone --depth=1 https://github.com/adap/flower.git \
            && mv flower/examples/quickstart-pytorch . \
            && rm -rf flower && cd quickstart-pytorch

2. Download the `compose.yml
   <https://github.com/adap/flower/blob/main/src/docker/complete/compose.yml>`_ file
   into the example directory:

   .. code-block:: bash
       :substitutions:

       $ curl https://raw.githubusercontent.com/adap/flower/refs/tags/v|stable_flwr_version|/src/docker/complete/compose.yml \
           -o compose.yml

3. Export the version of Flower that your environment uses. Then, build and start the
   services using the following command:

   .. code-block:: bash
       :substitutions:

       $ export FLWR_VERSION="|stable_flwr_version|" # update with your version
       $ docker compose up --build -d

4. Append the following lines to the end of the ``pyproject.toml`` file and save it:

   .. code-block:: toml
       :caption: pyproject.toml

       [tool.flwr.federations.local-deployment]
       address = "127.0.0.1:9093"
       insecure = true

   .. note::

       You can customize the string that follows ``tool.flwr.federations.`` to fit your
       needs. However, please note that the string cannot contain a dot (``.``).

       In this example, ``local-deployment`` has been used. Just remember to replace
       ``local-deployment`` with your chosen name in both the ``tool.flwr.federations.``
       string and the corresponding ``flwr run .`` command.

5. Run the example and follow the logs of the ``ServerApp`` :

   .. code-block:: bash

       $ flwr run . local-deployment --stream

That is all it takes! You can monitor the progress of the run through the logs of the
``ServerApp``.

Run a Different Quickstart Example
----------------------------------

To run a different quickstart example, such as ``quickstart-tensorflow``, first, shut
down the Docker Compose services of the current example:

.. code-block:: bash

    $ docker compose down

After that, you can repeat the steps above.

Limitations
-----------

.. list-table::
    :header-rows: 1

    - - Quickstart Example
      - Limitations
    - - quickstart-fastai
      - None
    - - quickstart-huggingface
      - None
    - - quickstart-jax
      - None
    - - quickstart-mlcube
      - The example has not yet been updated to work with the latest ``flwr`` version.
    - - quickstart-mlx
      - `Requires to run on macOS with Apple Silicon
        <https://ml-explore.github.io/mlx/build/html/install.html#python-installation>`_.
    - - quickstart-monai
      - None
    - - quickstart-pandas
      - None
    - - quickstart-pytorch-lightning
      - Requires an older pip version that is not supported by the Flower Docker images.
    - - quickstart-pytorch
      - None
    - - quickstart-sklearn-tabular
      - None
    - - quickstart-tabnet
      - The example has not yet been updated to work with the latest ``flwr`` version.
    - - quickstart-tensorflow
      - None
