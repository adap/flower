Deploy Flower on Multiple Machines with Docker Compose
======================================================

This guide will help you set up a Flower project on multiple machines using Docker
Compose.

You will learn how to run the Flower client and server components on two separate
machines, with Flower configured to use TLS encryption and persist SuperLink state
across restarts. A server consists of a SuperLink and ``SuperExec``. For more details
about the Flower architecture, refer to the :doc:`../explanation-flower-architecture`
explainer page.

This guide assumes you have completed the :doc:`tutorial-quickstart-docker-compose`
tutorial. It is highly recommended that you follow and understand the contents of that
tutorial before proceeding with this guide.

Prerequisites
-------------

Before you begin, make sure you have the following prerequisites:

- The ``flwr`` CLI is :doc:`installed <../how-to-install-flower>` locally.
- The Docker daemon is running on your local machine and the remote machine.
- Docker Compose V2 is installed on both your local machine and the remote machine.
- You can connect to the remote machine from your local machine.
- Ports ``9091`` and ``9093`` are accessible on the remote machine.

.. note::

    The guide uses the |quickstart_sklearn_tabular|_ example as an example project.

    If your project has a different name or location, please remember to adjust the
    commands/paths accordingly.

Step 1: Set Up
--------------

1. Clone the Flower repository and change to the ``distributed`` directory:

   .. code-block:: bash

       $ git clone --depth=1 https://github.com/adap/flower.git
       $ cd flower/src/docker/distributed

2. Get the IP address from the remote machine and save it for later.
3. Use the ``certs.yml`` Compose file to generate your own self-signed certificates. If
   you have certificates, you can continue with Step 2.

   .. important::

       These certificates should be used only for development purposes.

       For production environments, you may have to use dedicated services to obtain
       your certificates.

   First, set the environment variables ``SUPERLINK_IP`` and ``SUPEREXEC_IP`` with the
   IP address from the remote machine. For example, if the IP is ``192.168.2.33``,
   execute:

   .. code-block:: bash

       $ export SUPERLINK_IP=192.168.2.33
       $ export SUPEREXEC_IP=192.168.2.33

   Next, generate the self-signed certificates:

   .. code-block:: bash

       $ docker compose -f certs.yml -f ../complete/certs.yml run --rm --build gen-certs

Step 2: Copy the Server Compose Files
-------------------------------------

Use the method that works best for you to copy the ``server`` directory, the
certificates, and the ``pyproject.toml`` file of your Flower project to the remote
machine.

For example, you can use ``scp`` to copy the directories:

.. code-block:: bash

    $ scp -r ./server \
           ./superexec-certificates \
           ./superlink-certificates \
           ../../../examples/quickstart-sklearn-tabular/pyproject.toml remote:~/distributed

Step 3: Start the Flower Server Components
------------------------------------------

Log into the remote machine using ``ssh`` and run the following command to start the
SuperLink and SuperExec services:

.. code-block:: bash
    :linenos:

     $ ssh <your-remote-machine>
     # In your remote machine
     $ cd <path-to-``distributed``-directory>
     $ export PROJECT_DIR=../
     $ docker compose -f server/compose.yml up --build -d

.. note::

    The path to the ``PROJECT_DIR`` containing the ``pyproject.toml`` file should be
    relative to the location of the server ``compose.yml`` file.

.. note::

    When working with Docker Compose on Linux, you may need to create the ``state``
    directory first and change its ownership to ensure proper access and permissions.
    After exporting the ``PROJECT_DIR`` (after line 4), run the following commands:

    .. code-block:: bash

        $ mkdir server/state
        $ sudo chown -R 49999:49999 server/state

    For more information, consult the following page: :doc:`persist-superlink-state`.

Go back to your terminal on your local machine.

Step 4: Start the Flower Client Components
------------------------------------------

On your local machine, run the following command to start the client components:

.. code-block:: bash

    # In the `docker/distributed` directory
    $ export PROJECT_DIR=../../../../examples/quickstart-sklearn-tabular
    $ docker compose -f client/compose.yml up --build -d

.. note::

    The path to the ``PROJECT_DIR`` containing the ``pyproject.toml`` file should be
    relative to the location of the client ``compose.yml`` file.

Step 5: Run Your Flower Project
-------------------------------

Specify the remote SuperExec IP addresses and the path to the root certificate in the
``[tool.flwr.federations.remote-superexec]`` table in the ``pyproject.toml`` file. Here,
we have named our remote federation ``remote-superexec``:

.. code-block:: toml
    :caption: examples/quickstart-sklearn-tabular/pyproject.toml

    [tool.flwr.federations.remote-superexec]
    address = "192.168.2.33:9093"
    root-certificates = "../../src/docker/distributed/superexec-certificates/ca.crt"

.. note::

    The path of the ``root-certificates`` should be relative to the location of the
    ``pyproject.toml`` file.

Run the project and follow the ServerApp logs:

.. code-block:: bash

    $ flwr run ../../../examples/quickstart-sklearn-tabular remote-superexec --stream

That's it! With these steps, you've set up Flower on two separate machines and are ready
to start using it.

Step 6: Clean Up
----------------

Shut down the Flower client components:

.. code-block:: bash

    # In the `docker/distributed` directory
    $ docker compose -f client/compose.yml down

Shut down the Flower server components and delete the SuperLink state:

.. code-block:: bash

    $ ssh <your-remote-machine>
    $ cd <path-to-``distributed``-directory>
    $ docker compose -f server/compose.yml down -v

.. |quickstart_sklearn_tabular| replace:: ``examples/quickstart-sklearn-tabular``

.. _quickstart_sklearn_tabular: https://github.com/adap/flower/tree/main/examples/quickstart-sklearn-tabular
