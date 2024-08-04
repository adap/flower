Deploying Flower on Multiple Machines using Docker Compose
==========================================================

This guide will help you set up a Flower project on multiple machines using Docker Compose.

You will learn how to run the Flower client and server components on two different machines,
with Flower running with TLS enabled and with persisted SuperLink state.

This guide assumes that you have already completed the :doc:`tutorial-quickstart-docker-compose` tutorial.
It is highly recommended to follow and understand the contents of that tutorial before
proceeding with this guide.

Prerequisites
-------------

Before you begin, make sure you have the following prerequisites:

- The ``flwr`` CLI is :doc:`installed <../how-to-install-flower>` locally.
- The Docker daemon is running on your local and remote machines.
- Docker Compose is on both your local and remote machines.
- You have an existing Flower project. You can use ``flwr new`` to create one.
- You can connect to the remote machine from your local machine.
- Ports ``9091`` and ``9093`` are accessible on the remote machine.

.. note::

   The commands provided in this guide assume that your project name is
   ``quickstart-docker-compose`` and its location is within the ``distributed`` directory.

   If your project has a different name or location, please remember to adjust the commands accordingly.

Step 1: Set Up
--------------

#. Clone the Docker Compose ``distributed`` directory:

   .. code-block:: bash

      $ git clone --depth=1 https://github.com/adap/flower.git _tmp \
                  && mv _tmp/src/docker/distributed . \
                  && mv _tmp/src/docker/complete . \
                  && rm -rf _tmp && cd distributed

#. Get the ip address from the remote machine and save it for later.

#. If you don't have any certificates, you can generate self-signed certificates using the ``certs.yml``
   Compose file. If you have certificates you can continue with Step 2.

   .. important::

      These certificates should be used only for development purposes.

      For production environments, use a service like `Let's Encrypt <https://letsencrypt.org/>`_
      to obtain your certificates.

   First, set the environment variable ``SUPERLINK_IP`` and ``SUPEREXEC_IP`` with the IP address
   from the remote machine. For example, if the IP is ``192.168.2.33``, use:

   .. code-block:: bash

      $ export SUPERLINK_IP=192.168.2.33
      $ export SUPEREXEC_IP=192.168.2.33


   Next, generate the self-signed certificates:

   .. code-block:: bash

      $ docker compose -f certs.yml -f ../complete/certs.yml up --build


Step 2: Copy the Server Compose Files
-------------------------------------

Use the method that best works for you to copy the ``server`` directory, the certificates, and your
Flower project to the remote machine.

For example, you can use ``scp`` to copy the directories:

.. code-block:: bash

   $ scp -r ./server \
          ./superexec-certificates \
          ./superlink-certificates \
          ./quickstart-docker-compose remote:~/

Step 3: Start the Flower Server Components
------------------------------------------

Log into the remote machine using, for example, ssh and run the following command to start the
SuperLink and SuperExec services:

.. code-block:: bash

   $ ssh remote
   $ export PROJECT_DIR=../quickstart-docker-compose
   $ docker compose -f server/compose.yml -f server/with-tls.yml \
      -f server/with-state.yml up --build -d

.. note::

   The Path of the ``PROJECT_DIR`` should be relative to the location of the ``server`` Docker
   Compose files.

Go back to your terminal on your local machine.

Step 4: Start the Flower Client Components
------------------------------------------

On your local machine, run the following command to start the SuperNode services:

.. code-block:: bash

   $ export PROJECT_DIR=../quickstart-docker-compose
   $ docker compose -f client/compose.yml -f client/with-tls.yml up --build -d

.. note::

   The Path of the ``PROJECT_DIR`` should be relative to the location of the ``client`` Docker
   Compose files.

Step 5: Run Your Flower Project
-------------------------------

Specify the remote SuperExec IP addresses and the path to the root-certificate in the
``pyproject.toml`` file:

.. code-block:: toml
   :caption: quickstart-docker-compose/pyproject.toml

   [tool.flwr.federations.remote-superexec]
   address = "192.168.2.33:9093"
   root-certificates = "superexec-certificates/ca.crt"

To run the project, execute:

.. code-block:: bash

   $ flwr run quickstart-docker-compose remote-superexec

Step 6: Clean Up
-----------------

Shutdown the Flower client components:

.. code-block:: bash

   $ docker compose -f client/compose.yml -f client/with-tls.yml down

Shutdown the Flower server components and delete the SuperLink state:

.. code-block:: bash

   $ ssh remote
   $ docker compose -f server/compose.yml -f server/with-tls.yml \
      -f server/with-state.yml down -v
