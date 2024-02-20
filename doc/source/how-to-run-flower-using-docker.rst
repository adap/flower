Run Flower using Docker
=======================

The simplest way to get started with Flower is by using the pre-made Docker images, which you can
find on `Docker Hub <https://hub.docker.com/r/flwr/server/tags>`_.

Before you start, make sure that the Docker daemon is running:

.. code-block:: bash

  $ docker -v
  Docker version 24.0.7, build afdd53b

If you do not see the version of Docker but instead get an error saying that the command
was not found, you will need to install Docker first. You can find installation instruction
`here <https://docs.docker.com/get-docker/>`_.

.. note::

  On Linux, Docker commands require ``sudo`` privilege. If you want to avoid using ``sudo``,
  you can follow the `Post-installation steps <https://docs.docker.com/engine/install/linux-postinstall/>`_
  on the official Docker website.

Flower server
-------------

Quickstart
~~~~~~~~~~

If you're looking to try out Flower, you can use the following command:

.. code-block:: bash

  $ docker run --rm -p 9091:9091 -p 9092:9092 flwr/server:1.7.0-py3.11-ubuntu22.04 \
    --insecure

The command will pull the Docker image with the tag ``1.7.0-py3.11-ubuntu22.04`` from Docker Hub.
The tag contains the information which Flower, Python and Ubuntu is used. In this case, it
uses Flower 1.7.0, Python 3.11 and Ubuntu 22.04. The ``--rm`` flag tells Docker to remove
the container after it exits.

.. note::

  By default, the Flower server keeps state in-memory. When using the Docker flag
  ``--rm``, the state is not persisted between container starts. We will show below how to save the
  state in a file on your host system.

The ``-p <host>:<container>`` flag tells Docker to map the ports ``9091``/``9092`` of the host to
``9091``/``9092`` of the container, allowing you to access the Driver API on ``http://localhost:9091``
and the Fleet API on ``http://localhost:9092``. Lastly, any flag that comes after the tag is passed
to the Flower server. Here, we are passing the flag ``--insecure``.

.. attention::

  The ``--insecure`` flag enables insecure communication (using HTTP, not HTTPS) and should only be used
  for testing purposes. We strongly recommend enabling
  `SSL <https://flower.ai/docs/framework/how-to-run-flower-using-docker.html#enabling-ssl-for-secure-connections>`_
  when deploying to a production environment.

You can use ``--help`` to view all available flags that the server supports:

.. code-block:: bash

  $ docker run --rm flwr/server:1.7.0-py3.11-ubuntu22.04 --help

Mounting a volume to store the state on the host system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to persist the state of the server on your host system, all you need to do is specify a
path where you want to save the file on your host system and a name for the database file. In the
example below, we tell Docker via the flag ``-v`` to mount the user's home directory
(``~/`` on your host) into the ``/app/`` directory of the container. Furthermore, we use the
flag ``--database`` to specify the name of the database file.

.. code-block:: bash

  $ docker run --rm \
    -p 9091:9091 -p 9092:9092 -v ~/:/app/ flwr/server:1.7.0-py3.11-ubuntu22.04 \
    --insecure \
    --database state.db

As soon as the server starts, the file ``state.db`` is created in the user's home directory on
your host system. If the file already exists, the server tries to restore the state from the file.
To start the server with an empty database, simply remove the ``state.db`` file.

Enabling SSL for secure connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable SSL, you will need a CA certificate, a server certificate and a server private key.

.. note::
  For testing purposes, you can generate your own self-signed certificates. The
  `Enable SSL connections <https://flower.ai/docs/framework/how-to-enable-ssl-connections.html#certificates>`_
  page contains a section that will guide you through the process.

Assuming all files we need are in the local ``certificates`` directory, we can use the flag
``-v`` to mount the local directory into the ``/app/`` directory of the container. This allows the
server to access the files within the container. Finally, we pass the names of the certificates to
the server with the ``--certificates`` flag.

.. code-block:: bash

  $ docker run --rm \
    -p 9091:9091 -p 9092:9092 -v ./certificates/:/app/ flwr/server:1.7.0-py3.11-ubuntu22.04 \
    --certificates ca.crt server.pem server.key

Using a different Flower or Python version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use a different version of Flower or Python, you can do so by changing the tag.
All versions we provide are available on `Docker Hub <https://hub.docker.com/r/flwr/server/tags>`_.

Pinning a Docker image to a specific version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It may happen that we update the images behind the tags. Such updates usually include security
updates of system dependencies that should not change the functionality of Flower. However, if you
want to ensure that you always use the same image, you can specify the hash of the image instead of
the tag.

The following command returns the current image hash referenced by the ``server:1.7.0-py3.11-ubuntu22.04`` tag:

.. code-block:: bash

  $ docker inspect --format='{{index .RepoDigests 0}}' flwr/server:1.7.0-py3.11-ubuntu22.04
  flwr/server@sha256:c4be5012f9d73e3022e98735a889a463bb2f4f434448ebc19c61379920b1b327

Next, we can pin the hash when running a new server container:

.. code-block:: bash

  $ docker run \
    --rm flwr/server@sha256:c4be5012f9d73e3022e98735a889a463bb2f4f434448ebc19c61379920b1b327 \
    --insecure

Setting environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To set a variable inside a Docker container, you can use the ``-e <name>=<value>`` flag.

.. code-block:: bash

  $ docker run -e FLWR_TELEMETRY_ENABLED=0 \
    --rm flwr/server:1.7.0-py3.11-ubuntu22.04 --insecure
