Run Flower using Docker
=======================

The simplest way to get started with Flower is by using the pre-made Docker images, which you can
find on `Docker Hub <https://hub.docker.com/u/flwr>`_.

Before you start, make sure that the Docker daemon is running:

.. code-block:: bash

  $ docker -v
  Docker version 26.0.0, build 2ae903e

If you do not see the version of Docker but instead get an error saying that the command
was not found, you will need to install Docker first. You can find installation instruction
`here <https://docs.docker.com/get-docker/>`_.

.. note::

  On Linux, Docker commands require ``sudo`` privilege. If you want to avoid using ``sudo``,
  you can follow the `Post-installation steps <https://docs.docker.com/engine/install/linux-postinstall/>`_
  on the official Docker website.

Flower SuperLink
----------------

Quickstart
~~~~~~~~~~

If you're looking to try out Flower, you can use the following command:

.. code-block:: bash

  $ docker run --rm -p 9091:9091 -p 9092:9092 flwr/superlink:1.8.0 --insecure

The command pulls the Docker image with the tag ``1.8.0`` from Docker Hub. The tag specifies
the Flower version. In this case, Flower 1.8.0. The ``--rm`` flag tells Docker to remove the
container after it exits.

.. note::

  By default, the Flower SuperLink keeps state in-memory. When using the Docker flag ``--rm``, the
  state is not persisted between container starts. We will show below how to save the state in a
  file on your host system.

The ``-p <host>:<container>`` flag tells Docker to map the ports ``9091``/``9092`` of the host to
``9091``/``9092`` of the container, allowing you to access the Driver API on ``http://localhost:9091``
and the Fleet API on ``http://localhost:9092``. Lastly, any flag that comes after the tag is passed
to the Flower SuperLink. Here, we are passing the flag ``--insecure``.

.. attention::

  The ``--insecure`` flag enables insecure communication (using HTTP, not HTTPS) and should only be
  used for testing purposes. We strongly recommend enabling
  `SSL <https://flower.ai/docs/framework/how-to-run-flower-using-docker.html#enabling-ssl-for-secure-connections>`_
  when deploying to a production environment.

You can use ``--help`` to view all available flags that the SuperLink supports:

.. code-block:: bash

  $ docker run --rm flwr/superlink:1.8.0 --help

Mounting a volume to store the state on the host system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to persist the state of the SuperLink on your host system, all you need to do is specify
a path where you want to save the file on your host system and a name for the database file. In the
example below, we tell Docker via the flag ``-v`` to mount the user's home directory
(``~/`` on your host) into the ``/app/`` directory of the container. Furthermore, we use the
flag ``--database`` to specify the name of the database file.

.. code-block:: bash

  $ docker run --rm \
    -p 9091:9091 -p 9092:9092 -v ~/:/app/ flwr/superlink:1.8.0 \
    --insecure \
    --database state.db

As soon as the SuperLink starts, the file ``state.db`` is created in the user's home directory on
your host system. If the file already exists, the SuperLink tries to restore the state from the
file. To start the SuperLink with an empty database, simply remove the ``state.db`` file.

Enabling SSL for secure connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable SSL, you will need a PEM-encoded root certificate, a PEM-encoded private key and a
PEM-encoded certificate chain.

.. note::
  For testing purposes, you can generate your own self-signed certificates. The
  `Enable SSL connections <https://flower.ai/docs/framework/how-to-enable-ssl-connections.html#certificates>`_
  page contains a section that will guide you through the process.

Assuming all files we need are in the local ``certificates`` directory, we can use the flag
``-v`` to mount the local directory into the ``/app/`` directory of the container. This allows the
SuperLink to access the files within the container. Finally, we pass the names of the certificates
to the SuperLink with the ``--certificates`` flag.

.. code-block:: bash

  $ docker run --rm \
    -p 9091:9091 -p 9092:9092 -v ./certificates/:/app/ flwr/superlink:1.8.0 \
    --certificates ca.crt server.pem server.key

Using a different Flower version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use a different version of Flower, for example Flower nightly, you can do so by
changing the tag. All available versions are on
`Docker Hub <https://hub.docker.com/r/flwr/superlink/tags>`_.

Pinning a Docker image to a specific version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It may happen that we update the images behind the tags. Such updates usually include security
updates of system dependencies that should not change the functionality of Flower. However, if you
want to ensure that you always use the same image, you can specify the hash of the image instead of
the tag.

The following command returns the current image hash referenced by the ``superlink:1.8.0`` tag:

.. code-block:: bash

  $ docker inspect --format='{{index .RepoDigests 0}}' flwr/superlink:1.8.0
  flwr/superlink@sha256:1b855d1fa4e344e4d95db99793f2bb35d8c63f6a1decdd736863bfe4bb0fe46c

Next, we can pin the hash when running a new SuperLink container:

.. code-block:: bash

  $ docker run \
    --rm flwr/superlink@sha256:1b855d1fa4e344e4d95db99793f2bb35d8c63f6a1decdd736863bfe4bb0fe46c \
    --insecure

Setting environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To set a variable inside a Docker container, you can use the ``-e <name>=<value>`` flag.

.. code-block:: bash

  $ docker run -e FLWR_TELEMETRY_ENABLED=0 \
    --rm flwr/superlink:1.8.0 --insecure
