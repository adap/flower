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

Flower SuperNode
----------------

The SuperNode Docker image comes with a pre-installed version of Flower and serves as a base for
building your own SuperNode image.

We will use the ``app-pytorch`` example, which you can find in
the Flower repository, to illustrate how you can dockerize your client-app.

Prerequisites
~~~~~~~~~~~~~

Before we can start, we need to meet a few prerequisites in our local development environment.
You can skip the first part if you want to run your client-app instead of the ``app-pytorch``
example.

#. Clone the flower repository.

    .. code-block:: bash

      $ git clone https://github.com/adap/flower.git && cd flower/examples/app-pytorch

#. Verify the Docker daemon is running.

    Please follow the first section on
    :doc:`Run Flower using Docker <how-to-run-flower-using-docker>`
    which covers this step in more detail.


Creating a SuperNode Dockerfile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's assume the following project layout:

.. code-block:: bash

  $ tree .
  .
  ├── client.py        # client-app code
  ├── task.py          # client-app code
  ├── requirements.txt # client-app dependencies
  └── <other files>

First, we need to create a Dockerfile in the directory where the client-app code is located.
If you use the ``app-pytorch`` example, create a new file called ``Dockerfile`` in
``examples/app-pytorch``.

The ``Dockerfile`` contains the instructions that assemble the SuperNode image.

.. code-block:: dockerfile

  FROM flwr/supernode:nightly

  WORKDIR /app
  COPY requirements.txt .
  RUN python -m pip install -U --no-cache-dir -r requirements.txt && pyenv rehash

  COPY client.py task.py ./
  ENTRYPOINT ["flower-client-app"]

In the first two lines, we instruct Docker to use the SuperNode image tagged ``nightly`` as a base
image and set our working directory to ``/app``. The following instructions will now be
executed in the ``/app`` directory. Next, we install the client-app dependencies by copying the
``requirements.txt`` file into the image and run ``pip install``. In the last two lines,
we copy the client-app code (``client.py`` and ``task.py``) into the image and set the entry
point to ``flower-client-app``.

Building the SuperNode Docker image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we build the SuperNode Docker image by running the following command in the directory where
Dockerfile and client-app code are located.

.. code-block:: bash

  $ docker build -t flwr_supernode:0.0.1 .

We gave the image the name ``flwr_supernode``, and the tag ``0.0.1``. Remember that the here chosen
values only serve as an example. You can change them to your needs.


Running the SuperNode Docker image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have built the SuperNode image, we can finally run it.

.. code-block:: bash

  $ docker run --rm flwr_supernode:0.0.1 client:app \
    --insecure \
    --server 192.168.1.100:9092

Let's break down each part of this command:

* ``docker run``: This is the command to run a new Docker container.
* ``--rm``: This option specifies that the container should be automatically removed when it stops.
* | ``flwr_supernode:0.0.1``: The name the tag of the Docker image to use.
* | ``client:app``: The object reference of the client-app (``<module>:<attribute>``).
  | It points to the client-app that will be run inside the SuperNode container.
* ``--insecure``: This option enables insecure communication.

.. attention::

  The ``--insecure`` flag enables insecure communication (using HTTP, not HTTPS) and should only be
  used for testing purposes. We strongly recommend enabling
  `SSL <https://flower.ai/docs/framework/how-to-run-flower-using-docker.html#enabling-ssl-for-secure-connections>`_
  when deploying to a production environment.

* | ``--server 192.168.1.100:9092``: This option specifies the address of the SuperLinks Fleet
  | API to connect to. Remember to update it with your SuperLink IP.

.. note::
  Any argument that comes after the tag is passed to the Flower SuperNode binary.
  To see all available flags that the SuperNode supports, run:

  .. code-block:: bash

    $ docker run --rm flwr/supernode:nightly --help

Enabling SSL for secure connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable SSL, we will need to mount a PEM-encoded root certificate into your SuperNode container.

Assuming the certificate already exists locally, we can use the flag ``-v`` to mount the local
certificate into the container's ``/app/`` directory. This allows the SuperNode to access the
certificate within the container. Use the ``--certificates`` flag when starting the container.

.. code-block:: bash

  $ docker run --rm -v ./ca.crt:/app/ca.crt flwr_supernode:0.0.1 client:app \
    --insecure \
    --server 192.168.1.100:9092 \
    --certificates ca.crt

Advanced Docker options
-----------------------

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
