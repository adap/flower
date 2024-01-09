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
  `SSL <https://flower.dev/docs/framework/how-to-run-flower-using-docker.html#enabling-ssl-for-secure-connections>`_
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
  `Enable SSL connections <https://flower.dev/docs/framework/how-to-enable-ssl-connections.html#certificates>`_
  page contains a section that will guide you through the process.

Assuming all files we need are in the local ``certificates`` directory, we can use the flag
``-v`` to mount the local directory into the ``/app/`` directory of the container. This allows the
server to access the files within the container. Finally, we pass the names of the certificates to
the server with the ``--certificates`` flag.

.. code-block:: bash

  $ docker run --rm \
    -p 9091:9091 -p 9092:9092 -v ./certificates/:/app/ flwr/server:1.7.0-py3.11-ubuntu22.04 \
    --certificates ca.crt server.pem server.key

Flower client
-------------

Docker client images come with a pre-installed version of Flower and serve as a base for building
your own client image. Therefore, they don't run anything when you try to start them. We will use an
example to illustrate how you can dockerize your client code. You can find the full example
`here <https://github.com/adap/flower/tree/main/examples/docker-client>`_.

Project layout
~~~~~~~~~~~~~~

Let's assume the following project layout:

.. code-block:: bash

  $ tree .
  .
  ├── Dockerfile
  ├── client-code
  │   ├── client.py
  │   └── requirements.txt
  ├── driver.py
  └── requirements.txt

We briefly go through each of the files to understand what their purpose is.

**Dockerfile**

The ``Dockerfile`` contains the instructions that assemble the client image.

.. code-block:: dockerfile
  :linenos:

  FROM flwr/client:1.6.0-py3.8-ubuntu22.04

  WORKDIR /app
  COPY requirements.txt .
  RUN python -m pip install -U --no-cache-dir -r requirements.txt

  COPY client.py .
  ENTRYPOINT ["python", "-c", "from flwr.client import run_client; run_client()", "--callable", "client:flower"]

In the first three lines, we instruct Docker to use the client image tagged
``1.6.0-py3.8-ubuntu22.04`` as a base and set our working directory to ``/app``. All of the
following instructions will now be executed in the ``/app`` directory. In lines 4-5, we install the
Python dependencies by copying the ``client-code/requirements.txt`` file into the image and running
``pip`` install on it. In the last two lines, we copy the ``client-code/client.py`` file into
the image and set the entry point. The entry point may look a little unusual, but it is the same as
if we ran the command ``flower-client --callable client:flower`` in a terminal.

You may be wondering why we don't copy all files in the client-code directory in a single
``COPY`` instruction and then install the dependencies. The reason we do this split is to use the
Docker build cache to reduce build time. If both files are copied in a single ``COPY`` instruction,
exactly one image layer will be created. The layer will be cached for future builds. However, the
layer is recreated as soon as one of the files is changed. This means that the dependencies are
reinstalled every time, even though we only change the client code. To prevent this behavior, we
first install the dependencies and after that copy the client code.

**client-code directory**

The directory contains the client implementation (``client.py``) and a list of python dependencies
(``requirements.txt``) which are used in the client code.

**driver.py and requirements.txt**

``driver.py`` contains the driver implementation and ``requirements.txt`` the list of python
dependencies which are used in the driver code.

Building the client image
~~~~~~~~~~~~~~~~~~~~~~~~~

First, we will build the client image and install the ``driver.py`` dependencies.

.. code-block:: bash

  $ cd examples/docker-client
  $ docker build -f Dockerfile -t flwr-client:0.1.0 client-code
  $ pip install -r requirements.txt

Next, we create a new bridge network called ``flwr-net``. User-defined networks like
``flwr-net`` can resolve a container name to an IP address. This feature is not available in the
default bridge network. Using container names simplifies the example because we don't have to first
figure out what the server's IP address is.

.. code-block:: bash

  $ docker network create --driver bridge flwr-net

Running the example
~~~~~~~~~~~~~~~~~~~

Start the long-running Flower server.

.. code-block:: bash

  $ docker run --name flwr-server \
    --rm -p 9091:9091 -p 9092:9092 \
    --network flwr-net \
    flwr/server:1.6.0-py3.11-ubuntu22.04 --insecure

In a new terminal window, start the first long-running Flower client.

.. code-block:: bash

  $ docker run --rm \
    --network flwr-net \
    flwr-client:0.1.0 --insecure --server flwr-server:9092

In yet another new terminal window, start the second long-running Flower client.

.. code-block:: bash

  $ docker run --rm \
    --network flwr-net \
    flwr-client:0.1.0 --insecure --server flwr-server:9092

Start the driver script.

.. code-block:: bash

  $ python driver.py

As soon as the driver script has finished, we can stop the server and client containers and
remove the previously created network.

.. code-block:: bash

  $ docker network rm flwr-net


Advanced Docker options
-----------------------

Using a different Flower or Python version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use a different version of Flower or Python, you can do so by changing the tag.
All versions we provide are available on `Docker Hub <https://hub.docker.com/r/flwr>`_.

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
