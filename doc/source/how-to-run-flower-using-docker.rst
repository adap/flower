Run Flower using Docker
=======================

The simplest way to get started with Flower is by using the pre-made Docker images, which you can
find on `Docker Hub <https://hub.docker.com/u/flwr>`__. Supported architectures include ``amd64``
and ``arm64v8``.

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

.. important::

  To ensure optimal performance and compatibility, the SuperLink, SuperNode and ServerApp image
  must have the same version when running together. This guarantees seamless integration and
  avoids potential conflicts or issues that may arise from using different versions.

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
  `SSL <https://flower.ai/docs/framework/how-to-run-flower-using-docker.html#enabling-ssl-for-secure-connections>`__
  when deploying to a production environment.

You can use ``--help`` to view all available flags that the SuperLink supports:

.. code-block:: bash

  $ docker run --rm flwr/superlink:1.8.0 --help

Mounting a volume to store the state on the host system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to persist the state of the SuperLink on your host system, all you need to do is specify
a directory where you want to save the file on your host system and a name for the database file. By
default, the SuperLink container runs with a non-root user called ``app`` with the user ID
``49999``. It is recommended to create new directory and change the user ID of the directory to
``49999`` to ensure the mounted directory has the proper permissions. If you later want to delete
the directory, you can change the user ID back to the current user ID by running
``sudo chown -R $USER:$(id -gn) state``.

In the example below, we create a new directory, change the user ID and tell Docker via the flag
``--volume`` to mount the local ``state`` directory into the ``/app/state`` directory of the
container. Furthermore, we use the flag ``--database`` to specify the name of the database file.

.. code-block:: bash

  $ mkdir state
  $ sudo chown -R 49999:49999 state
  $ docker run --rm \
    -p 9091:9091 -p 9092:9092 --volume ./state/:/app/state flwr/superlink:1.8.0 \
    --insecure \
    --database state.db

As soon as the SuperLink starts, the file ``state.db`` is created in the ``state`` directory on
your host system. If the file already exists, the SuperLink tries to restore the state from the
file. To start the SuperLink with an empty database, simply remove the ``state.db`` file.

Enabling SSL for secure connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable SSL, you will need a PEM-encoded root certificate, a PEM-encoded private key and a
PEM-encoded certificate chain.

.. note::
  For testing purposes, you can generate your own self-signed certificates. The
  `Enable SSL connections <https://flower.ai/docs/framework/how-to-enable-ssl-connections.html#certificates>`__
  page contains a section that will guide you through the process.

Assuming all files we need are in the local ``certificates`` directory, we can use the flag
``--volume`` to mount the local directory into the ``/app/certificates/`` directory of the container.
This allows the SuperLink to access the files within the container. The ``ro`` stands for
``read-only``. Docker volumes default to ``read-write``; that option tells Docker to make the volume
``read-only`` instead. Finally, we pass the names of the certificates and key file to the SuperLink
with the ``--ssl-ca-certfile``, ``--ssl-certfile`` and ``--ssl-keyfile`` flag.

.. code-block:: bash

  $ docker run --rm \
    -p 9091:9091 -p 9092:9092 \
    --volume ./certificates/:/app/certificates/:ro flwr/superlink:nightly \
    --ssl-ca-certfile certificates/ca.crt \
    --ssl-certfile certificates/server.pem \
    --ssl-keyfile certificates/server.key

.. note::

  Because Flower containers, by default, run with a non-root user ``app``, the mounted files and
  directories must have the proper permissions for the user ID ``49999``. For example, to change the
  user ID of all files in the ``certificates/`` directory, you can run
  ``sudo chown -R 49999:49999 certificates/*``.

Flower SuperNode
----------------

The SuperNode Docker image comes with a pre-installed version of Flower and serves as a base for
building your own SuperNode image.

.. important::

  The SuperNode Docker image currently works only with the 1.9.0-nightly release. A stable version
  will be available when Flower 1.9.0 (stable) gets released (ETA: May). A SuperNode nightly image
  must be paired with the corresponding SuperLink and ServerApp nightly images released on the same
  day. To ensure the versions are in sync, using the concrete tag, e.g., ``1.9.0.dev20240501``
  instead of ``nightly`` is recommended.

We will use the ``quickstart-pytorch`` example, which you can find in
the Flower repository, to illustrate how you can dockerize your ClientApp.

.. _SuperNode Prerequisites:

Prerequisites
~~~~~~~~~~~~~

Before we can start, we need to meet a few prerequisites in our local development environment.
You can skip the first part if you want to run your ClientApp instead of the ``quickstart-pytorch``
example.

#. Clone the Flower repository.

    .. code-block:: bash

      $ git clone --depth=1 https://github.com/adap/flower.git && cd flower/examples/quickstart-pytorch

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
  ├── client.py        # ClientApp code
  └── <other files>

First, we need to create a ``requirements.txt`` file in the directory where the ``ClientApp`` code
is located. In the file, we list all the dependencies that the ClientApp requires.

.. code-block::

  flwr-datasets[vision]>=0.1.0,<1.0.0
  torch==2.2.1
  torchvision==0.17.1
  tqdm==4.66.3

.. important::

  Note that `flwr <https://pypi.org/project/flwr/>`__ is already installed in the ``flwr/supernode``
  base image, so you only need to include other package dependencies in your ``requirements.txt``,
  such as ``torch``, ``tensorflow``, etc.

Next, we create a Dockerfile. If you use the ``quickstart-pytorch`` example, create a new
file called ``Dockerfile.supernode`` in ``examples/quickstart-pytorch``.

The ``Dockerfile.supernode`` contains the instructions that assemble the SuperNode image.

.. code-block:: dockerfile

  FROM flwr/supernode:nightly

  WORKDIR /app

  COPY requirements.txt .
  RUN python -m pip install -U --no-cache-dir -r requirements.txt

  COPY client.py ./
  ENTRYPOINT ["flower-client-app", "client:app"]

In the first two lines, we instruct Docker to use the SuperNode image tagged ``nightly`` as a base
image and set our working directory to ``/app``. The following instructions will now be
executed in the ``/app`` directory. Next, we install the ClientApp dependencies by copying the
``requirements.txt`` file into the image and run ``pip install``. In the last two lines,
we copy the ``client.py`` module into the image and set the entry point to ``flower-client-app`` with
the argument ``client:app``. The argument is the object reference of the ClientApp
(``<module>:<attribute>``) that will be run inside the ClientApp.

Building the SuperNode Docker image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we build the SuperNode Docker image by running the following command in the directory where
Dockerfile and ClientApp code are located.

.. code-block:: bash

  $ docker build -f Dockerfile.supernode -t flwr_supernode:0.0.1 .

We gave the image the name ``flwr_supernode``, and the tag ``0.0.1``. Remember that the here chosen
values only serve as an example. You can change them to your needs.


Running the SuperNode Docker image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have built the SuperNode image, we can finally run it.

.. code-block:: bash

  $ docker run --rm flwr_supernode:0.0.1 \
    --insecure \
    --superlink 192.168.1.100:9092

Let's break down each part of this command:

* ``docker run``: This is the command to run a new Docker container.
* ``--rm``: This option specifies that the container should be automatically removed when it stops.
* ``flwr_supernode:0.0.1``: The name the tag of the Docker image to use.
* ``--insecure``: This option enables insecure communication.

.. attention::

  The ``--insecure`` flag enables insecure communication (using HTTP, not HTTPS) and should only be
  used for testing purposes. We strongly recommend enabling
  `SSL <https://flower.ai/docs/framework/how-to-run-flower-using-docker.html#enabling-ssl-for-secure-connections>`__
  when deploying to a production environment.

* | ``--superlink 192.168.1.100:9092``: This option specifies the address of the SuperLinks Fleet
  | API to connect to. Remember to update it with your SuperLink IP.

.. note::

  To test running Flower locally, you can create a
  `bridge network <https://docs.docker.com/network/network-tutorial-standalone/#use-user-defined-bridge-networks>`__,
  use the ``--network`` argument and pass the name of the Docker network to run your SuperNodes.

Any argument that comes after the tag is passed to the Flower SuperNode binary.
To see all available flags that the SuperNode supports, run:

.. code-block:: bash

  $ docker run --rm flwr/supernode:nightly --help

Enabling SSL for secure connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable SSL, we will need to mount a PEM-encoded root certificate into your SuperNode container.

Assuming the certificate already exists locally, we can use the flag ``--volume`` to mount the local
certificate into the container's ``/app/`` directory. This allows the SuperNode to access the
certificate within the container. Use the ``--root-certificates`` flag when starting the container.

.. code-block:: bash


  $ docker run --rm --volume ./ca.crt:/app/ca.crt flwr_supernode:0.0.1 \
    --superlink 192.168.1.100:9092 \
    --root-certificates ca.crt

Flower ServerApp
----------------

The procedure for building and running a ServerApp image is almost identical to the SuperNode image.

Similar to the SuperNode image, the ServerApp Docker image comes with a pre-installed version of
Flower and serves as a base for building your own ServerApp image.

We will use the same ``quickstart-pytorch`` example as we do in the Flower SuperNode section.
If you have not already done so, please follow the `SuperNode Prerequisites`_ before proceeding.


Creating a ServerApp Dockerfile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's assume the following project layout:

.. code-block:: bash

  $ tree .
  .
  ├── server.py        # ServerApp code
  └── <other files>

First, we need to create a Dockerfile in the directory where the ``ServerApp`` code is located.
If you use the ``quickstart-pytorch`` example, create a new file called ``Dockerfile.serverapp`` in
``examples/quickstart-pytorch``.

The ``Dockerfile.serverapp`` contains the instructions that assemble the ServerApp image.

.. code-block:: dockerfile

  FROM flwr/serverapp:1.8.0

  WORKDIR /app

  COPY server.py ./
  ENTRYPOINT ["flower-server-app", "server:app"]

In the first two lines, we instruct Docker to use the ServerApp image tagged ``1.8.0`` as a base
image and set our working directory to ``/app``. The following instructions will now be
executed in the ``/app`` directory. In the last two lines, we copy the ``server.py`` module into the
image and set the entry point to ``flower-server-app`` with the argument ``server:app``.
The argument is the object reference of the ServerApp (``<module>:<attribute>``) that will be run
inside the ServerApp container.

Building the ServerApp Docker image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we build the ServerApp Docker image by running the following command in the directory where
Dockerfile and ServerApp code are located.

.. code-block:: bash

  $ docker build -f Dockerfile.serverapp -t flwr_serverapp:0.0.1 .

We gave the image the name ``flwr_serverapp``, and the tag ``0.0.1``. Remember that the here chosen
values only serve as an example. You can change them to your needs.


Running the ServerApp Docker image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have built the ServerApp image, we can finally run it.

.. code-block:: bash

  $ docker run --rm flwr_serverapp:0.0.1 \
    --insecure \
    --superlink 192.168.1.100:9091

Let's break down each part of this command:

* ``docker run``: This is the command to run a new Docker container.
* ``--rm``: This option specifies that the container should be automatically removed when it stops.
* ``flwr_serverapp:0.0.1``: The name the tag of the Docker image to use.
* ``--insecure``: This option enables insecure communication.

.. attention::

  The ``--insecure`` flag enables insecure communication (using HTTP, not HTTPS) and should only be
  used for testing purposes. We strongly recommend enabling
  `SSL <https://flower.ai/docs/framework/how-to-run-flower-using-docker.html#enabling-ssl-for-secure-connections>`__
  when deploying to a production environment.

* | ``--superlink 192.168.1.100:9091``: This option specifies the address of the SuperLinks Driver
  | API to connect to. Remember to update it with your SuperLink IP.

.. note::
  To test running Flower locally, you can create a
  `bridge network <https://docs.docker.com/network/network-tutorial-standalone/#use-user-defined-bridge-networks>`__,
  use the ``--network`` argument and pass the name of the Docker network to run your ServerApps.

Any argument that comes after the tag is passed to the Flower ServerApp binary.
To see all available flags that the ServerApp supports, run:

.. code-block:: bash

  $ docker run --rm flwr/serverapp:1.8.0 --help

Enabling SSL for secure connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable SSL, we will need to mount a PEM-encoded root certificate into your ServerApp container.

Assuming the certificate already exists locally, we can use the flag ``--volume`` to mount the local
certificate into the container's ``/app/`` directory. This allows the ServerApp to access the
certificate within the container. Use the ``--root-certificates`` flags when starting the container.

.. code-block:: bash

  $ docker run --rm --volume ./ca.crt:/app/ca.crt flwr_serverapp:0.0.1 \
    --superlink 192.168.1.100:9091 \
    --root-certificates ca.crt

Advanced Docker options
-----------------------

Run with root user privileges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Flower Docker images, by default, run with a non-root user (username/groupname: ``app``,
UID/GID: ``49999``). Using root user is not recommended unless it is necessary for specific
tasks during the build process. Always make sure to run the container as a non-root user in
production to maintain security best practices.

**Run a container with root user privileges**

Run the Docker image with the ``-u`` flag and specify ``root`` as the username:

.. code-block:: bash

   $ docker run --rm -u root flwr/superlink:1.8.0

This command will run the Docker container with root user privileges.

**Run the build process with root user privileges**

If you want to switch to the root user during the build process of the Docker image to install
missing system dependencies, you can use the ``USER root`` directive within your Dockerfile.

.. code-block:: dockerfile

  FROM flwr/supernode:1.8.0

  # Switch to root user
  USER root

  # Install missing dependencies (requires root access)
  RUN apt-get update && apt-get install -y <required-package-name>

  # Switch back to non-root user app
  USER app

  # Continue with your Docker image build process
  ...

Using a different Flower version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use a different version of Flower, for example Flower nightly, you can do so by
changing the tag. All available versions are on `Docker Hub <https://hub.docker.com/u/flwr>`__.

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
