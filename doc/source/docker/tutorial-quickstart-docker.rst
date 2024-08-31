Quickstart with Docker
======================

This quickstart aims to guide you through the process of containerizing a Flower project and
running it end to end using Docker on your local machine.

This tutorial does not use production-ready settings, so you can focus on understanding the basic
workflow that uses the minimum configurations.

Prerequisites
-------------

Before you start, make sure that:

- The ``flwr`` CLI is :doc:`installed <../how-to-install-flower>` locally.
- The Docker daemon is running.

Step 1: Set Up
--------------

#. Create a new Flower project (PyTorch):

   .. code-block:: bash

      $ flwr new quickstart-docker --framework PyTorch --username flower

      🔨 Creating Flower project quickstart-docker...
      🎊 Project creation successful.

      Use the following command to run your project:

            cd quickstart-docker
            pip install -e .
            flwr run

      $ cd quickstart-docker
      $ pip install -e .

#. Create a new Docker bridge network called ``flwr-network``:

   .. code-block:: bash

      $ docker network create --driver bridge flwr-network

   User-defined networks, such as ``flwr-network``, enable IP resolution of container names, a feature
   absent in the default bridge network. This simplifies quickstart example by avoiding the need to
   determine host IP first.

Step 2: Start the SuperLink
---------------------------

Open your terminal and run:

.. code-block:: bash
   :substitutions:

   $ docker run --rm \
         -p 9091:9091 -p 9092:9092 \
         --network flwr-network \
         --name superlink \
         --detach \
         flwr/superlink:|stable_flwr_version| --insecure

.. dropdown:: Understand the command

   * ``docker run``: This tells Docker to run a container from an image.
   * ``--rm``: Remove the container once it is stopped or the command exits.
   * | ``-p 9091:9091 -p 9092:9092``: Map port ``9091`` and ``9092`` of the container to the same port of
     | the host machine, allowing other services to access the Driver API on
     | ``http://localhost:9091`` and the Fleet API on ``http://localhost:9092``.
   * ``--network flwr-network``: Make the container join the network named ``flwr-network``.
   * ``--name superlink``: Assign the name ``superlink`` to the container.
   * ``--detach``: Run the container in the background, freeing up the terminal.
   * | :substitution-code:`flwr/superlink:|stable_flwr_version|`: The name of the image to be run and the specific
     | tag of the image. The tag :substitution-code:`|stable_flwr_version|` represents a :doc:`specific version <pin-version>` of the image.
   * | ``--insecure``: This flag tells the container to operate in an insecure mode, allowing
     | unencrypted communication.

Step 3: Start the SuperNode
---------------------------

Start two SuperNode containers.

#. Start the first container:

   .. code-block:: bash
      :substitutions:

      $ docker run --rm \
          -p 9094:9094 \
          --network flwr-network \
          --name supernode-1 \
          --detach \
          flwr/supernode:|stable_flwr_version|  \
          --insecure \
          --superlink superlink:9092 \
          --node-config "partition-id=0 num-partitions=2" \
          --supernode-address 0.0.0.0:9094 \
          --isolation process

   .. dropdown:: Understand the command

      * ``docker run``: This tells Docker to run a container from an image.
      * ``--rm``: Remove the container once it is stopped or the command exits.
      * | ``-p 9094:9094``: Map port ``9094`` of the container to the same port of
        | the host machine, allowing other services to access the SuperNode API on
        | ``http://localhost:9094``.
      * ``--network flwr-network``: Make the container join the network named ``flwr-network``.
      * ``--name supernode-1``: Assign the name ``supernode-1`` to the container.
      * ``--detach``: Run the container in the background, freeing up the terminal.
      * | ``flwr/supernode:|stable_flwr_version|``: This is the name of the image to be run and the specific tag
        | of the image.
      * | ``--insecure``: This flag tells the container to operate in an insecure mode, allowing
        | unencrypted communication.
      * | ``--superlink superlink:9092``: Connect to the SuperLink's Fleet API at the address
        | ``superlink:9092``.
      * | ``--node-config "partition-id=0 num-partitions=2"``: Set the partition ID to ``0`` and the
        | number of partitions to ``2`` for the SuperNode configuration.
      * | ``--supernode-address 0.0.0.0:9094``: Set the address and port number that the SuperNode
        | is listening on.
      * | ``--isolation process``: Tells the SuperNode that the ClientApp is created by separate
        | independent process. The SuperNode does not attempt to create it.

#. Start the second container:

   .. code-block:: shell
      :substitutions:

      $ docker run --rm \
          -p 9095:9095 \
          --network flwr-network \
          --name supernode-2 \
          --detach \
          flwr/supernode:|stable_flwr_version|  \
          --insecure \
          --superlink superlink:9092 \
          --node-config "partition-id=1 num-partitions=2" \
          --supernode-address 0.0.0.0:9095 \
          --isolation process

Step 4: Start the ClientApp
---------------------------

The ClientApp Docker image comes with a pre-installed version of Flower and serves as a base for
building your own ClientApp image. In order to install the FAB dependencies, you will need to create
a Dockerfile that extends the ClientApp image and installs the required dependencies.

#. Create a ClientApp Dockerfile called ``Dockerfile.clientapp`` and paste the following code into it:

   .. code-block:: dockerfile
      :caption: Dockerfile.clientapp
      :linenos:
      :substitutions:

      FROM flwr/clientapp:|stable_flwr_version|

      WORKDIR /app
      COPY pyproject.toml .
      RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
          && python -m pip install -U --no-cache-dir .

      ENTRYPOINT ["flwr-clientapp"]

   .. dropdown:: Understand the Dockerfile

      * | :substitution-code:`FROM flwr/clientapp:|stable_flwr_version|`: This line specifies that the Docker image
        | to be built from is the ``flwr/clientapp image``, version :substitution-code:`|stable_flwr_version|`.
      * | ``WORKDIR /app``: Set the working directory for the container to ``/app``.
        | Any subsequent commands that reference a directory will be relative to this directory.
      * | ``COPY pyproject.toml .``: Copy the ``pyproject.toml`` file
        | from the current working directory into the container's ``/app`` directory.
      * | ``RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml``: Remove the ``flwr`` dependency
        | from the ``pyproject.toml``.
      * | ``python -m pip install -U --no-cache-dir .``: Run the ``pip`` install command to
        | install the dependencies defined in the ``pyproject.toml`` file
        |
        | The ``-U`` flag indicates that any existing packages should be upgraded, and
        | ``--no-cache-dir`` prevents pip from using the cache to speed up the installation.
      * | ``ENTRYPOINT ["flwr-clientapp"]``: Set the command ``flwr-clientapp`` to be
        | the default command run when the container is started.

   .. important::

      Note that `flwr <https://pypi.org/project/flwr/>`__ is already installed in the ``flwr/clientapp``
      base image, so only other package dependencies such as ``flwr-datasets``, ``torch``, etc.,
      need to be installed. As a result, the ``flwr`` dependency is removed from the
      ``pyproject.toml`` after it has been copied into the Docker image (see line 5).

#. Next, build the ClientApp Docker image by running the following command in the directory where
   the Dockerfile is located:

   .. code-block:: bash

      $ docker build -f Dockerfile.clientapp -t flwr_clientapp:0.0.1 .

   .. note::

      The image name was set as ``flwr_clientapp`` with the tag ``0.0.1``. Remember that
      these values are merely examples, and you can customize them according to your requirements.

#. Start the first ClientApp container:

   .. code-block:: bash

      $ docker run --rm \
          --network flwr-network \
          --detach \
          flwr_clientapp:0.0.1  \
          --supernode supernode-1:9094

   .. dropdown:: Understand the command

      * ``docker run``: This tells Docker to run a container from an image.
      * ``--rm``: Remove the container once it is stopped or the command exits.
      * ``--network flwr-network``: Make the container join the network named ``flwr-network``.
      * ``--detach``: Run the container in the background, freeing up the terminal.
      * | ``flwr_clientapp:0.0.1``: This is the name of the image to be run and the specific tag
        | of the image.
      * | ``--supernode supernode-1:9094``: Connect to the SuperNode's Fleet API at the address
        | ``supernode-1:9094``.

#. Start the second ClientApp container:

   .. code-block:: shell

      $ docker run --rm \
          --network flwr-network \
          --detach \
          flwr_clientapp:0.0.1 \
          --supernode supernode-2:9095

Step 5: Start the SuperExec
---------------------------

The procedure for building and running a SuperExec image is almost identical to the ClientApp image.

Similar to the ClientApp image, you will need to create a Dockerfile that extends the SuperExec
image and installs the required FAB dependencies.

#. Create a SuperExec Dockerfile called ``Dockerfile.superexec`` and paste the following code in:

   .. code-block:: dockerfile
      :caption: Dockerfile.superexec
      :substitutions:

      FROM flwr/superexec:|stable_flwr_version|

      WORKDIR /app

      COPY pyproject.toml .
      RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
         && python -m pip install -U --no-cache-dir .

      ENTRYPOINT ["flower-superexec", "--executor", "flwr.superexec.deployment:executor"]

   .. dropdown:: Understand the Dockerfile

      * | :substitution-code:`FROM flwr/superexec:|stable_flwr_version|`: This line specifies that the Docker image
        | to be built from is the ``flwr/superexec image``, version :substitution-code:`|stable_flwr_version|`.
      * | ``WORKDIR /app``: Set the working directory for the container to ``/app``.
        | Any subsequent commands that reference a directory will be relative to this directory.
      * | ``COPY pyproject.toml .``: Copy the ``pyproject.toml`` file
        | from the current working directory into the container's ``/app`` directory.
      * | ``RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml``: Remove the ``flwr`` dependency
        | from the ``pyproject.toml``.
      * | ``python -m pip install -U --no-cache-dir .``: Run the ``pip`` install command to
        | install the dependencies defined in the ``pyproject.toml`` file
        |
        | The ``-U`` flag indicates that any existing packages should be upgraded, and
        | ``--no-cache-dir`` prevents pip from using the cache to speed up the installation.
      * | ``ENTRYPOINT ["flower-superexec"``: Set the command ``flower-superexec`` to be
        | the default command run when the container is started.
        |
        | ``"--executor", "flwr.superexec.deployment:executor"]`` Use the
        | ``flwr.superexec.deployment:executor`` executor to run the ServerApps.

#. Afterward, in the directory that holds the Dockerfile, execute this Docker command to
   build the SuperExec image:

   .. code-block:: bash

      $ docker build -f Dockerfile.superexec -t flwr_superexec:0.0.1 .


#. Start the SuperExec container:

   .. code-block:: bash

      $ docker run --rm \
         -p 9093:9093 \
          --network flwr-network \
          --name superexec \
          --detach \
          flwr_superexec:0.0.1 \
          --insecure \
          --executor-config superlink=\"superlink:9091\"

   .. dropdown:: Understand the command

      * ``docker run``: This tells Docker to run a container from an image.
      * ``--rm``: Remove the container once it is stopped or the command exits.
      * | ``-p 9093:9093``: Map port ``9093`` of the container to the same port of
        | the host machine, allowing you to access the SuperExec API on ``http://localhost:9093``.
      * ``--network flwr-network``: Make the container join the network named ``flwr-network``.
      * ``--name superexec``: Assign the name ``superexec`` to the container.
      * ``--detach``: Run the container in the background, freeing up the terminal.
      * | ``flwr_superexec:0.0.1``: This is the name of the image to be run and the specific tag
        | of the image.
      * | ``--insecure``: This flag tells the container to operate in an insecure mode, allowing
        | unencrypted communication.
      * | ``--executor-config superlink=\"superlink:9091\"``: Configure the SuperExec executor to
        | connect to the SuperLink running on port ``9091``.

Step 6: Run the Quickstart Project
----------------------------------

#. Add the following lines to the ``pyproject.toml``:

   .. code-block:: toml
      :caption: pyproject.toml

      [tool.flwr.federations.docker]
      address = "127.0.0.1:9093"
      insecure = true

#. Run the ``quickstart-docker`` project by executing the command:

   .. code-block:: bash

      $ flwr run . docker

#. Follow the SuperExec logs to track the execution of the run:

   .. code-block:: bash

      $ docker logs -f superexec

Step 7: Update the Application
------------------------------

#. Change the application code. For example, change the  ``seed`` in ``quickstart_docker/task.py``
   to ``43`` and save it:

   .. code-block:: python
      :caption: quickstart_docker/task.py

      # ...
      partition_train_test = partition.train_test_split(test_size=0.2, seed=43)
      # ...

#. Stop the current ClientApp containers:

   .. code-block:: bash

      $ docker stop $(docker ps -a -q  --filter ancestor=flwr_clientapp:0.0.1)

#. Rebuild the FAB and ClientApp image:

   .. code-block:: bash

      $ docker build -f Dockerfile.clientapp -t flwr_clientapp:0.0.1 .

#. Launch two new ClientApp containers based on the newly built image:

   .. code-block:: bash

      $ docker run --rm \
          --network flwr-network \
          --detach \
          flwr_clientapp:0.0.1  \
          --supernode supernode-1:9094
      $ docker run --rm \
          --network flwr-network \
          --detach \
          flwr_clientapp:0.0.1 \
          --supernode supernode-2:9095

#. Run the updated project:

   .. code-block:: bash

      $ flwr run . docker

Step 8: Clean Up
----------------

Remove the containers and the bridge network:

.. code-block:: bash

   $ docker stop $(docker ps -a -q  --filter ancestor=flwr_clientapp:0.0.1) \
      supernode-1 \
      supernode-2 \
      superexec \
      superlink
   $ docker network rm flwr-network

Where to Go Next
----------------

* :doc:`enable-tls`
* :doc:`persist-superlink-state`
* :doc:`tutorial-quickstart-docker-compose`
