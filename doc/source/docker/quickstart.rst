Quickstart
==========

This quickstart amis to guide you through the process of containerizing a Flower project and
running it end to end using Docker.

This tutorial does not use production-ready settings, so you can focus on understanding the basic
workflow without being overwhelmed by additional configurations.

Prerequisites
-------------

Before you start, make sure that:

- The ``flwr`` CLI is installed locally.
- The Docker daemon is running.

Step 1: Set Up
--------------

#. Create a new Flower project (PyTorch):

   .. code-block:: bash

      $ flwr new quickstart-docker

      💬 Please provide your Flower username: flower

      💬 Please select ML framework by typing in the number

      [ 0] FlowerTune
      [ 1] HF
      [ 2] JAX
      [ 3] MLX
      [ 4] NumPy
      [ 5] PyTorch
      [ 6] TensorFlow
      [ 7] sklearn


      : 5

      🔨 Creating Flower project quickstart-docker...
      🎊 Project creation successful.

      Use the following command to run your project:

            cd quickstart-docker
            pip install -e .
            flwr run

      $ cd quickstart-docker
      $ pip install -e .

#. Create a new Docker bridge network called ``flwr``:

   .. code-block:: bash

      $ docker network create --driver bridge flwr

   User-defined networks, such as ``flwr``, enable IP resolution of container names, a feature
   absent in the default bridge network. This simplifies quickstart by avoiding the need to
   determine host IP first.

Step 2: Start the SuperLink
---------------------------

Open your terminal and run:

.. code-block:: bash
   :substitutions:

   $ docker run --rm \
         -p 9091:9091 -p 9092:9092 \
         --network flwr \
         --name superlink \
         --detach \
         flwr/superlink:|latest_version_docker| --insecure

.. dropdown:: Understand the command

   * ``docker run``: This tells Docker to run a container from an image.
   * ``--rm``: Remove the container once it is stopped or the command exits.
   * | ``-p 9091:9091 -p 9092:9092``: Map port ``9091`` and ``9092`` of the container to the same port of
     | the host machine, allowing you to access the Driver API on ``http://localhost:9091`` and
     | the Fleet API on ``http://localhost:9092``.
   * ``--network flwr``: Make the container join the network named ``flwr``.
   * ``--name superlink``: Assign the name ``superlink`` to the container.
   * ``--detach``: Run the container in the background, freeing up the terminal.
   * | :substitution-code:`flwr/superlink:|latest_version_docker|`: The name of the image to be run and the specific
     | tag of the image. The tag :substitution-code:`|latest_version_docker|` represents a specific version of the image.
   * | ``--insecure``: This flag tells the container to operate in an insecure mode, allowing
     | unencrypted communication.

Step 3: Start the SuperNode
---------------------------

The SuperNode Docker image comes with a pre-installed version of Flower and serves as a base for
building your own SuperNode image.

#. Create a SuperNode Dockerfile called ``Dockerfile.supernode`` and paste the following code into it:

   .. code-block:: dockerfile
      :caption: Dockerfile.supernode
      :substitutions:

      FROM flwr/supernode:|latest_version_docker|

      WORKDIR /app
      RUN python -m pip install -U --no-cache-dir \

      ENTRYPOINT ["flower-supernode"]

#. Copy the dependencies of the Flower project. These can be found in the ``project`` ``dependencies``
   section of the ``pyproject.toml`` file. As seen in the example below, to copy the dependencies,
   you would copy the lines 7-9 in the ``pyproject.toml`` file.

   .. important::

      Note that `flwr <https://pypi.org/project/flwr/>`__ is already installed in the ``flwr/supernode``
      base image, so you only need to copy other package dependencies such as ``flwr-datasets``,
      ``torch``, etc.

   .. code-block:: toml
      :linenos:
      :emphasize-lines: 7-9
      :caption: pyproject.toml

      ...
      [project]
      name = "quickstart-docker"
      version = "1.0.0"
      dependencies = [
          "flwr[simulation]>=1.10.0",
          "flwr-datasets[vision]>=0.0.2,<1.0.0",
          "torch==2.2.1",
          "torchvision==0.17.1",
      ]
      ...

   After the line 4, paste the dependencies copied from the previous step.
   Make sure to remove the comma at the end of each line and add a space and a backslash at the
   end, except the last one.

   .. code-block:: dockerfile
      :linenos:
      :emphasize-lines: 5-7
      :caption: Dockerfile.supernode
      :substitutions:

      FROM flwr/supernode:|latest_version_docker|

      WORKDIR /app
      RUN python -m pip install -U --no-cache-dir \
          "flwr-datasets[vision]>=0.0.2,<1.0.0" \
          "torch==2.2.1" \
          "torchvision==0.17.1"

      ENTRYPOINT ["flower-supernode"]

   .. dropdown:: Understand the Dockerfile

      * :substitution-code:`FROM flwr/supernode:|latest_version_docker|`: This line specifies that
        | the Docker image to be built from is the ``flwr/supernode image``, version
        | :substitution-code:`|latest_version_docker|`.
      * | ``WORKDIR /app``: Set the working directory for the container to ``/app``.
        | Any subsequent commands that reference a directory will be relative to this directory.
      * | ``RUN python -m pip install -U --no-cache-dir \``: Run the ``pip`` install command to
        | install the required packages.
        |
        | The ``-U`` flag indicates that any existing packages should be upgraded, and
        | ``--no-cache-dir`` prevents pip from using the cache to speed up the installation.
        |
        | The packages listed after the backslash are the ones to be installed.
      * | ``ENTRYPOINT ["flower-supernode"]``: Set the command ``flower-supernode`` to be
        | the default command run when the container is started.

#. Next, build the SuperNode Docker image by running the following command in the directory where
   Dockerfile is located:

   .. code-block:: bash

      $ docker build -f Dockerfile.supernode -t flwr_supernode:0.0.1 .

   .. Note::

      The image name was set as ``flwr_supernode`` with the tag ``0.0.1``. Remember that
      these values are merely examples, and you can customize them according to your requirements.

#. Start the first SuperNode container:

   .. code-block:: bash

      $ docker run --rm \
          --network flwr \
          --detach \
          flwr_supernode:0.0.1 \
          --insecure \
          --superlink superlink:9092 \
          --node-config \
          partition-id=0,num-partitions=2

   .. dropdown:: Understand the command

      * ``docker run``: This tells Docker to run a container from an image.
      * ``--rm``: Remove the container once it is stopped or the command exits.
      * ``--network flwr``: Make the container join the network named ``flwr``.
      * ``--detach``: Run the container in the background, freeing up the terminal.
      * | ``flwr_supernode:0.0.1``: This is the name of the image to be run and the specific tag
        | of the image.
      * | ``--insecure``: This flag tells the container to operate in an insecure mode, allowing
        | unencrypted communication.
      * | ``--superlink superlink:9092``: Connect to the SuperLinks Fleet API on the address
        | ``superlink:9092``.

#. Start the second SuperNode container:

   .. code-block:: shell

      $ docker run --rm \
          --network flwr \
          --detach \
          flwr_supernode:0.0.1 \
          --insecure \
          --superlink superlink:9092 \
          --node-config \
          partition-id=1,num-partitions=2

Step 4: Start the SuperExec
---------------------------

The procedure for building and running a SuperExec image is almost identical to the SuperNode image.

Similar to the SuperNode image, the SuperExec Docker image comes with a pre-installed version of
Flower and serves as a base for building your own SuperExec image.

#. Create a SuperNode Dockerfile called ``Dockerfile.superexec`` and paste the following code in:

   .. code-block:: dockerfile
      :caption: Dockerfile.superexec
      :substitutions:

      FROM flwr/superexec:|latest_version_docker|

      WORKDIR /app
      RUN python -m pip install -U --no-cache-dir \

      ENTRYPOINT ["flower-superexec", "--executor", "flwr.superexec.deployment:executor"]

#. As you did for the SuperNode image, copy the dependencies of the Flower project.

   .. code-block:: toml
      :linenos:
      :emphasize-lines: 7-9
      :caption: pyproject.toml

      ...
      [project]
      name = "quickstart-docker"
      version = "1.0.0"
      dependencies = [
          "flwr[simulation]>=1.10.0",
          "flwr-datasets[vision]>=0.0.2,<1.0.0",
          "torch==2.2.1",
          "torchvision==0.17.1",
      ]
      ...

   After the line 4, paste the dependencies copied from the previous step.
   Make sure to remove the comma at the end of each line and add a space and a backslash at the
   end, except the last one.

   .. code-block:: dockerfile
      :linenos:
      :emphasize-lines: 5-7
      :caption: Dockerfile.superexec
      :substitutions:

      FROM flwr/superexec:|latest_version_docker|

      WORKDIR /app
      RUN python -m pip install -U --no-cache-dir \
          "flwr-datasets[vision]>=0.0.2,<1.0.0" \
          "torch==2.2.1" \
          "torchvision==0.17.1"

      ENTRYPOINT ["flower-superexec", "--executor", "flwr.superexec.deployment:executor"]

   .. dropdown:: Understand the Dockerfile

      * :substitution-code:`FROM flwr/superexec:|latest_version_docker|`: This line specifies that
        | the Docker image to be built from is the ``flwr/superexec image``, version
        | :substitution-code:`|latest_version_docker|`.
      * | ``WORKDIR /app``: Set the working directory for the container to ``/app``.
        | Any subsequent commands that reference a directory will be relative to this directory.
      * | ``RUN python -m pip install -U --no-cache-dir \``: Run the ``pip`` install command to
        | install the required packages.
        |
        | The ``-U`` flag indicates that any existing packages should be upgraded, and
        | ``--no-cache-dir`` prevents pip from using the cache to speed up the installation.
        |
        | The packages listed after the backslash are the ones to be installed.
      * | ``ENTRYPOINT ["flower-superexec" ``: Set the command ``flower-superexec`` to be
        | the default command run when the container is started.
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
          --network flwr \
          --detach \
          flwr_superexec:0.0.1 \
          --insecure \
          --executor-config \
          superlink=superlink:9091

   .. dropdown:: Understand the command

      * ``docker run``: This tells Docker to run a container from an image.
      * ``--rm``: Remove the container once it is stopped or the command exits.
      * | ``-p 9093:9093``: Map port ``9093`` of the container to the same port of
        | the host machine, allowing you to access the SuperExec API on ``http://localhost:9093``.
      * ``--network flwr``: Make the container join the network named ``flwr``.
      * ``--detach``: Run the container in the background, freeing up the terminal.
      * | ``flwr_superexec:0.0.1``: This is the name of the image to be run and the specific tag
        | of the image.
      * | ``--insecure``: This flag tells the container to operate in an insecure mode, allowing
        | unencrypted communication.
      * | ``--executor-config superlink=superlink:9091``: Configure the SuperExec executor to
        | connect to the SuperLink running on port ``9091``.

Step 5: Run the Quickstart Project
----------------------------------

#. After all the Flower components are up and running, run the quickstart-docker project
   by executing the command:

   .. code-block:: bash

      $ flwr run

#. Wait until the run is complete

TODO:

* add flwr logs


Step 6: Clean Up
----------------

#. Remove the containers and the bridge network:

   .. code-block:: bash

      $ docker stop $(docker ps -a -q  --filter ancestor=flwr_supernode:0.0.1) \
         $(docker ps -a -q  --filter ancestor=flwr_superexec:0.0.1) \
         superlink
      $ docker network rm flwr

Where to Go Next
----------------

* :doc:`Enabling TLS for secure connections <tls>`
* :doc:`Persist the state of the SuperLink <persist-state>`
