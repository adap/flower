How to build Docker Flower images locally
=========================================

Flower provides pre-made docker images on `Docker Hub <https://hub.docker.com/r/flwr/server/tags>`_
that include all necessary dependencies for running the server. You can also build your own custom
docker images from scratch with a different version of Python or Ubuntu if that is what you need.
In this guide, we will explain what images exist and how to build them locally.

Before we can start, we need to meet a few prerequisites in our local development environment.

1. Clone the flower repository.

.. code-block:: bash

  $ git clone https://github.com/adap/flower.git && cd flower

2. Verify the Docker daemon is running.

Please follow the first section on
`Run Flower in Docker <https://flower.dev/docs/framework/how-to-run-flower-in-docker>`_
which covers this step in more detail.

Currently, Flower provides two images, a base image and a server image. There will also be a client
image soon. The base image, as the name suggests, contains basic dependencies that both the server
and the client need. This includes system dependencies, Python and Python tools. The server image is
based on the base image, but it additionally installs the Flower server using ``pip``.

Building the base image
-----------------------

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Build argument
     - Description
     - Required
   * - ``PYTHON_VERSION``
     - Version of ``python`` to be installed.
     - Yes
   * - ``PIP_VERSION``
     - Version of ``pip`` to be installed.
     - Yes
   * - ``SETUPTOOLS_VERSION``
     - Version of ``setuptools`` to be installed.
     - Yes
   * - ``UBUNTU_VERSION``
     - Version of the official Ubuntu Docker image.
     - Defaults to ``22.04``

.. code-block:: bash

  $ cd src/docker/base/
  $ docker build \
    --build-arg PYTHON_VERSION=3.11.0 \
    --build-arg PIP_VERSION=23.0.1 \
    --build-arg SETUPTOOLS_VERSION=69.0.2 \
    --build-arg UBUNTU_VERSION=22.04 \
    -t local:my-base-image .

Building the server image
-------------------------

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Build argument
     - Description
     - Required
   * - ``BASE_REPOSITORY``
     - The repository name of the base image.
     - Defaults to ``flwr/server``
   * - ``BASE_IMAGE_TAG``
     - The image tag of the base image.
     - Defaults to ``py3.11-ubuntu22.04``
   * - ``FLWR_VERSION``
     - Version of Flower to be installed.
     - Yes

.. code-block:: bash

  $ cd src/docker/server/
  $ docker build \
    --build-arg BASE_IMAGE_TAG=py3.11-ubuntu22.04 \
    --build-arg FLWR_VERSION=1.6.0 \
    -t local:my-server-image .

  $ docker run --rm local:my-server-image --help

note:
- if you want to use a different your own base image, you will need to set he `BASE_REPOSITORY`.

.. code-block:: bash

  $ cd src/docker/server/
  $ docker build \
    --build-arg BASE_REPOSITORY=local \
    --build-arg BASE_IMAGE_TAG=my-base-image \
    --build-arg FLWR_VERSION=1.6.0 \
    -t local:my-server-image .

  $ docker run --rm local:my-server-image --help
