How to build Docker Flower images locally
=========================================

Flower provides pre-made docker images on `Docker Hub <https://hub.docker.com/r/flwr/server/tags>`_
that include all necessary dependencies for running the server. You can also build your own custom
docker images from scratch with a different version of Python or Ubuntu if that is what you need.
In this guide, we will explain what images exist and how to build them locally.

Before we can start, we need to meet a few prerequisites in our local development environment.

#. Clone the flower repository.

    .. code-block:: bash

      $ git clone https://github.com/adap/flower.git && cd flower

#. Verify the Docker daemon is running.

    Please follow the first section on
    :doc:`Run Flower using Docker <how-to-run-flower-using-docker>`
    which covers this step in more detail.

Currently, Flower provides two images, a base image and a server image. There will also be a client
image soon. The base image, as the name suggests, contains basic dependencies that both the server
and the client need. This includes system dependencies, Python and Python tools. The server image is
based on the base image, but it additionally installs the Flower server using ``pip``.

The build instructions that assemble the images are located in the respective Dockerfiles. You
can find them in the subdirectories of ``src/docker``.

Both, base and server image are configured via build arguments. Through build arguments, we can make
our build more flexible. For example, in the base image, we can specify the version of Python to
install using the ``PYTHON_VERSION`` build argument. Some of the build arguments have default
values, others must be specified when building the image. All available build arguments for each
image are listed in one of the tables below.

Building the base image
-----------------------

.. list-table::
   :widths: 25 45 15 15
   :header-rows: 1

   * - Build argument
     - Description
     - Required
     - Example
   * - ``PYTHON_VERSION``
     - Version of ``python`` to be installed.
     - Yes
     - ``3.11``
   * - ``PIP_VERSION``
     - Version of ``pip`` to be installed.
     - Yes
     - ``23.0.1``
   * - ``SETUPTOOLS_VERSION``
     - Version of ``setuptools`` to be installed.
     - Yes
     - ``69.0.2``
   * - ``UBUNTU_VERSION``
     - Version of the official Ubuntu Docker image.
     - Defaults to ``22.04``.
     -

The following example creates a base image with Python 3.11.0, pip 23.0.1 and setuptools 69.0.2:

.. code-block:: bash

  $ cd src/docker/base/
  $ docker build \
    --build-arg PYTHON_VERSION=3.11.0 \
    --build-arg PIP_VERSION=23.0.1 \
    --build-arg SETUPTOOLS_VERSION=69.0.2 \
    -t flwr_base:0.1.0 .

The name of image is ``flwr_base`` and the tag ``0.1.0``. Remember that the build arguments as well
as the name and tag can be adapted to your needs. These values serve as examples only.

Building the server image
-------------------------

.. list-table::
   :widths: 25 45 15 15
   :header-rows: 1

   * - Build argument
     - Description
     - Required
     - Example
   * - ``BASE_REPOSITORY``
     - The repository name of the base image.
     - Defaults to ``flwr/server``.
     -
   * - ``BASE_IMAGE_TAG``
     - The image tag of the base image.
     - Defaults to ``py3.11-ubuntu22.04``.
     -
   * - ``FLWR_VERSION``
     - Version of Flower to be installed.
     - Yes
     - ``1.7.0``

The following example creates a server image with the official Flower base image py3.11-ubuntu22.04
and Flower 1.7.0:

.. code-block:: bash

  $ cd src/docker/server/
  $ docker build \
    --build-arg BASE_IMAGE_TAG=py3.11-ubuntu22.04 \
    --build-arg FLWR_VERSION=1.7.0 \
    -t flwr_server:0.1.0 .

The name of image is ``flwr_server`` and the tag ``0.1.0``. Remember that the build arguments as well
as the name and tag can be adapted to your needs. These values serve as examples only.

If you want to use your own base image instead of the official Flower base image, all you need to do
is set the ``BASE_REPOSITORY`` and ``BASE_IMAGE_TAG`` build arguments. The value of
``BASE_REPOSITORY`` must match the name of your image and the value of ``BASE_IMAGE_TAG`` must match
the tag of your image.

.. code-block:: bash

  $ cd src/docker/server/
  $ docker build \
    --build-arg BASE_REPOSITORY=flwr_base \
    --build-arg BASE_IMAGE_TAG=0.1.0 \
    --build-arg FLWR_VERSION=1.7.0 \
    -t flwr_server:0.1.0 .

After creating the image, we can test whether the image is working:

.. code-block:: bash

  $ docker run --rm flwr_server:0.1.0 --help
