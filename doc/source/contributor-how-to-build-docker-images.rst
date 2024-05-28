How to build Docker Flower images locally
=========================================

Flower provides pre-made docker images on `Docker Hub <https://hub.docker.com/u/flwr>`_
that include all necessary dependencies for running the SuperLink, SuperNode or ServerApp.
You can also build your own custom docker images from scratch with a different version of Python
or Linux distribution (Ubuntu/Alpine) if that is what you need. In this guide, we will explain what
images exist and how to build them locally.

Before we can start, we need to meet a few prerequisites in our local development environment.

#. Clone the flower repository.

    .. code-block:: bash

      $ git clone https://github.com/adap/flower.git && cd flower

#. Verify the Docker daemon is running.

    Please follow the first section on
    :doc:`Run Flower using Docker <how-to-run-flower-using-docker>`
    which covers this step in more detail.


The build instructions that assemble the images are located in the respective Dockerfiles. You
can find them in the subdirectories of ``src/docker``.

Flower Docker images are configured via build arguments. Through build arguments, we can make the
creation of images more flexible. For example, in the base image, we can specify the version of
Python to install using the ``PYTHON_VERSION`` build argument. Some of the build arguments have
default values, others must be specified when building the image. All available build arguments for
each image are listed in one of the tables below.

Building the base image
-----------------------

.. list-table::
   :widths: 25 45 15 15
   :header-rows: 1

   * - Build argument
     - Description
     - Required
     - Example
   * - ``DISTRO``
     - The Linux distribution to use as the base image.
     - Defaults to ``ubuntu`` for Ubuntu and ``alpine`` for Alpine.
     -
   * - ``DISTRO_VERSION``
     - Version of the Linux distribution.
     - Defaults to ``22.04`` for Ubuntu and ``3.19`` for Alpine.
     -
   * - ``PYTHON_VERSION``
     - Version of ``python`` to be installed.
     - Defaults to ``3.11``
     -
   * - ``PIP_VERSION``
     - Version of ``pip`` to be installed.
     - Yes
     - ``23.0.1``
   * - ``SETUPTOOLS_VERSION``
     - Version of ``setuptools`` to be installed.
     - Yes
     - ``69.0.2``
   * - ``FLWR_VERSION``
     - Version of Flower to be installed.
     - Yes
     - ``1.8.0``
   * - ``FLWR_PACKAGE``
     - The Flower package to be installed (``flwr`` or ``flwr-nightly``).
     - Defaults to ``flwr``
     -


The following example creates a base Ubuntu 22.02/Alpine 3.19 image with Python 3.11.0, pip 23.0.1,
setuptools 69.0.2 and Flower 1.8.0:

.. code-block:: bash

  $ cd src/docker/base/<ubuntu|alpine>
  $ docker build \
    --build-arg FLWR_VERSION=1.8.0 \
    --build-arg PIP_VERSION=23.0.1 \
    --build-arg SETUPTOOLS_VERSION=69.0.2 \
    -t flwr_base:0.1.0 .

The name of image is ``flwr_base`` and the tag ``0.1.0``. Remember that the build arguments as well
as the name and tag can be adapted to your needs. These values serve as examples only.

Building the SuperLink/SuperNode or ServerApp image
---------------------------------------------------

.. list-table::
   :widths: 25 45 15 15
   :header-rows: 1

   * - Build argument
     - Description
     - Required
     - Example
   * - ``BASE_REPOSITORY``
     - The repository name of the base image.
     - Defaults to ``flwr/base``.
     -
   * - ``BASE_IMAGE``
     - The Tag of the Flower base image.
     - Yes
     - ``1.8.0-py3.10-ubuntu22.04``

The following example creates a SuperLink/SuperNode or ServerApp image with the official Flower
base image 1.8.0-py3.10-ubuntu22.04:

.. code-block:: bash

  $ cd src/docker/<superlink|supernode|serverapp>/
  $ docker build \
    --build-arg BASE_IMAGE=1.8.0-py3.10-ubuntu22.04 \
    -t flwr_superlink:0.1.0 .


If you want to use your own base image instead of the official Flower base image, all you need to do
is set the ``BASE_REPOSITORY`` build argument.

.. code-block:: bash

  $ cd src/docker/superlink/
  $ docker build \
    --build-arg BASE_REPOSITORY=flwr_base \
    --build-arg BASE_IMAGE=0.1.0
    -t flwr_superlink:0.1.0 .

After creating the image, we can test whether the image is working:

.. code-block:: bash

  $ docker run --rm flwr_superlink:0.1.0 --help
