Pinning a Docker Image to a Specific Version
============================================

It may happen that we update the images behind the tags. Such updates usually include security
updates of system dependencies that should not change the functionality of Flower. However, if you
want to ensure that you always use the same image, you can specify the hash of the image instead of
the tag.

Example
-------

The following command returns the current image hash referenced by the
:substitution-code:`superlink:|latest_version_docker|` tag:

.. code-block:: bash
   :substitutions:

   $ docker inspect --format='{{index .RepoDigests 0}}' flwr/superlink:|latest_version_docker|
   flwr/superlink@sha256:|latest_version_docker_sha|

Next, we can pin the hash when running a new SuperLink container:

.. code-block:: bash
   :substitutions:

   $ docker run \
        --rm flwr/superlink@sha256:|latest_version_docker_sha|
