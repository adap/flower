:og:description: Pin Flower Docker images to specific versions using image digests, ensuring consistent deployments while receiving essential security updates.
.. meta::
    :description: Pin Flower Docker images to specific versions using image digests, ensuring consistent deployments while receiving essential security updates.

##########################################
 Pin a Docker Image to a Specific Version
##########################################

It may happen that we update the images behind the tags. Such updates usually include
security updates of system dependencies that should not change the functionality of
Flower. However, if you want to ensure that you use a fixed version of the Docker image
in your deployments, you can `specify the digest
<https://docs.docker.com/reference/cli/docker/image/pull/#pull-an-image-by-digest-immutable-identifier>`_
of the image instead of the tag.

*********
 Example
*********

The following command returns the current image digest referenced by the
:substitution-code:`superlink:|stable_flwr_version|` tag:

.. code-block:: bash
    :substitutions:

    $ docker pull flwr/superlink:|stable_flwr_version|
    $ docker inspect --format='{{index .RepoDigests 0}}' flwr/superlink:|stable_flwr_version|

This will output

.. code-block:: bash
    :substitutions:

    flwr/superlink@sha256:|stable_flwr_superlink_docker_digest|

Next, we can pin the digest when running a new SuperLink container:

.. code-block:: bash
    :substitutions:

    $ docker run \
         --rm flwr/superlink@sha256:|stable_flwr_superlink_docker_digest| \
         <additional-args>
