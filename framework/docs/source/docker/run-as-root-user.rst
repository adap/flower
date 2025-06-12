:og:description: Run Flower Docker containers with root privileges for specific build tasks, adhering to security best practices for production environments.
.. meta::
    :description: Run Flower Docker containers with root privileges for specific build tasks, adhering to security best practices for production environments.

Run with Root User Privileges
=============================

Flower Docker images, by default, run with a non-root user (username/groupname: ``app``,
UID/GID: ``49999``). Using root user is **not recommended** unless it is necessary for
specific tasks during the build process.

Always make sure to run the container as a non-root user in production to maintain
security best practices.

Run a Container with Root User Privileges
-----------------------------------------

Run the Docker image with the ``-u`` flag and specify ``root`` as the username:

.. code-block:: bash
    :substitutions:

    $ docker run --rm -u root flwr/superlink:|stable_flwr_version| <additional-args>

This command will run the Docker container with root user privileges.

Run the Build Process with Root User Privileges
-----------------------------------------------

If you want to switch to the root user during the build process of the Docker image to
install missing system dependencies, you can use the ``USER root`` directive within your
Dockerfile.

.. code-block:: dockerfile
    :caption: SuperNode Dockerfile
    :substitutions:

    FROM flwr/supernode:|stable_flwr_version|

    # Switch to root user
    USER root

    # Install missing dependencies (requires root access)
    RUN apt-get update && apt-get install -y <required-package-name>

    # Switch back to non-root user app
    USER app

    # Continue with your Docker image build process
    # ...
