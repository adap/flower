:og:description: Learn how to configure environment variables in Flower Docker containers using the -e flag to customize settings like telemetry and logging for your federated learning setup.

.. title:: How-to Tutorial: Set Environment Variables in Flower Docker Containers

.. meta::
   :description: Learn how to configure environment variables in Flower Docker containers using the -e flag to customize settings like telemetry and logging for your federated learning setup.

Set Environment Variables
=========================

To set a variable inside a Docker container, you can use the ``-e <name>=<value>`` flag.
Multiple ``-e`` flags can be used to set multiple environment variables for a container.

Example
-------

.. code-block:: bash
   :substitutions:

   $ docker run -e FLWR_TELEMETRY_ENABLED=0 -e FLWR_TELEMETRY_LOGGING=0 \
        --rm flwr/superlink:|stable_flwr_version|
