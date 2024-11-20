Set Environment Variables
=========================

To set a variable inside a Docker container, you can use the ``-e <name>=<value>`` flag.
Multiple ``-e`` flags can be used to set multiple environment variables for a container.

Example
-------

.. code-block:: bash
    :substitutions:

    $ docker run -e FLWR_TELEMETRY_ENABLED=0 -e FLWR_TELEMETRY_LOGGING=0 \
         --rm flwr/superlink:|stable_flwr_version| \
         <additional-args>
