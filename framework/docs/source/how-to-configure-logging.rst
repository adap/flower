:og:description: Configure the logging level for your Flower processes.
.. meta::
    :description: Configure the logging level for your Flower processes.

Configure logging
=================

By default, the Flower logger uses logging level ``INFO``. This can be changed via the
``PYTHONLOGLEVEL`` environment variable to any other levels that Python's `logging
module <https://docs.python.org/3/library/logging.html#logging-levels>`_ supports. For
example, launch your ``SuperLink`` showing ``DEBUG`` logs:

.. code-block:: shell
    :emphasize-lines: 2,11,12

    # Launch the SuperLink with TLS (or use --insecure)
    PYTHONLOGLEVEL=DEBUG flower-superlink \
        --ssl-ca-certfile certificates/ca.crt \
        --ssl-certfile certificates/server.pem \
        --ssl-keyfile certificates/server.key

    INFO 2025-01-27 11:46:41,690:      Starting Flower SuperLink
    INFO 2025-01-27 11:46:41,697:      Flower Deployment Engine: Starting Exec API on 0.0.0.0:9093
    INFO 2025-01-27 11:46:41,724:      Flower ECE: Starting ServerAppIo API (gRPC-rere) on 0.0.0.0:9091
    INFO 2025-01-27 11:46:41,728:      Flower ECE: Starting Fleet API (gRPC-rere) on 0.0.0.0:9092
    DEBUG 2025-01-27 11:46:41,730:     Started flwr-serverapp scheduler thread.
    DEBUG 2025-01-27 11:46:41,730:     Using InMemoryState

.. note::

    You can make use of the ``PYTHONLOGLEVEL`` environment variable when executing other
    Flower commands to provision the different components in a Flower Federation (see
    :doc:`how-to-run-flower-with-deployment-engine`) or using the `flwr CLI
    <ref-api-cli.html>`_.
