Quickstart with Docker Compose
==============================

This quickstart shows you how to set up Flower using Docker Compose in a single command,
allowing you to focus on developing your application without worrying about the underlying
infrastructure.

You will also learn how to easily enable TLS encryption and persist application state locally,
giving you the freedom to choose the configuration that best suits your project's needs.

Prerequisites
-------------

Before you start, make sure that:

- The ``flwr`` CLI is :doc:`installed <../how-to-install-flower>` locally.
- The Docker daemon is running.
- Docker Compose is `installed <https://docs.docker.com/compose/install/>`_.

Step 1: Set Up
--------------

#. Clone the Docker Compose ``complete`` directory:

   .. code-block:: bash

      $ git clone --depth=1 https://github.com/adap/flower.git _tmp \
                  && mv _tmp/src/docker/complete . \
                  && rm -rf _tmp && cd complete

#. Create a new Flower project (PyTorch):

   .. code-block:: bash

      $ flwr new quickstart-compose --framework PyTorch --username flower

#. Export the path of the newly created project. The path should be relative to the location of the
   Docker Compose files:

   .. code-block:: bash

      $ export PROJECT_DIR=quickstart-compose

   Setting the ``PROJECT_DIR`` helps Docker Compose locate the ``pyproject.toml`` file, allowing
   it to install dependencies in the SuperExec and SuperNode images correctly.

Step 2: Run Flower in insecure mode
-----------------------------------

To begin, start Flower with the most basic configuration. In this setup, Flower
will run without TLS and without persisting the state.

.. note::

   Without TLS, the data sent between the services remains **unencrypted**. Use it only for development
   purposes.

   For production-oriented use cases, :ref:`enable TLS<TLS>` for secure data transmission.

Open your terminal and run:

.. code-block:: bash

   $ docker compose -f compose.yml up --build -d

.. dropdown:: Understand the command

   * ``docker compose``: The Docker command to run the Docker Compose tool.
   * ``-f compose.yml``: Specify the YAML file that contains the basic Flower service definitions.
   * ``--build``: Rebuild the images for each service if they don't already exist.
   * ``-d``: Detach the containers from the terminal and run them in the background.

Step 3: Run the Quickstart Project
----------------------------------

Now that the Flower services have been started via Docker Compose, it is time to run the
quickstart example.

To ensure the ``flwr`` CLI connects to the SuperExec, you need to specify the SuperExec addresses
in the ``pyproject.toml`` file.

#. Add the following lines to the ``quickstart-compose/pyproject.toml``:

   .. code-block:: toml
      :caption: quickstart-compose/pyproject.toml

      [tool.flwr.federations.docker-compose]
      address = "127.0.0.1:9093"
      insecure = true

#. Execute the command to run the quickstart example:

   .. code-block:: bash

      $ flwr run quickstart-compose docker-compose

#. Monitor the SuperExec logs and wait for the summary to appear:

   .. code-block:: bash

      $ docker compose logs superexec -f

Step 4: Update the Application
------------------------------

In the next step, change the application code.

#. For example, go to the ``task.py`` file in the ``quickstart-compose/quickstart_compose/``
   directory and add a ``print`` call in the ``get_weights`` function:

   .. code-block:: python
      :caption: quickstart-compose/quickstart_compose/task.py

      # ...
      def get_weights(net):
          print("Get weights")
          return [val.cpu().numpy() for _, val in net.state_dict().items()]
      # ...

#. Rebuild and restart the services.

   .. note::

      If you have modified the dependencies listed in your ``pyproject.toml`` file, it is essential
      to rebuild images.

      If you haven't made any changes, you can skip this step.

   Run the following command to rebuild and restart the services:

   .. code-block:: bash

      $ docker compose -f compose.yml up --build -d

#. Run the updated quickstart example:

   .. code-block:: bash

      $ flwr run quickstart-compose docker-compose
      $ docker compose logs superexec -f

   In the SuperExec logs, you should find the ``Get weights`` line:

   .. code-block::
      :emphasize-lines: 9

      superexec-1  | INFO :      Starting Flower SuperExec
      superexec-1  | WARNING :   Option `--insecure` was set. Starting insecure HTTP server.
      superexec-1  | INFO :      Starting Flower SuperExec gRPC server on 0.0.0.0:9093
      superexec-1  | INFO :      ExecServicer.StartRun
      superexec-1  | 🎊 Successfully installed quickstart-compose to /app/.flwr/apps/flower/quickstart-compose/1.0.0.
      superexec-1  | INFO :      Created run -6767165609169293507
      superexec-1  | INFO :      Started run -6767165609169293507
      superexec-1  | WARNING :   Option `--insecure` was set. Starting insecure HTTP client connected to superlink:9091.
      superexec-1  | Get weights
      superexec-1  | INFO :      Starting Flower ServerApp, config: num_rounds=3, no round_timeout

Step 5: Persisting the SuperLink State
--------------------------------------

In this step, Flower services are configured to persist the state of the SuperLink service,
ensuring that it maintains its state even after a restart.

.. note::

    When working with Docker Compose on Linux, you may need to create the ``state`` directory first
    and change its ownership to ensure proper access and permissions.

    For more information, consult the following page: :doc:`persist-superlink-state`.

#. Run the command:

   .. code-block:: bash

      $ docker compose -f compose.yml -f with-state.yml up --build -d

   .. dropdown:: Understand the command

      * ``docker compose``: The Docker command to run the Docker Compose tool.
      * ``-f compose.yml``: Specify the YAML file that contains the basic Flower service definitions.
      * | ``-f with-state.yml``: Specifies the path to an additional Docker Compose file that
        | contains the configuration for persisting the SuperLink state.
        |
        | Docker merges Compose files according to `merging rules <https://docs.docker.com/compose/multiple-compose-files/merge/#merging-rules>`_.
      * ``--build``: Rebuild the images for each service if they don't already exist.
      * ``-d``: Detach the containers from the terminal and run them in the background.

#. Rerun the ``quickstart-compose`` project:

   .. code-block:: bash

      $ flwr run quickstart-compose docker-compose

#. Check the content of the ``state`` directory:

   .. code-block:: bash

      $ ls state/
      state.db

   You should see a ``state.db`` file in the ``state`` directory. If you restart the service, the
   state file will be used to restore the state from the previously saved data. This ensures that
   the data persists even if the containers are stopped and started again.

.. _TLS:

Step 6: Run Flower with TLS
---------------------------

#. To demonstrate how to enable TLS, generate self-signed certificates using the ``certs.yml``
   Compose file.

   .. important::

      These certificates should be used only for development purposes.

      For production environments, use a service like `Let's Encrypt <https://letsencrypt.org/>`_
      to obtain your certificates.

   Run the command:

   .. code-block:: bash

      $ docker compose -f certs.yml up --build

#. Add the following lines to the ``quickstart-compose/pyproject.toml``:

   .. code-block:: toml
      :caption: quickstart-compose/pyproject.toml

      [tool.flwr.federations.docker-compose-tls]
      address = "127.0.0.1:9093"
      root-certificates = "superexec-certificates/ca.crt"

#. Restart the services with TLS enabled:

   .. code-block:: bash

      $ docker compose -f compose.yml -f with-tls.yml up --build -d

#. Rerun the ``quickstart-compose`` project:

   .. code-block:: bash

      $ flwr run quickstart-compose docker-compose-tls
      $ docker compose logs superexec -f

Step 7: Add another SuperNode
-----------------------------

You can add more SuperNodes by duplicating the SuperNode definition in the ``compose.yml`` file.

Just make sure to give each new SuperNode service a unique service name like ``supernode-3``, ``supernode-4``, etc.

In ``compose.yml``, add the following:

.. code-block:: yaml
   :caption: compose.yml

   services:
     # other service definitions

     supernode-3:
       user: root
       deploy:
         resources:
           limits:
             cpus: "2"
       command:
         - --superlink
         - superlink:9092
         - --insecure
       depends_on:
         - superlink
       volumes:
         - apps-volume:/app/.flwr/apps/:ro
       build:
         context: ${PROJECT_DIR:-.}
         dockerfile_inline: |
           FROM flwr/supernode:${FLWR_VERSION:-1.10.0}

           WORKDIR /app
           COPY --chown=app:app pyproject.toml .
           RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
             && python -m pip install -U --no-cache-dir .

           ENTRYPOINT ["flower-supernode", "--node-config", "partition-id=0,num-partitions=2"]

If you also want to enable TLS for the new SuperNodes, duplicate the SuperNode definition for
each new SuperNode service in the ``with-tls.yml`` file.

Make sure that the names of the services match with the one in the ``compose.yml`` file.

In ``with-tls.yml``, add the following:

.. code-block:: yaml
   :caption: with-tls.yml

   services:
     # other service definitions

     supernode-3:
       command:
         - --superlink
         - superlink:9092
         - --root-certificates
         - certificates/ca.crt
       secrets:
         - source: superlink-ca-certfile
           target: /app/certificates/ca.crt

Step 8: Persisting the SuperLink State and Enabling TLS
-------------------------------------------------------

To run Flower with persisted SuperLink state and enabled TLS, a slight change in the ``with-state.yml``
file is required:

#. Comment out the lines 3-5 and uncomment the lines 6-10:

   .. code-block:: yaml
      :caption: with-state.yml
      :linenos:
      :emphasize-lines: 3-10

      services:
        superlink:
          # command:
          #   - --insecure
          #   - --database=state/state.db
          command:
            - --ssl-ca-certfile=certificates/ca.crt
            - --ssl-certfile=certificates/server.pem
            - --ssl-keyfile=certificates/server.key
            - --database=state/state.db
          volumes:
            - ./state/:/app/state/:rw

#. Restart the services:

   .. code-block:: bash

      $ docker compose -f compose.yml -f with-tls.yml -f with-state.yml up --build -d

#. Rerun the ``quickstart-compose`` project:

   .. code-block:: bash

      $ flwr run quickstart-compose docker-compose-tls
      $ docker compose logs superexec -f

Step 9: Merge Multiple Compose Files
------------------------------------

You can merge multiple Compose files into a single file. For instance, if you wish to combine
the basic configuration with the TLS configuration, execute the following command:

.. code-block:: bash

   $ docker compose -f compose.yml \
      -f with-tls.yml config --no-path-resolution > my_compose.yml

This will merge the contents of ``compose.yml`` and ``with-tls.yml`` into a new file called
``my_compose.yml``.

Step 10: Clean Up
-----------------

Remove all services and volumes:

.. code-block:: bash

   $ docker compose down -v
   $ docker compose -f certs.yml down -v
