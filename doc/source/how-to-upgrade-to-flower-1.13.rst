Upgrade to Flower 1.13
======================

Welcome to the migration guide for updating Flower to Flower 1.13! Whether you're a
seasoned user or just getting started, this guide will help you smoothly transition your
existing setup to take advantage of the latest features and improvements in Flower 1.13,
starting from version 1.8.

.. note::

    This guide shows how to reuse pre-``1.8`` Flower code with minimum code changes by
    using the *compatibility layer* in Flower 1.13. In another guide, we will show how
    to run Flower 1.13 end-to-end with pure Flower 1.13 APIs.

Let's dive in!

..
    Generate link text as literal. Refs:
    - https://stackoverflow.com/q/71651598
    - https://github.com/jgm/pandoc/issues/3973#issuecomment-337087394

.. |clientapp_link| replace:: ``ClientApp()``

.. |serverapp_link| replace:: ``ServerApp()``

.. |startclient_link| replace:: ``start_client()``

.. |startserver_link| replace:: ``start_server()``

.. |startsim_link| replace:: ``start_simulation()``

.. |runsim_link| replace:: ``run_simulation()``

.. |flowernext_clientapp_link| replace:: ``flower-client-app``

.. |flowernext_serverapp_link| replace:: ``flower-server-app``

.. |flowernext_superlink_link| replace:: ``flower-superlink``

.. |flowernext_supernode_link| replace:: ``flower-supernode``

.. |flower_architecture_link| replace:: Flower Architecture

.. |flower_how_to_run_simulations_link| replace:: How-to Run Simulations

.. |flower_simulation_link| replace:: ``flower-simulation``

.. _clientapp_link: ref-api/flwr.client.ClientApp.html

.. _flower_architecture_link: explanation-flower-architecture.html

.. _flower_how_to_run_simulations_link: how-to-run-simulations.html

.. _flower_simulation_link: ref-api-cli.html#flower-simulation

.. _flowernext_clientapp_link: ref-api-cli.html#flower-client-app

.. _flowernext_serverapp_link: ref-api-cli.html#flower-server-app

.. _flowernext_superlink_link: ref-api-cli.html#flower-superlink

.. _flowernext_supernode_link: ref-api-cli.html#flower-supernode

.. _runsim_link: ref-api/flwr.simulation.run_simulation.html

.. _serverapp_link: ref-api/flwr.server.ServerApp.html

.. _startclient_link: ref-api/flwr.client.start_client.html

.. _startserver_link: ref-api/flwr.server.start_server.html

.. _startsim_link: ref-api/flwr.simulation.start_simulation.html

Install update
--------------

Here's how to update an existing installation of Flower to Flower 1.13 with ``pip``:

.. code-block:: bash

    $ python -m pip install -U flwr

or if you need Flower 1.13 with simulation:

.. code-block:: bash

    $ python -m pip install -U "flwr[simulation]"

.. code-block::

    # Without simulation support
    flwr>=1.8,<2.0

    # With simulation support
    flwr[simulation]>=1.8, <2.0

or ``pyproject.toml``:

.. code-block:: toml

    # Without simulation support
    dependencies = ["flwr>=1.8,2.0"]

    # With simulation support
    dependencies = ["flwr[simulation]>=1.8,2.0"]

Required changes
----------------

In Flower 1.13, the *infrastructure* and *application layers* have been decoupled.
Instead of starting a client in code via ``start_client()``, you create a
|clientapp_link|_ and start it via the command line. Instead of starting a server in
code via ``start_server()``, you create a |serverapp_link|_ and start it via the command
line. The long-running components of server and client are called `SuperLink` and
`SuperNode`, for more details please see the |flower_architecture_link|_ . The following
non-breaking changes that require manual updates and allow you to run your project both
in the traditional way and in the Flower 1.13 way:

|clientapp_link|_
~~~~~~~~~~~~~~~~~

- Wrap your existing client with |clientapp_link|_ instead of launching it via
  |startclient_link|_. Here's an example:

.. code-block:: python
    :emphasize-lines: 2,6

    # Flower v1.11+
    def client_fn(context: flwr.common.Context):
        return flwr.client.FlowerClient().to_client()


    app = flwr.client.ClientApp(
        client_fn=client_fn,
    )


    # Flower v1.8 - v1.10
    def client_fn(cid: str):
        return flwr.client.FlowerClient().to_client()


    app = flwr.client.ClientApp(
        client_fn=client_fn,
    )

    # Flower v1.7
    if __name__ == "__main__":
        flwr.client.start_client(
            server_address="127.0.0.1:8080",
            client=flwr.client.FlowerClient().to_client(),
        )

|serverapp_link|_
~~~~~~~~~~~~~~~~~

- Wrap your existing strategy with |serverapp_link|_ instead of starting the server via
  |startserver_link|_. Here's an example:

.. code-block:: python
    :emphasize-lines: 2,8

    # Flower v1.11+
    def server_fn(context: flwr.common.Context):
        strategy = flwr.server.strategy.FedAvg()
        config = flwr.server.ServerConfig()
        return flwr.server.ServerAppComponents(strategy=strategy, config=config)


    app = flwr.server.ServerApp(
        server_fn=server_fn,
    )

    # Flower v1.8 - v1.11
    app = flwr.server.ServerApp(
        config=config,
        strategy=strategy,
    )

    # Flower v1.7
    if __name__ == "__main__":
        flwr.server.start_server(
            server_address="0.0.0.0:8080",
            config=config,
            strategy=strategy,
        )

Deployment
~~~~~~~~~~

- Run the ``SuperLink`` using |flowernext_superlink_link|_ before running, in sequence,
  |flowernext_supernode_link|_ (2x).
- Here's an example to start the server without HTTPS (only for prototyping):

.. code-block:: bash
    :emphasize-lines: 2,5,12

    # Start a Superlink
    $ flower-superlink --insecure

    # In a new terminal window, start a long-running SuperNode
    $ flower-supernode \
         --insecure \
         --superlink 127.0.0.1:9092 \
         --node-config "..." \
         --supernode-address 127.0.0.1:9094

    # In another terminal window, start another long-running SuperNode (at least 2 SuperNodes are required)
    $ flower-supernode \
         --insecure \
         --superlink 127.0.0.1:9092 \
         --node-config "..." \
         --supernode-address 127.0.0.1:9095

- Here's another example to start with HTTPS. Use the ``--ssl-ca-certfile``,
  ``--ssl-certfile``, and ``--ssl-keyfile`` command line options to pass paths to (CA
  certificate, server certificate, and server private key).

.. code-block:: bash
    :emphasize-lines: 2,8,15

    # Start a secure Superlink
    $ flower-superlink \
        --ssl-ca-certfile <your-ca-cert-filepath> \
        --ssl-certfile <your-server-cert-filepath> \
        --ssl-keyfile <your-privatekey-filepath>

    # In a new terminal window, start a long-running SuperNode
    $ flower-supernode \
         --superlink 127.0.0.1:9092 \
         --node-config "..." \
         --supernode-address 127.0.0.1:9094 \
         --root-certificates <your-ca-cert-filepath>

    # In another terminal window, start another long-running SuperNode (at least 2 SuperNodes are required)
    $ flower-supernode \
         --superlink 127.0.0.1:9092 \
         --node-config "..." \
         --supernode-address 127.0.0.1:9095 \
         --root-certificates <your-ca-cert-filepath>

Simulation in CLI
~~~~~~~~~~~~~~~~~

Wrap your existing client and strategy with |clientapp_link|_ and |serverapp_link|_,
respectively. There is no need to use |startsim_link|_ anymore. Here's an example:

.. tip::

    For more advanced information regarding Flower simulation please read the
    |flower_how_to_run_simulations_link|_ guide.

.. code-block:: python
    :emphasize-lines: 9,13,18,25

    # Regular Flower client implementation
    class FlowerClient(NumPyClient):
        # ...
        pass


    # Flower v1.11+
    # [file: client_app.py]
    def client_fn(context: flwr.common.Context):
        return flwr.client.FlowerClient().to_client()


    app = flwr.client.ClientApp(
        client_fn=client_fn,
    )


    # [file: server_app.py]
    def server_fn(context: flwr.common.Context):
        strategy = flwr.server.strategy.FedAvg(...)
        config = flwr.server.ServerConfig(...)
        return flwr.server.ServerAppComponents(strategy=strategy, config=config)


    server_app = flwr.server.ServerApp(
        server_fn=server_fn,
    )


    # Flower v1.8 - v1.10
    def client_fn(cid: str):
        return FlowerClient().to_client()


    client_app = flwr.client.ClientApp(
        client_fn=client_fn,
    )

    server_app = flwr.server.ServerApp(
        config=config,
        strategy=strategy,
    )

    # Flower v1.7
    if __name__ == "__main__":
        hist = flwr.simulation.start_simulation(
            num_clients=10,
            # ...
        )

Depending on your Flower version, you can run your simulation as follows:

- for Flower versions 1.11 and onwards, run ``flwr run`` in CLI.
- for Flower versions between 1.8 to 1.10, run |flower_simulation_link|_ in CLI and
  point to the ``server_app`` / ``client_app`` object in the code instead of executing
  the Python script. In the code snippet below, there is an example (assuming the
  ``server_app`` and ``client_app`` objects are in a ``sim.py`` module).
- for Flower versions before 1.8, run the Python script directly.

.. code-block:: bash
    :emphasize-lines: 2

    # Flower v1.11+
    $ flwr run


    # Flower v1.8 - v1.10
    $ flower-simulation \
        --server-app=sim:server_app \
        --client-app=sim:client_app \
        --num-supernodes=10


    # Flower v1.7
    $ python sim.py

Depending on your Flower version, you can also define the default resources as follows:

- for Flower versions 1.11 and onwards, you can edit your pyproject.toml file and then
  run ``flwr run`` in CLI as shown in the example below.
- for Flower versions between 1.8 to 1.10, you can adjust the resources for each
  |clientapp_link|_ using the ``--backend-config`` command line argument instead of
  setting the ``client_resources`` argument in |startsim_link|_.
- for Flower versions before 1.8, you need to run |startsim_link|_ and pass a dictionary
  of the required resources to the ``client_resources`` argument.

.. code-block:: bash
    :emphasize-lines: 2,8

    # Flower v.1.11+
    # pyproject.toml
    [tool.flwr.federations.local-sim-gpu]
    options.num-supernodes = 10
    options.backend.client-resources.num-cpus = 2
    options.backend.client-resources.num-gpus = 0.25

    $ flwr run

    # Flower v1.8 - v1.10
    $ flower-simulation \
        --client-app=sim:client_app \
        --server-app=sim:server_app \
        --num-supernodes=10 \
        --backend-config='{"client_resources": {"num_cpus": 2, "num_gpus": 0.25}}'

    # Flower v1.7 (in `sim.py`)
    if __name__ == "__main__":
        hist = flwr.simulation.start_simulation(
            num_clients=10, client_resources={"num_cpus": 2, "num_gpus": 0.25}, ...
        )

Simulation in a Notebook
~~~~~~~~~~~~~~~~~~~~~~~~

To run your simulation from within a notebook, please consider the following examples
depending on your Flower version:

- for Flower versions 1.11 and onwards, you need to run |runsim_link|_ in your notebook
  instead of |startsim_link|_.
- for Flower versions between 1.8 to 1.10, you need to run |runsim_link|_ in your
  notebook instead of |startsim_link|_ and configure the resources.
- for Flower versions before 1.8, you need to run |startsim_link|_ and pass a dictionary
  of the required resources to the ``client_resources`` argument.

.. tip::

    For more advanced information regarding Flower simulation please read the
    |flower_how_to_run_simulations_link|_ guide.

.. code-block:: python
    :emphasize-lines: 2,6,10,14

    # Flower v1.11+
    def client_fn(context: flwr.common.Context):
        return flwr.client.FlowerClient().to_client()


    client_app = flwr.server.ClientApp(
        client_fn=client_fn,
    )

    server_app = flwr.server.ServerApp(
        server_fn=server_fn,
    )

    flwr.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
    )


    # Flower v1.8 - v1.10
    NUM_CLIENTS = 10  # Replace by any integer greater than zero
    backend_config = {"client_resources": {"num_cpus": 2, "num_gpus": 0.25}}


    def client_fn(cid: str):
        # ...
        return FlowerClient().to_client()


    client_app = flwr.client.ClientApp(
        client_fn=client_fn,
    )

    server_app = flwr.server.ServerApp(
        config=config,
        strategy=strategy,
    )

    flwr.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )

    # Flower v1.7
    NUM_CLIENTS = 10  # Replace by any integer greater than zero
    backend_config = {"client_resources": {"num_cpus": 2, "num_gpus": 0.25}}
    flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=config,
        strategy=strategy,
        client_resources=backend_config["client_resources"],
    )

Further help
------------

Some official `Flower code examples <https://flower.ai/docs/examples/>`_ are already
updated to Flower 1.13 so they can serve as a reference for using the Flower 1.13 API.
If there are further questions, `join the Flower Slack <https://flower.ai/join-slack/>`_
and use the channel ``#questions``. You can also `participate in Flower Discuss
<https://discuss.flower.ai/>`_ where you can find us answering questions, or share and
learn from others about migrating to Flower 1.13.

.. admonition:: Important

    As we continuously enhance Flower 1.13 at a rapid pace, we'll be periodically
    updating this guide. Please feel free to share any feedback with us!

..
    [TODO] Add links to Flower Next 101 and Flower Glossary

Happy migrating! 🚀
