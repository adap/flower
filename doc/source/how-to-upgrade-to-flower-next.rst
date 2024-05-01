Upgrade to Flower Next
======================

Welcome to the migration guide for updating Flower to Flower Next! Whether you're a seasoned user
or just getting started, this guide will help you smoothly transition your existing setup to take
advantage of the latest features and improvements in Flower Next, starting from version 1.8.

.. note::
    This guide shows how to reuse pre-``1.8`` Flower code with minimum code changes by
    using the *compatibility layer* in Flower Next. In another guide, we will show how
    to run Flower Next end-to-end with pure Flower Next APIs.

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
.. |flowernext_superlink_link| replace:: ``flower-superlink``
.. |flowernext_clientapp_link| replace:: ``flower-client-app``
.. |flowernext_serverapp_link| replace:: ``flower-server-app``
.. _clientapp_link: ref-api/flwr.client.ClientApp.html
.. _serverapp_link: ref-api/flwr.server.ServerApp.html
.. _startclient_link: ref-api/flwr.client.start_client.html
.. _startserver_link: ref-api/flwr.server.start_server.html
.. _startsim_link: ref-api/flwr.simulation.start_simulation.html
.. _runsim_link: ref-api/flwr.simulation.run_simulation.html
.. _flowernext_superlink_link: ref-api-cli.html#flower-superlink
.. _flowernext_clientapp_link: ref-api-cli.html#flower-client-app
.. _flowernext_serverapp_link: ref-api-cli.html#flower-server-app

Install update
--------------

Using pip
~~~~~~~~~

Here's how to update an existing installation of Flower to Flower Next with ``pip``:

.. code-block:: bash
    
    $ python -m pip install -U flwr

or if you need Flower Next with simulation:

.. code-block:: bash
    
    $ python -m pip install -U flwr[simulation]


Ensure you set the following version constraint in your ``requirements.txt``

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

Using Poetry
~~~~~~~~~~~~

Update the ``flwr`` dependency in ``pyproject.toml`` and then reinstall (don't forget to delete ``poetry.lock`` via ``rm poetry.lock`` before running ``poetry install``).

Ensure you set the following version constraint in your ``pyproject.toml``:

.. code-block:: toml

    [tool.poetry.dependencies]
    python = "^3.8"

    # Without simulation support
    flwr = ">=1.8,<2.0"

    # With simulation support
    flwr = { version = ">=1.8,<2.0", extras = ["simulation"] }

Required changes
----------------

In Flower Next, the *infrastructure* and *application layers* have been decoupled.
Instead of starting a client in code via ``start_client()``, you create a |clientapp_link|_ and start it via the command line.
Instead of starting a server in code via ``start_server()``, you create a |serverapp_link|_ and start it via the command line.
The long-running components of server and client are called SuperLink and SuperNode.
The following non-breaking changes that require manual updates and allow you to run your project both in the traditional way and in the Flower Next way:

|clientapp_link|_
~~~~~~~~~~~~~~~~~
- Wrap your existing client with |clientapp_link|_ instead of launching it via
  |startclient_link|_. Here's an example:

.. code-block:: python
    :emphasize-lines: 5,11

    # Flower 1.8
    def client_fn(cid: str):
        return flwr.client.FlowerClient().to_client() 
    
    app = flwr.client.ClientApp(
       client_fn=client_fn,
    )

    # Flower 1.7
    if __name__ == "__main__":
        flwr.client.start_client(
           server_address="127.0.0.1:8080",
           client=flwr.client.FlowerClient().to_client(),
        )

|serverapp_link|_
~~~~~~~~~~~~~~~~~
- Wrap your existing strategy with |serverapp_link|_ instead of starting the server
  via |startserver_link|_. Here's an example:

.. code-block:: python
    :emphasize-lines: 2,9

    # Flower 1.8
    app = flwr.server.ServerApp(
        config=config,
        strategy=strategy,
    )

    # Flower 1.7
    if __name__ == "__main__":
        flwr.server.start_server(
            server_address="0.0.0.0:8080",
            config=config,
            strategy=strategy,
        )

Deployment
~~~~~~~~~~
- Run the ``SuperLink`` using |flowernext_superlink_link|_ before running, in sequence,
  |flowernext_clientapp_link|_ (2x) and |flowernext_serverapp_link|_. There is no need to
  execute `client.py` and `server.py` as Python scripts.
- Here's an example to start the server without HTTPS (only for prototyping):

.. code-block:: bash
    
    # Start a Superlink
    $ flower-superlink --insecure

    # In a new terminal window, start a long-running SuperNode
    $ flower-client-app client:app --insecure

    # In another terminal window, start another long-running SuperNode (at least 2 SuperNodes are required)
    $ flower-client-app client:app --insecure

    # In yet another terminal window, run the ServerApp (this starts the actual training run)
    $ flower-server-app server:app --insecure

- Here's another example to start with HTTPS. Use the ``--certificates`` command line
  argument to pass paths to (CA certificate, server certificate, and server private key).

.. code-block:: bash

    # Start a secure Superlink
    $ flower-superlink --certificates \
        <your-ca-cert-filepath> \
        <your-server-cert-filepath> \
        <your-privatekey-filepath>

    # In a new terminal window, start a long-running secure SuperNode
    $ flower-client-app client:app \
        --root-certificates <your-ca-cert-filepath> \
        --server 127.0.0.1:9092

    # In another terminal window, start another long-running secure SuperNode (at least 2 SuperNodes are required)
    $ flower-client-app client:app \
        --root-certificates <your-ca-cert-filepath> \
        --server 127.0.0.1:9092

    # In yet another terminal window, run the ServerApp (this starts the actual training run)
    $ flower-server-app server:app \
        --root-certificates <your-ca-cert-filepath> \
        --server 127.0.0.1:9091

Simulation in CLI
~~~~~~~~~~~~~~~~~
- Wrap your existing client and strategy with |clientapp_link|_ and |serverapp_link|_,
  respectively. There is no need to use |startsim_link|_ anymore. Here's an example:

.. code-block:: python
    :emphasize-lines: 9,13,20

    # Regular Flower client implementation
    class FlowerClient(NumPyClient):
        # ...

    # Flower 1.8
    def client_fn(cid: str):
        return FlowerClient().to_client() 
    
    client_app = flwr.client.ClientApp(
       client_fn=client_fn,
    )

    server_app = flwr.server.ServerApp(
        config=config,
        strategy=strategy,
    )

    # Flower 1.7
    if __name__ == "__main__":
        hist = flwr.simulation.start_simulation(
            num_clients=100,
            ...
        )

- Run :code:`flower-simulation` in CLI and point to the ``server_app`` / ``client_app`` object in the
  code instead of executing the Python script. Here's an example (assuming the
  ``server_app`` and ``client_app`` objects are in a ``sim.py`` module):

.. code-block:: bash

    # Flower 1.8
    $ flower-simulation \
        --server-app=sim:server_app \
        --client-app=sim:client_app \
        --num-supernodes=100

.. code-block:: bash

    # Flower 1.7
    $ python sim.py

- Set default resources for each |clientapp_link|_ using the ``--backend-config`` command
  line argument instead of setting the ``client_resources`` argument in
  |startsim_link|_. Here's an example:

.. code-block:: bash
    :emphasize-lines: 6

    # Flower 1.8
    $ flower-simulation \
        --client-app=sim:client_app \
        --server-app=sim:server_app \
        --num-supernodes=100 \
        --backend-config='{"client_resources": {"num_cpus": 2, "num_gpus": 0.25}}'

.. code-block:: python
    :emphasize-lines: 5

    # Flower 1.7 (in `sim.py`)
    if __name__ == "__main__":
        hist = flwr.simulation.start_simulation(
            num_clients=100,
            client_resources = {'num_cpus': 2, "num_gpus": 0.25},
            ...
        )

Simulation in a Notebook
~~~~~~~~~~~~~~~~~~~~~~~~
- Run |runsim_link|_ in your notebook instead of |startsim_link|_. Here's an example:

.. code-block:: python
    :emphasize-lines: 19,27

    NUM_CLIENTS = <specify-an-integer>

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

    backend_config = {"client_resources": {"num_cpus": 2, "num_gpus": 0.25}}

    # Flower 1.8
    flwr.simulation.run_simulation(
        server_app=server_app, 
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )

    # Flower 1.7
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
updated to Flower Next so they can serve as a reference for using the Flower Next API. If there are
further questions, `join the Flower Slack <https://flower.ai/join-slack/>`_ and use the channel ``#questions``.
You can also `participate in Flower Discuss <https://discuss.flower.ai/>`_ where you can find us
answering questions, or share and learn from others about migrating to Flower Next.

.. admonition:: Important
    :class: important

    As we continuously enhance Flower Next at a rapid pace, we'll be periodically
    updating this guide. Please feel free to share any feedback with us!

..
    [TODO] Add links to Flower Next 101 and Flower Glossary

Happy migrating! 🚀
