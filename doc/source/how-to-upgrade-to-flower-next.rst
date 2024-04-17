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
.. |runsim_link| replace:: ``flower-simulation``
.. |flowernext_superlink_link| replace:: ``flower-superlink``
.. |flowernext_clientapp_link| replace:: ``flower-client-app``
.. |flowernext_serverapp_link| replace:: ``flower-server-app``
.. _clientapp_link: ref-api/flwr.client.ClientApp.html
.. _serverapp_link: ref-api/flwr.server.ServerApp.html
.. _startclient_link: ref-api/flwr.client.start_client.html
.. _startserver_link: ref-api/flwr.server.start_server.html
.. _startsim_link: ref-api/flwr.simulation.start_simulation.html
.. _runsim_link: ref-api/flwr.simulation.run_simulation_from_cli.html
.. _flowernext_superlink_link: ref-api-cli.html#flower-superlink
.. _flowernext_clientapp_link: ref-api-cli.html#flower-client-app
.. _flowernext_serverapp_link: ref-api-cli.html#flower-server-app

Install update
--------------

Here's how to update an existing installation of Flower to Flower Next with ``pip``:

.. code-block:: bash
    
    $ python -m pip install -U flwr

or if you need Flower Next with simulation:

.. code-block:: bash
    
    $ python -m pip install -U flwr[simulation]

Dependencies
------------
Ensure you set the following version constraint in your ``requirements.txt``

.. code-block:: 

    flwr>=1.8.0,<2.0

or ``pyproject.toml``:

.. code-block:: toml

    dependencies = ["flwr>=1.8.0,2.0"]


Required changes
----------------

In Flower Next, the *infrastructure* and *application layers* have been decoupled.
Therefore, the main changes are in the setup of the ``SuperNode``, |clientapp_link|_,
|serverapp_link|_ and execution of federated learning. These are the following non-breaking
changes that require manual updates.

``SuperNode``/|clientapp_link|_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Wrap your existing client with |clientapp_link|_ instead of launching it via
  |startclient_link|_. Here's an example:

.. code-block:: python

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
- Run the ``SuperLink`` with |flowernext_superlink_link|_ before running, in sequence,
  |flowernext_clientapp_link|_ and |flowernext_serverapp_link|_. There is no need to
  execute `client.py` and `server.py` as Python scripts.
- Here's an example to start the server without HTTPS:

.. code-block:: bash
    
    # Start a Superlink
    $ flower-superlink --insecure

    # In a new terminal window, start a long-running SuperNode
    $ flower-client-app client:app --insecure

    # In another terminal window, start a long-running SuperNode (at least 2 SuperNodes are required)
    $ flower-client-app client:app --insecure

    # In another terminal window, run the apps
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

    # In another terminal window, start a long-running secure SuperNode (at least 2 SuperNodes are required)
    $ flower-client-app client:app \
        --root-certificates <your-ca-cert-filepath> \
        --server 127.0.0.1:9092

    # In another terminal window, run the apps
    $ flower-server-app server:app \
        --root-certificates <your-ca-cert-filepath> \
        --server 127.0.0.1:9091

Simulation
~~~~~~~~~~
- Wrap your existing client and strategy with |clientapp_link|_ and |serverapp_link|_,
  respectively. There is no need to use |startsim_link|_ anymore. Here's an example:

.. code-block:: python

    # Flower 1.8
    def client_fn(cid: str):
        return flwr.client.FlowerClient().to_client() 
    
    client = flwr.client.ClientApp(
       client_fn=client_fn,
    )

    server = flwr.server.ServerApp(
        config=config,
        strategy=strategy,
    )

    # Flower 1.7
    if __name__ == "__main__":
        flwr.simulation.start_simulation(
            ...
        )

- Run |runsim_link|_ in CLI and point to the ``server``/``client`` object in the
  code instead of executing the Python script. Here's an example (assuming the
  ``server`` and ``client`` are in a ``sim.py`` file):

.. code-block:: bash

    # Flower 1.8
    $ flower-simulation --client-app=sim:client --server-app=sim:server --num-supernodes=100

    # Flower 1.7
    $ python <your_script>.py

- Set default resources for each |clientapp_link|_ using the ``--backend-config`` command
  line argument instead of setting the ``client_resources`` argument in
  |startsim_link|_. Here's an example:

.. code-block:: bash

    # Flower 1.8
    $ flower-simulation --client-app=sim:client --server-app=sim:server --num-supernodes=100 \
        --backend-config='{"client_resources": {"num_cpus":2, "num_gpus":0.25}}'

.. code-block:: python

    # Flower 1.7 (in <your_script>.py)
    hist = flwr.simulation.start_simulation(
        ...
        client_resources = {'num_cpus': 2, "num_gpus": 0.25}
    )

Further help
------------

Some official `Flower code examples <https://github.com/adap/flower/tree/main/examples>`_ are already
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