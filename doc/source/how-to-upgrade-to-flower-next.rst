Upgrade to Flower Next
======================

Welcome to the migration guide for updating Flower to Flower Next! Whether you're a seasoned user
or just getting started, this guide will help you smoothly transition your existing setup to take
advantage of the latest features and improvements in Flower Next, starting from version 1.8. Let's dive in!

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
Therefore, the main changes are in the execution of federated learning, clients, and
server. These are the following non-breaking changes that require manual updates.

Client
~~~~~~
- Use ``flwr.client.ClientApp(client_fn)`` instead of ``flwr.client.start_client(server_address, client)``.
  Here's an example:

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

Server
~~~~~~
- Use ``flwr.server.ServerApp(config, strategy)`` instead of ``flwr.server.start_server(server_address, config, strategy)``.
  Here's an example:

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

Simulation
~~~~~~~~~~
- Use ``flwr.client.ClientApp()`` and ``flwr.server.ServerApp()`` instead of ``flwr.simulation.start_simulation()``.
  Here's an example:

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

- Run ``flower-simulation`` in CLI and point to the ``server``/``client`` object in the code instead of
  executing the Python script. Here's an example (assuming the ``server`` and ``client`` are in a ``sim.py`` file):

.. code-block:: bash

    # Flower 1.8
    $ flower-simulation --client-app=sim:client --server-app=sim:server --num-supernodes=100

    # Flower 1.7
    $ python sim.py

- Change default resources for each ``ClientApp`` using the ``--backend-config`` argument instead of custom arguments
  parsed by ``argparse``. Here's an example:

.. code-block:: bash

    # Flower 1.8
    $ flower-simulation --client-app=sim:client --server-app=sim:server --num-supernodes=100 \
        --backend-config='{"client_resources": {"num_cpus":2, "num_gpus":0.25}}'

    # Flower 1.7
    $ python sim.py --num_cpus=2 --num_gpus=0.25


Deployment
~~~~~~~~~~
Run the ``SuperLink`` before running ``ServerApp`` and ``SuperNode`` instead of executing `client.py` and
`server.py` as Python scripts. Here's an example:

.. code-block:: bash
    
    # Start a Superlink
    $ flower-superlink --insecure

    # In a new terminal window, start a long-running SuperNode
    $ flower-client-app client:app --insecure

    # In another terminal window, start a long-running SuperNode (at least 2 SuperNodes are required)
    $ flower-client-app client:app --insecure

    # In another terminal window, run the apps
    $ flower-server-app server:app --insecure



Further help
------------

Some official `Flower code examples <https://github.com/adap/flower/tree/main/examples>`_ are already
updated to Flower Next so they can serve as a reference for using the Flower Next API. If there are
further questions, `join the Flower Slack <https://flower.ai/join-slack/>`_ and use the channel ``#questions``.
You can also `participate in Flower Discuss <https://discuss.flower.ai/>`_ where you can find us
answering questions, or share and learn from others about migrating to Flower Next.

Happy migrating!