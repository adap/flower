Configure clients
=================

Along with model parameters, Flower can send configuration values to clients. Configuration values can be used for various purposes. They are, for example, a popular way to control client-side hyperparameters from the server.

Configuration values
--------------------

Configuration values are represented as a dictionary with ``str`` keys and values of type ``bool``, ``bytes``, ``double`` (64-bit precision float), ``int``, or ``str`` (or equivalent types in different languages). Here is an example of a configuration dictionary in Python:

.. code-block:: python

    config_dict = {
        "dropout": True,        # str key, bool value
        "learning_rate": 0.01,  # str key, float value
        "batch_size": 32,       # str key, int value
        "optimizer": "sgd",     # str key, str value
    }

Flower serializes these configuration dictionaries (or *config dict* for short) to their ProtoBuf representation, transports them to the client using gRPC, and then deserializes them back to Python dictionaries.

.. note::

  Currently, there is no support for directly sending collection types (e.g., ``Set``, ``List``, ``Map``) as values in configuration dictionaries. There are several workarounds to send collections as values by converting them to one of the supported value types (and converting them back on the client-side).

  One can, for example, convert a list of floating-point numbers to a JSON string, then send the JSON string using the configuration dictionary, and then convert the JSON string back to a list of floating-point numbers on the client.


Configuration through built-in strategies
-----------------------------------------

The easiest way to send configuration values to clients is to use a built-in strategy like :code:`FedAvg`. Built-in strategies support so-called configuration functions. A configuration function is a function that the built-in strategy calls to get the configuration dictionary for the current round. It then forwards the configuration dictionary to all the clients selected during that round.

Let's start with a simple example. Imagine we want to send (a) the batch size that the client should use, (b) the current global round of federated learning, and (c) the number of epochs to train on the client-side. Our configuration function could look like this:

.. code-block:: python

    def fit_config(server_round: int):
        """Return training configuration dict for each round."""
        config = {
            "batch_size": 32,
            "current_round": server_round,
            "local_epochs": 2,
        }
        return config

To make the built-in strategies use this function, we can pass it to ``FedAvg`` during initialization using the parameter :code:`on_fit_config_fn`:

.. code-block:: python

    strategy = FedAvg(
        ...,                          # Other FedAvg parameters
        on_fit_config_fn=fit_config,  # The fit_config function we defined earlier
    )

One the client side, we receive the configuration dictionary in ``fit``:

.. code-block:: python

    class FlowerClient(flwr.client.NumPyClient):
        def fit(parameters, config):
            print(config["batch_size"])  # Prints `32`
            print(config["current_round"])  # Prints `1`/`2`/`...`
            print(config["local_epochs"])  # Prints `2`
            # ... (rest of `fit` method)

There is also an `on_evaluate_config_fn` to configure evaluation, which works the same way. They are separate functions because one might want to send different configuration values to `evaluate` (for example, to use a different batch size).

The built-in strategies call this function every round (that is, every time `Strategy.configure_fit` or `Strategy.configure_evaluate` runs). Calling `on_evaluate_config_fn` every round allows us to vary/change the config dict over consecutive rounds. If we wanted to implement a hyperparameter schedule, for example, to increase the number of local epochs during later rounds, we could do the following:

.. code-block:: python

    def fit_config(server_round: int):
        """Return training configuration dict for each round."""
        config = {
            "batch_size": 32,
            "current_round": server_round,
            "local_epochs": 1 if server_round < 2 else 2,
        }
        return config

The :code:`FedAvg` strategy will call this function *every round*.

Configuring individual clients
------------------------------

In some cases, it is necessary to send different configuration values to different clients.

This can be achieved by customizing an existing strategy or by :doc:`implementing a custom strategy from scratch <how-to-implement-strategies>`. Here's a nonsensical example that customizes :code:`FedAvg` by adding a custom ``"hello": "world"`` configuration key/value pair to the config dict of a *single client* (only the first client in the list, the other clients in this round to not receive this "special" config value):

.. code-block:: python

    class CustomClientConfigStrategy(fl.server.strategy.FedAvg):
        def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, FitIns]]:
            client_instructions = super().configure_fit(server_round, parameters, client_manager)

            # Add special "hello": "world" config key/value pair,
            # but only to the first client in the list
            _, fit_ins = client_instructions[0]  # First (ClientProxy, FitIns) pair
            fit_ins.config["hello"] = "world"  # Change config for this client only

            return client_instructions

    # Create strategy and run server
    strategy = CustomClientConfigStrategy(
        # ... (same arguments as plain FedAvg here)
    )
    fl.server.start_server(strategy=strategy)
