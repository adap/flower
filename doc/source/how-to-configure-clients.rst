Configure Clients
=================

Flower provides the ability to send configuration values to clients, allowing
server-side control over client behavior. This feature enables flexible and
dynamic adjustment of client-side hyperparameters, improving collaboration and
experimentation.

Configuration values
--------------------

Configuration values are represented as a dictionary with ``str`` keys and
values of type ``bool``, ``bytes``, ``float``, ``int``, or ``str`` (or
equivalent types in different languages). Here is an example of a configuration
dictionary in Python:

.. code-block:: python

    config_dict = {
        "dropout": True,  # str key, bool value
        "learning_rate": 0.01,  # str key, float value
        "batch_size": 32,  # str key, int value
        "optimizer": "sgd",  # str key, str value
    }

Flower serializes these configuration dictionaries (or *config dict* for short)
to their ProtoBuf representation, transports them to the client using gRPC, and
then deserializes them back to Python dictionaries.

.. note::

    Currently, there is no support for directly sending collection types (e.g.,
    ``Set``, ``List``, ``Map``) as values in configuration dictionaries. To
    send collections, convert them to a supported type (e.g., JSON string) and
    decode on the client side.

    Example:

    .. code-block:: python

        import json

        # On the server
        config_dict = {"data_splits": json.dumps([0.8, 0.1, 0.1])}

        # On the client
        data_splits = json.loads(config["data_splits"])

Using Built-in Strategies for Configuration
-------------------------------------------

Flower supports configuration functions to dynamically adjust parameters sent
to clients. Built-in strategies like ``FedAvg`` allow for setting configuration
values for each round via a function.

Example: Sending Training Configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Imagine we want to send (a) the batch size, (b) the current global round, and
(c) the number of local epochs. Our configuration function could look like
this:

.. code-block:: python

    def fit_config(server_round: int):
        """Generate training configuration for each round."""
        return {
            "batch_size": 32,
            "current_round": server_round,
            "local_epochs": 2,
        }

To use this function with a built-in strategy like ``FedAvg``, pass it during
initialization:

.. code-block:: python

    strategy = FedAvg(
        on_fit_config_fn=fit_config  # Assign the configuration function
    )

With the latest version of Flower, you no longer use `fl.server.start_server`.
Instead, the server is defined as a `ServerApp`:

.. code-block:: python

    from flwr.server import ServerApp, ServerAppComponents
    from flwr.server.strategy import FedAvg

    def server_fn(context):
        """Define server behavior."""
        strategy = FedAvg(
            on_fit_config_fn=fit_config,
            # Additional parameters...
        )
        return ServerAppComponents(strategy=strategy)

    app = ServerApp(server_fn=server_fn)

Client-Side Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

On the client side, configurations are received as input to the `fit` and
`evaluate` methods. For example:

.. code-block:: python

    class FlowerClient(flwr.client.NumPyClient):
        def fit(self, parameters, config):
            print(config["batch_size"])  # Output: 32
            print(config["current_round"])  # Output: current round number
            print(config["local_epochs"])  # Output: 2
            # Training logic here

        def evaluate(self, parameters, config):
            # Handle evaluation configurations if needed
            pass

Dynamic Configurations per Round
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configuration functions are called at the beginning of every round. This
allows for dynamic adjustments based on progress. For example, increasing
the number of local epochs in later rounds:

.. code-block:: python

    def fit_config(server_round: int):
        """Dynamic configuration for training."""
        return {
            "batch_size": 32,
            "current_round": server_round,
            "local_epochs": 1 if server_round < 3 else 2,
        }

Customizing Client Configurations
---------------------------------

In some cases, it may be necessary to send different configurations to
individual clients. To achieve this, you can create a custom strategy by
extending a built-in one, such as ``FedAvg``:

Example: Client-Specific Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from flwr.server.strategy import FedAvg

    class CustomClientConfigStrategy(FedAvg):
        def configure_fit(self, server_round, parameters, client_manager):
            client_instructions = super().configure_fit(
                server_round, parameters, client_manager
            )

            # Modify configuration for a specific client
            client_proxy, fit_ins = client_instructions[0]
            fit_ins.config["special_key"] = "special_value"

            return client_instructions

To use this custom strategy:

.. code-block:: python

    def server_fn(context):
        strategy = CustomClientConfigStrategy(
            # Other FedAvg parameters
        )
        return ServerAppComponents(strategy=strategy)

    app = ServerApp(server_fn=server_fn)

Summary of Enhancements
-----------------------

- **ServerApp Usage**: Allows modular configuration using `server_fn`.
- **Dynamic Configurations**: Enables per-round adjustments via functions.
- **Advanced Customization**: Supports client-specific strategies.
- **Client-Side Integration**: Configurations accessible in `fit` and
  `evaluate`.