:og:description: Configure Flower clients by sending parameters from the server to control client-side hyperparameters using configuration dictionaries and strategies.
.. meta::
    :description: Configure Flower clients by sending parameters from the server to control client-side hyperparameters using configuration dictionaries and strategies.

Configure Clients
=================

Flower provides the ability to send configuration values to clients, allowing
server-side control over client behavior. This feature enables flexible and dynamic
adjustment of client-side hyperparameters, improving collaboration and experimentation.

Configuration values
--------------------

``FitConfig`` and ``EvaluateConfig`` are dictionaries containing configuration values
that the server sends to clients during federated learning rounds. These values must be
of type ``Scalar``, which includes ``bool``, ``bytes``, ``float``, ``int``, or ``str``
(or equivalent types in different languages). Scalar is the value type directly
supported by Flower for these configurations.

For example, a ``FitConfig`` dictionary might look like this:

.. code-block:: python

    config = {
        "batch_size": 32,  # int value
        "learning_rate": 0.01,  # float value
        "optimizer": "sgd",  # str value
        "dropout": True,  # bool value
    }

Flower serializes these configuration dictionaries (or *config dicts* for short) to
their ProtoBuf representation, transports them to the client using gRPC, and then
deserializes them back to Python dictionaries.

.. note::

    Currently, there is no support for directly sending collection types (e.g., ``Set``,
    ``List``, ``Map``) as values in configuration dictionaries. To send collections,
    convert them to a supported type (e.g., JSON string) and decode on the client side.

    Example:

    .. code-block:: python

        import json

        # On the server
        config_dict = {"data_splits": json.dumps([0.8, 0.1, 0.1])}

        # On the client
        data_splits = json.loads(config["data_splits"])

Configuration through Built-in Strategies
-----------------------------------------

Flower provides configuration options to control client behavior dynamically through
``FitConfig`` and ``EvaluateConfig``. These configurations allow server-side control
over client-side parameters such as batch size, number of local epochs, learning rate,
and evaluation settings, improving collaboration and experimentation.

``FitConfig`` and ``EvaluateConfig``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``FitConfig`` and ``EvaluateConfig`` are dictionaries containing configuration values
that the server sends to clients during federated learning rounds. These dictionaries
enable the server to adjust client-side hyperparameters and monitor progress
effectively.

``FitConfig``
+++++++++++++

``FitConfig`` specifies the hyperparameters for training rounds, such as the batch size,
number of local epochs, and other parameters that influence training.

For example, a ``fit_config`` callback might look like this:

.. code-block:: python

    import json


    def fit_config(server_round: int):
        """Generate training configuration for each round."""
        # Create the configuration dictionary
        config = {
            "batch_size": 32,
            "current_round": server_round,
            "local_epochs": 2,
            "data_splits": json.dumps([0.8, 0.1, 0.1]),  # Example of serialized list
        }
        return config

You can then pass this ``fit_config`` callback to a built-in strategy such as
``FedAvg``:

.. code-block:: python

    from flwr.server.strategy import FedAvg

    strategy = FedAvg(
        on_fit_config_fn=fit_config,  # Pass the `fit_config` function
    )

On the client side, the configuration is received in the ``fit`` method, where it can be
read and used:

.. code-block:: python

    import json

    from flwr.client import NumPyClient


    class FlowerClient(NumPyClient):
        def fit(self, parameters, config):
            # Read configuration values
            batch_size = config["batch_size"]
            local_epochs = config["local_epochs"]
            data_splits = json.loads(config["data_splits"])  # Deserialize JSON

            # Use configuration values
            print(f"Training with batch size {batch_size}, epochs {local_epochs}")
            print(f"Data splits: {data_splits}")
            # Training logic here

``EvaluateConfig``
++++++++++++++++++

``EvaluateConfig`` specifies hyperparameters for the evaluation process, such as the
batch size, evaluation frequency, or metrics to compute during evaluation.

For example, an ``evaluate_config`` callback might look like this:

.. code-block:: python

    def evaluate_config(server_round: int):
        """Generate evaluation configuration for each round."""
        # Create the configuration dictionary
        config = {
            "batch_size": 64,
            "current_round": server_round,
            "metrics": ["accuracy"],  # Example metrics to compute
        }
        return config

You can pass this ``evaluate_config`` callback to a built-in strategy like ``FedAvg``:

.. code-block:: python

    strategy = FedAvg(
        on_evaluate_config_fn=evaluate_config  # Assign the evaluate_config function
    )

On the client side, the configuration is received in the ``evaluate`` method, where it
can be used during the evaluation process:

.. code-block:: python

    from flwr.client import NumPyClient


    class FlowerClient(NumPyClient):
        def evaluate(self, parameters, config):
            # Read configuration values
            batch_size = config["batch_size"]
            current_round = config["current_round"]
            metrics = config["metrics"]

            # Use configuration values
            print(f"Evaluating with batch size {batch_size}")
            print(f"Metrics to compute: {metrics}")

            # Evaluation logic here

            return 0.5, {"accuracy": 0.85}  # Example return values

Example: Sending Training Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Imagine we want to send (a) the batch size, (b) the current global round, and (c) the
number of local epochs. Our configuration function could look like this:

.. code-block:: python

    def fit_config(server_round: int):
        """Generate training configuration for each round."""
        return {
            "batch_size": 32,
            "current_round": server_round,
            "local_epochs": 2,
        }

To use this function with a built-in strategy like ``FedAvg``, pass it to the ``FedAvg``
constructor (typically in your ``server_fn``):

.. code-block:: python

    from flwr.server import ServerApp, ServerAppComponents
    from flwr.server.strategy import FedAvg


    def server_fn(context):
        """Define server behavior."""
        strategy = FedAvg(
            on_fit_config_fn=fit_config,
            # Other arguments...
        )
        return ServerAppComponents(strategy=strategy, ...)


    app = ServerApp(server_fn=server_fn)

Client-Side Configuration
+++++++++++++++++++++++++

On the client side, configurations are received as input to the ``fit`` and ``evaluate``
methods. For example:

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
++++++++++++++++++++++++++++++++

Configuration functions are called at the beginning of every round. This allows for
dynamic adjustments based on progress. For example, you can increase the number of local
epochs in later rounds:

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

In some cases, it may be necessary to send different configurations to individual
clients. To achieve this, you can create a custom strategy by extending a built-in one,
such as ``FedAvg``:

Example: Client-Specific Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Next, use this custom strategy as usual:

.. code-block:: python

    def server_fn(context):
        strategy = CustomClientConfigStrategy(
            # Other FedAvg parameters
        )
        return ServerAppComponents(strategy=strategy, ...)


    app = ServerApp(server_fn=server_fn)

Summary of Enhancements
-----------------------

- **Dynamic Configurations**: Enables per-round adjustments via functions.
- **Advanced Customization**: Supports client-specific strategies.
- **Client-Side Integration**: Configurations accessible in ``fit`` and ``evaluate``.
