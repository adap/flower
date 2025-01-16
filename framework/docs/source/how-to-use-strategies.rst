:og:description: Customize federated learning in Flower with built-in strategies, callbacks, and custom server-side implementations for maximum flexibility and control.
.. meta::
    :description: Customize federated learning in Flower with built-in strategies, callbacks, and custom server-side implementations for maximum flexibility and control.

Use strategies
==============

Flower allows full customization of the learning process through the ``Strategy``
abstraction. A number of built-in strategies are provided in the core framework.

There are three ways to customize the way Flower orchestrates the learning process on
the server side:

- Use an existing strategy, for example, ``FedAvg``
- Customize an existing strategy with callback functions
- Implement a novel strategy

Use an existing strategy
------------------------

Flower comes with a number of popular federated learning Strategies which can be
instantiated as follows:

.. code-block:: python

    from flwr.common import Context
    from flwr.server.strategy import FedAvg
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig


    def server_fn(context: Context):
        # Optional context-based parameters specification
        num_rounds = context.run_config["num-server-rounds"]
        config = ServerConfig(num_rounds=num_rounds)

        # Instantiate FedAvg strategy
        strategy = FedAvg(
            fraction_fit=context.run_config["fraction-fit"],
            fraction_evaluate=1.0,
        )

        # Create and return ServerAppComponents
        return ServerAppComponents(strategy=strategy, config=config)


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

To make the ``ServerApp`` use this strategy, pass a ``server_fn`` function to the
``ServerApp`` constructor. The ``server_fn`` function should return a
``ServerAppComponents`` object that contains the strategy instance and a
``ServerConfig`` instance.

Both ``Strategy`` and ``ServerConfig`` classes can be configured with parameters. The
``Context`` object passed to ``server_fn`` contains the values specified in the
``[tool.flwr.app.config]`` table in your ``pyproject.toml`` (a snippet is shown below).
To access these values, use ``context.run_config``.

.. code-block:: toml

    # ...

    [tool.flwr.app.config]
    num-server-rounds = 10
    fraction-fit = 0.5

    # ...

Customize an existing strategy with callback functions
------------------------------------------------------

Existing strategies provide several ways to customize their behavior. Callback functions
allow strategies to call user-provided code during execution. This approach enables you
to modify the strategy's partial behavior without rewriting the whole class from zero.

Configuring client fit and client evaluate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The server can pass new configuration values to the client each round by providing a
function to ``on_fit_config_fn``. The provided function will be called by the strategy
and must return a dictionary of configuration key value pairs that will be sent to the
client. It must return a dictionary of arbitrary configuration values ``client.fit`` and
``client.evaluate`` functions during each round of federated learning.

.. code-block:: python

    from flwr.common import Context
    from flwr.server.strategy import FedAvg
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig


    def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
        """Return a function which returns training configurations."""

        def fit_config(server_round: int) -> Dict[str, str]:
            """Return a configuration with static batch size and (local) epochs."""
            config = {
                "learning_rate": str(0.001),
                "batch_size": str(32),
            }
            return config

        return fit_config


    def server_fn(context: Context):
        # Read num_rounds from context
        num_rounds = context.run_config["num-server-rounds"]
        config = ServerConfig(num_rounds=num_rounds)

        # Instantiate FedAvg strategy
        strategy = FedAvg(
            fraction_fit=context.run_config["fraction-fit"],
            fraction_evaluate=1.0,
            on_fit_config_fn=get_on_fit_config_fn(),
        )

        # Create and return ServerAppComponents
        return ServerAppComponents(strategy=strategy, config=config)


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

The ``on_fit_config_fn`` can be used to pass arbitrary configuration values from server
to client and potentially change these values each round, for example, to adjust the
learning rate. The client will receive the dictionary returned by the
``on_fit_config_fn`` in its own ``client.fit()`` function. And while the values can be
also passed directly via the context this function can be a place to implement finer
control over the `fit` behaviour that may not be achieved by the context, which sets
fixed values.

Similar to ``on_fit_config_fn``, there is also ``on_evaluate_config_fn`` to customize
the configuration sent to ``client.evaluate()``

Configuring server-side evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Server-side evaluation can be enabled by passing an evaluation function to
``evaluate_fn``.

Implement a novel strategy
--------------------------

Writing a fully custom strategy is a bit more involved, but it provides the most
flexibility. Read the `Implementing Strategies <how-to-implement-strategies.html>`_
guide to learn more.
