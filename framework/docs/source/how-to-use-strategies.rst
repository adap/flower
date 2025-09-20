:og:description: Customize federated learning in Flower with built-in strategies, callbacks, and custom server-side implementations for maximum flexibility and control.
.. meta::
    :description: Customize federated learning in Flower with built-in strategies, callbacks, and custom server-side implementations for maximum flexibility and control.

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.server.ServerApp.html

.. |strategy_link| replace:: ``Strategy``

.. _strategy_link: ref-api/flwr.serverapp.strategy.Strategy.html

.. |strategy_start_link| replace:: ``start``

.. _strategy_start_link: ref-api/flwr.serverapp.strategy.Strategy.html#flwr.serverapp.strategy.Strategy.start

.. |fedavg_link| replace:: ``FedAvg``

.. _fedavg_link: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.app.Message.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.app.ArrayRecord.html

Use strategies
==============

Flower allows full customization of the learning process through the |strategy_link|_
abstraction. A number of built-in `strategies <ref-api/flwr.serverapp.strategy.html>`_
are provided in the core framework.

There are four ways to customize the way Flower orchestrates the learning process on the
server side:

- Use an existing strategy, for example, ``FedAvg``
- Customize an existing strategy with callback functions to its ``start`` method
- Customize an existing strategy by overriding one or more of it's methods.
- Implement a novel strategy from scratch

Use an existing strategy
------------------------

Flower comes with a number of popular federated learning ``Strategies`` which can be
instantiated as follows as part of a simle |serverapp_link|_:

.. code-block:: python

    # Create ServerApp
    app = ServerApp()


    @app.main()
    def main(grid: Grid, context: Context) -> None:
        """Main entry point for the ServerApp."""

        # Load global model
        global_model = Net()
        arrays = ArrayRecord(global_model.state_dict())

        # Initialize FedAvg strategy with default settings
        strategy = FedAvg()

        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
        )

In the code above, instantiating ``FedAvg`` does not launch the logic built into the
strategy (i.e. sampling nodes, communicating |message_link|_, perform aggregation, etc).
In order to do so, we need to execute the |strategy_start_link|_ method.

The above ``ServerApp`` is very minimal, makes use of the default settings for
``FedAvg`` and only passes the required arguments to the ``start`` method. Let's see in
a bit more detail what options do we have when instantiating strategies and when
launching it.

Parameterizing an existing strategy
-----------------------------------

.. code-block:: python


Using the strategy's ``start`` method
-------------------------------------

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

    from flwr.app import Context
    from flwr.server.strategy import FedAvg
    from flwr.server import ServerAppComponents, ServerConfig
    from flwr.serverapp import ServerApp


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
