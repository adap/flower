Use strategies
==============

Flower allows full customization of the learning process through the :code:`Strategy` abstraction. A number of built-in strategies are provided in the core framework.  

There are three ways to customize the way Flower orchestrates the learning process on the server side:

* Use an existing strategy, for example, :code:`FedAvg`
* Customize an existing strategy with callback functions
* Implement a novel strategy


Use an existing strategy
------------------------

Flower comes with a number of popular federated learning strategies built-in. A built-in strategy can be instantiated as follows:

.. code-block:: python

    import flwr as fl

    strategy = fl.server.strategy.FedAvg()
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)

This creates a strategy with all parameters left at their default values and passes it to the :code:`start_server` function. It is usually recommended to adjust a few parameters during instantiation:

.. code-block:: python

    import flwr as fl

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  # Sample 10% of available clients for the next round
        min_fit_clients=10,  # Minimum number of clients to be sampled for the next round
        min_available_clients=80,  # Minimum number of clients that need to be connected to the server before a training round can start
    )
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)


Customize an existing strategy with callback functions
------------------------------------------------------

Existing strategies provide several ways to customize their behaviour. Callback functions allow strategies to call user-provided code during execution.

Configuring client fit and client evaluate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The server can pass new configuration values to the client each round by providing a function to :code:`on_fit_config_fn`. The provided function will be called by the strategy and must return a dictionary of configuration key values pairs that will be sent to the client.
It must return a dictionary of arbitrary configuration values  :code:`client.fit` and :code:`client.evaluate` functions during each round of federated learning. 

.. code-block:: python

    import flwr as fl

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

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        min_fit_clients=10,
        min_available_clients=80,
        on_fit_config_fn=get_on_fit_config_fn(),
    )
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)

The :code:`on_fit_config_fn` can be used to pass arbitrary configuration values from server to client, and potentially change these values each round, for example, to adjust the learning rate.
The client will receive the dictionary returned by the :code:`on_fit_config_fn` in its own :code:`client.fit()` function.

Similar to :code:`on_fit_config_fn`, there is also :code:`on_evaluate_config_fn` to customize the configuration sent to :code:`client.evaluate()`

Configuring server-side evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Server-side evaluation can be enabled by passing an evaluation function to :code:`evaluate_fn`.


Implement a novel strategy
--------------------------

Writing a fully custom strategy is a bit more involved, but it provides the most flexibility. Read the `Implementing Strategies <how-to-implement-strategies.html>`_ guide to learn more.
