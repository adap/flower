:og:description: Configure Flower clients by sending configurations from the strategy to the clients and control client-side hyperparameters dynamically.
.. meta::
    :description: Configure Flower clients by sending configurations from the strategy to the clients and control client-side hyperparameters dynamically.

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.app.Message.html

.. |configrecord_link| replace:: ``ConfigRecord``

.. _configrecord_link: ref-api/flwr.app.ConfigRecord.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.app.ArrayRecord.html

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.clientapp.ClientApp.html

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.serverapp.ServerApp.html

.. |fedavg_link| replace:: ``FedAvg``

.. _fedavg_link: ref-api/flwr.serverapp.strategy.FedAvg.html

.. |strategy_start_link| replace:: ``start``

.. _strategy_start_link: ref-api/flwr.serverapp.strategy.Strategy.html#flwr.serverapp.strategy.Strategy.start

Configure a ``ClientApp``
=========================

Flower provides the ability to send configuration values to clients, allowing
server-side control over client behavior. This feature enables flexible and dynamic
adjustment of client-side hyperparameters, improving collaboration and experimentation.

Sending ``ConfigRecords`` to a ``ClientApp``
--------------------------------------------

Make use of a |configrecord_link|_ to send configuration values in a |message_link|_
from your |serverapp_link|_ to a |clientapp_link|_. A ``ConfigRecord`` is a special type
of Python dictionary that allows communicating basic types such as ``int``, ``float``,
``string``, ``bool`` and also ``bytes`` if you need to communicate more complex data
structures that need to be serialized. Lists of these types are also supported.

Let's see a few examples:

.. code-block:: python

    from flwr.app import ConfigRecord

    # A config record can hold basic scalars
    config = ConfigRecord({"lr": 0.1, "max-local-steps": 20000, "loss-w": [0.1, 0.2]})

    # It can also communicate strings and booleans
    config = ConfigRecord({"augment": True, "wandb-project-name": "awesome-flower-app"})

When you use a Flower strategy, the easiest way to get your ``ConfigRecord``
communicated as part of the ``Message`` that gets sent to the ``ClientApp`` is by
passing it to the |strategy_start_link|_ of your strategy of choice (e.g.
|fedavg_link|_). Let's see how this looks in code:

.. code-block:: python
    :emphasize-lines: 15,21

    # Create ServerApp
    app = ServerApp()


    @app.main()
    def main(grid: Grid, context: Context) -> None:
        """Main entry point for the ServerApp."""

        # ... Read run config, initialize global model, etc
        # Initialize FedAvg strategy
        strategy = FedAvg()

        # Construct the config to be embedded into the Messages that will
        # be sent to the ClientApps
        config = ConfigRecord({"lr": 0.1, "optim": "adam-w", "augment": True})

        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=config,
            num_rounds=10,
        )

Passing the above ``ConfigRecord`` to the strategy's ``start`` method ensures that the
exact same ``ConfigRecord`` is received on the client side. But what if we'd like the
configuration to change during the course of the federated learning process or as rounds
advance?

.. note::

    Note that Flower strategies insert the current server round number into the
    ``ConfigRecord`` for you under the key ``server-round``. In this way, the
    ``ClientApp`` knows what's the current round of the federated learning process. Note
    this is always inserted even if no ``ConfigRecord`` is passed to the strategy
    ``start`` method. When that's the case, the only content of the ``ConfigRecord``
    that arrives to the ``ClientApp`` will be such key with the corresponding round
    number.

Dynamic modification of ``ConfigRecord``
----------------------------------------

Given a ``ConfigRecord`` passed upon starting the execution of a strategy (i.e. passed
to the |strategy_start_link|_ method), the contents of the ``ConfigRecord`` that arrive
to the ``ClientApp`` won't change (with the exception of the value under the
``server-round`` key).

However, some applications do benefit or even require certain dynamism in the
configuration values that one might send over to the ``ClientApps``. For example, the
learning rate the local optimizers at the ``ClientApps`` make use of. As the federated
learning rounds go by, it is often reasonable to reduce the learning rate. This dynamism
can be introduced at the strategy by implementing a custom strategy that just overrides
the ``configure_train`` method. This method is responsible for, among other aspects, to
create the ``Messages`` that will be sent to the ``ClientApps``. These ``Messages``
would typically include an |arrayrecord_link|_ carrying the parameters of the model to
be federated as well as the ``ConfigRecord`` containing the configurations that the
``ClientApp`` should use. Let's see how to design a custom strategy that alters the
``ConfigRecord`` passed to the ``start`` method.

.. tip::

    To learn more about how ``configure_train`` and other methods in the strategies
    check the :doc:`Strategies Explainer <how-to-implement-strategies>`.

Let's create a new class inheriting from |fedavg_link|_ and override the
``configure_train`` method. We then use this new strategy in our ``ServerApp``.

.. code-block:: python
    :emphasize-lines: 13,14

    from typing import Iterable
    from flwr.serverapp import Grid
    from flwr.serverapp.strategy import FedAvg
    from flwr.app import ArrayRecord, ConfigRecord, Message


    class CustomFedAdagrad(FedAvg):
        def configure_train(
            self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
        ) -> Iterable[Message]:
            """Configure the next round of federated training and maybe do LR decay."""
            # Decrease learning rate by a factor of 0.5 every 5 rounds
            # Note: server_round starts at 1, not 0
            if server_round % 5 == 0:
                config["lr"] *= 0.5
                print("LR decreased to:", config["lr"])
            # Pass the updated config and the rest of arguments to the parent class
            return super().configure_train(server_round, arrays, config, grid)

In this how-to guide, we have shown how to define (when calling the ``start`` method of
the strategy) and modify (by overriding the ``configure_train`` method) a
``ConfigRecord`` to customize how ``ClientApps`` perform training. You can follow
equivalent steps to define and customize the ``ConfigRecord`` for an evaluation round.
To do this, use the ``evaluate_config`` argument in the strategy's ``start`` method and
then optionally override the ``configure_evaluate`` method.
