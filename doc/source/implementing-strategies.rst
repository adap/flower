Implementing Strategies
=======================

The strategy abstraction enables implementation of fully custom strategies. A strategy is basically the federated learning algorithm that runs on the server. Strategies decide how to sample clients, how to configure clients for traning, how to aggregate updates, and how to evaluate models. Flower provides a few buil-in strategies which are based on the same API described below.

The :code:`Strategy` abstraction
------------------------

All strategy implementation are derived from the abstract base class `flwr.server.strategy.Strategy`, both built-in implementations and third party implementations. This means that custom strategy implementations have the exact same capabilities at their disposal as built-in ones.

The strategy abstraction defines a few abstract methods that clients need to override:

.. code-block:: python

    import flwr as fl

    # Coming soon

The :code:`on_configure_fit` method
-----------------------------------

*coming soon*

The :code:`on_aggregate_fit` method
-----------------------------------

*coming soon*
