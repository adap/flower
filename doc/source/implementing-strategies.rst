Implementing Strategies
=======================

The strategy abstraction enables implementation of fully custom strategies. A
strategy is basically the federated learning algorithm that runs on the server.
Strategies decide how to sample clients, how to configure clients for training,
how to aggregate updates, and how to evaluate models. Flower provides a few
built-in strategies which are based on the same API described below.

The :code:`Strategy` abstraction
--------------------------------

All strategy implementation are derived from the abstract base class
:code:`flwr.server.strategy.Strategy`, both built-in implementations and third
party implementations. This means that custom strategy implementations have the
exact same capabilities at their disposal as built-in ones.

The strategy abstraction defines a few abstract methods that need to be
implemented:

.. code-block:: python

    class Strategy(ABC):

        @abstractmethod
        def configure_fit(
            self, rnd: int, weights: Weights, client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, FitIns]]:

        @abstractmethod
        def aggregate_fit(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException],
        ) -> Optional[Weights]:

        @abstractmethod
        def configure_evaluate(
            self, rnd: int, weights: Weights, client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, EvaluateIns]]:

        @abstractmethod
        def aggregate_evaluate(
            self,
            rnd: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[BaseException],
        ) -> Optional[float]:

        @abstractmethod
        def evaluate(self, weights: Weights) -> Optional[Tuple[float, float]]:

Creating a new strategy means implmenting a new :code:`class` derived from the
abstract base class :code:`Strategy` which provides implementations for the
previously shown abstract methods:

.. code-block:: python

    class SotaStrategy(Strategy):

        def configure_fit(self, rnd, weights, client_manager):
            # Your implementation here

        def aggregate_fit(self, rnd, results, failures):
            # Your implementation here

        def configure_evaluate(self, rnd, weights, client_manager):
            # Your implementation here

        def aggregate_evaluate(self, rnd, results, failures):
            # Your implementation here

        def evaluate(self, weights):
            # Your implementation here

The following sections describe each of those methods in detail.

The :code:`configure_fit` method
-----------------------------------

*coming soon*

The :code:`aggregate_fit` method
-----------------------------------

*coming soon*

The :code:`configure_evaluate` method
----------------------------------------

*coming soon*

The :code:`aggregate_evaluate` method
----------------------------------------

*coming soon*

The :code:`evaluate` method
---------------------------

*coming soon*

Deprecated methods
------------------

The following methods were replaced by updated versions with the same type
signature. Migrate to the new versions by renaming them (i.e., remove the
:code:`on_` prefix):

* :code:`on_configure_fit` (replaced by :code:`configure_fit`)
* :code:`on_aggregate_fit` (replaced by :code:`aggregate_fit`)
* :code:`on_configure_evaluate` (replaced by :code:`configure_evaluate`)
* :code:`on_aggregate_evaluate` (replaced by :code:`aggregate_evaluate`)
