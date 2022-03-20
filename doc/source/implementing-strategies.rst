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
      """Abstract base class for server strategy implementations."""

      @abstractmethod
      def initialize_parameters(
          self, client_manager: ClientManager
      ) -> Optional[Parameters]:
          """Initialize the (global) model parameters."""

      @abstractmethod
      def configure_fit(
          self, rnd: int, parameters: Parameters, client_manager: ClientManager
      ) -> List[Tuple[ClientProxy, FitIns]]:
          """Configure the next round of training."""

      @abstractmethod
      def aggregate_fit(
          self,
          rnd: int,
          results: List[Tuple[ClientProxy, FitRes]],
          failures: List[BaseException],
      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
          """Aggregate training results."""

      @abstractmethod
      def configure_evaluate(
          self, rnd: int, parameters: Parameters, client_manager: ClientManager
      ) -> List[Tuple[ClientProxy, EvaluateIns]]:
          """Configure the next round of evaluation."""

      @abstractmethod
      def aggregate_evaluate(
          self,
          rnd: int,
          results: List[Tuple[ClientProxy, EvaluateRes]],
          failures: List[BaseException],
      ) -> Tuple[Optional[float], Dict[str, Scalar]]:
          """Aggregate evaluation results."""

      @abstractmethod
      def evaluate(
          self, parameters: Parameters
      ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
          """Evaluate the current model parameters."""


Creating a new strategy means implementing a new :code:`class` (derived from the
abstract base class :code:`Strategy`) that implements for the previously shown
abstract methods:

.. code-block:: python

    class SotaStrategy(Strategy):
        def initialize_parameters(self, client_manager):
            # Your implementation here

        def configure_fit(self, rnd, parameters, client_manager):
            # Your implementation here

        def aggregate_fit(self, rnd, results, failures):
            # Your implementation here

        def configure_evaluate(self, rnd, parameters, client_manager):
            # Your implementation here

        def aggregate_evaluate(self, rnd, results, failures):
            # Your implementation here

        def evaluate(self, parameters):
            # Your implementation here

The Flower server calls these methods in the following order:

.. mermaid::

   sequenceDiagram
      participant Strategy
      participant S as Flower Server<br/>start_server
      participant C1 as Flower Client
      participant C2 as Flower Client
      Note left of S: Get initial <br/>model parameters
      S->>Strategy: initialize_parameters
      activate Strategy
      Strategy-->>S: Parameters
      deactivate Strategy
      
      Note left of S: Federated<br/>Training
      rect rgb(249, 219, 130)

      S->>Strategy: configure_fit
      activate Strategy
      Strategy-->>S: List[Tuple[ClientProxy, FitIns]]
      deactivate Strategy
      
      S->>C1: FitIns
      activate C1
      S->>C2: FitIns
      activate C2

      C1-->>S: FitRes
      deactivate C1
      C2-->>S: FitRes
      deactivate C2

      S->>Strategy: aggregate_fit<br/>List[FitRes]
      activate Strategy
      Strategy-->>S: Aggregated model parameters
      deactivate Strategy
      
      end

      Note left of S: Federated<br/>Evaluation
      rect rgb(249, 219, 130)

      S->>Strategy: configure_evaluate
      activate Strategy
      Strategy-->>S: List[Tuple[ClientProxy, EvaluateIns]]
      deactivate Strategy
      
      S->>C1: EvaluateIns
      activate C1
      S->>C2: EvaluateIns
      activate C2

      C1-->>S: EvaluateRes
      deactivate C1
      C2-->>S: EvaluateRes
      deactivate C2

      S->>Strategy: aggregate_evaluate<br/>List[EvaluateRes]
      activate Strategy
      Strategy-->>S: Aggregated evaluation results
      deactivate Strategy
      
      end
      
      Note left of S: Centralized<br/>Evaluation
      rect rgb(249, 219, 130)

      S->>Strategy: evaluate
      activate Strategy
      Strategy-->>S: Centralized evaluation result
      deactivate Strategy
      
      end

      Note left of S: Next round, continue<br/>with federated training

The following sections describe each of those methods in more detail.

The :code:`initialize_parameters` method
----------------------------------------

:code:`initialize_parameters` is called only once, at the very beginning of an execution. It is responsible for providing the initial global model parameters in a serialized form (i.e., as a :code:`Parameters` object).

Built-in strategies return user-provided initial parameters. The following example shows how initial parameters can be passed to :code:`FedAvg`:

.. code-block:: python

    import flwr as fl
    import tensorflow as tf

    # Load model for server-side parameter initialization
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(32, 32, 3), weights=None, classes=10
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Get model weights as a list of NumPy ndarray's
    weights = model.get_weights()

    # Serialize ndarrays to `Parameters`
    parameters = fl.common.weights_to_parameters(weights)

    # Use the serialized parameters as the initial global parameters 
    strategy = fl.server.strategy.FedAvg(
        initial_parameters=parameters,
    )
    fl.server.start_server(config={"num_rounds": 3}, strategy=strategy)

The Flower server will call :code:`initialize_parameters`, which either returns the parameters that were passed to :code:`initial_parameters`, or :code:`None`. If no parameters are returned from :code:`initialize_parameters` (i.e., :code:`None`), the server will randomly select one client and ask it to provide its parameters. This is a convenience feature and not recommended in practice, but it can be useful for prototyping. In practice, it is recommended to always use server-side parameter initialization.

.. note::
    Server-side parameter initialization is a powerful mechanism. It can be used, for example, to resume training from a previously saved checkpoint. It is also the fundamental capability needed to implement hybrid approaches, for example, to fine-tune a pre-trained model using federated learning.

The :code:`configure_fit` method
--------------------------------

*coming soon*

The :code:`aggregate_fit` method
--------------------------------

*coming soon*

The :code:`configure_evaluate` method
-------------------------------------

*coming soon*

The :code:`aggregate_evaluate` method
-------------------------------------

*coming soon*

The :code:`evaluate` method
---------------------------

*coming soon*
