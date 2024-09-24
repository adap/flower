Implement strategies
====================

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
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, FitIns]]:
            """Configure the next round of training."""

        @abstractmethod
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate training results."""

        @abstractmethod
        def configure_evaluate(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, EvaluateIns]]:
            """Configure the next round of evaluation."""

        @abstractmethod
        def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
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

        def configure_fit(self, server_round, parameters, client_manager):
            # Your implementation here

        def aggregate_fit(self, server_round, results, failures):
            # Your implementation here

        def configure_evaluate(self, server_round, parameters, client_manager):
            # Your implementation here

        def aggregate_evaluate(self, server_round, results, failures):
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

        Note left of S: Centralized<br/>Evaluation
        rect rgb(249, 219, 130)

        S->>Strategy: evaluate
        activate Strategy
        Strategy-->>S: Centralized evaluation result
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
    parameters = fl.common.ndarrays_to_parameters(weights)

    # Use the serialized parameters as the initial global parameters 
    strategy = fl.server.strategy.FedAvg(
        initial_parameters=parameters,
    )
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)

The Flower server will call :code:`initialize_parameters`, which either returns the parameters that were passed to :code:`initial_parameters`, or :code:`None`. If no parameters are returned from :code:`initialize_parameters` (i.e., :code:`None`), the server will randomly select one client and ask it to provide its parameters. This is a convenience feature and not recommended in practice, but it can be useful for prototyping. In practice, it is recommended to always use server-side parameter initialization.

.. note::

    Server-side parameter initialization is a powerful mechanism. It can be used, for example, to resume training from a previously saved checkpoint. It is also the fundamental capability needed to implement hybrid approaches, for example, to fine-tune a pre-trained model using federated learning.

The :code:`configure_fit` method
--------------------------------

:code:`configure_fit` is responsible for configuring the upcoming round of training. What does *configure* mean in this context? Configuring a round means selecting clients and deciding what instructions to send to these clients. The signature of :code:`configure_fit` makes this clear:

.. code-block:: python

    @abstractmethod
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

The return value is a list of tuples, each representing the instructions that will be sent to a particular client. Strategy implementations usually perform the following steps in :code:`configure_fit`:

* Use the :code:`client_manager` to randomly sample all (or a subset of) available clients (each represented as a :code:`ClientProxy` object)
* Pair each :code:`ClientProxy` with the same :code:`FitIns` holding the current global model :code:`parameters` and :code:`config` dict

More sophisticated implementations can use :code:`configure_fit` to implement custom client selection logic. A client will only participate in a round if the corresponding :code:`ClientProxy` is included in the list returned from :code:`configure_fit`.

.. note::

    The structure of this return value provides a lot of flexibility to the user. Since instructions are defined on a per-client basis, different instructions can be sent to each client. This enables custom strategies to train, for example, different models on different clients, or use different hyperparameters on different clients (via the :code:`config` dict).

The :code:`aggregate_fit` method
--------------------------------

:code:`aggregate_fit` is responsible for aggregating the results returned by the clients that were selected and asked to train in :code:`configure_fit`.

.. code-block:: python

    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""

Of course, failures can happen, so there is no guarantee that the server will get results from all the clients it sent instructions to (via :code:`configure_fit`). :code:`aggregate_fit` therefore receives a list of :code:`results`, but also a list of :code:`failures`.

:code:`aggregate_fit` returns an optional :code:`Parameters` object and a dictionary of aggregated metrics. The :code:`Parameters` return value is optional because :code:`aggregate_fit` might decide that the results provided are not sufficient for aggregation (e.g., too many failures).

The :code:`configure_evaluate` method
-------------------------------------

:code:`configure_evaluate` is responsible for configuring the upcoming round of evaluation. What does *configure* mean in this context? Configuring a round means selecting clients and deciding what instructions to send to these clients. The signature of :code:`configure_evaluate` makes this clear:

.. code-block:: python

    @abstractmethod
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""

The return value is a list of tuples, each representing the instructions that will be sent to a particular client. Strategy implementations usually perform the following steps in :code:`configure_evaluate`:

* Use the :code:`client_manager` to randomly sample all (or a subset of) available clients (each represented as a :code:`ClientProxy` object)
* Pair each :code:`ClientProxy` with the same :code:`EvaluateIns` holding the current global model :code:`parameters` and :code:`config` dict

More sophisticated implementations can use :code:`configure_evaluate` to implement custom client selection logic. A client will only participate in a round if the corresponding :code:`ClientProxy` is included in the list returned from :code:`configure_evaluate`.

.. note::

    The structure of this return value provides a lot of flexibility to the user. Since instructions are defined on a per-client basis, different instructions can be sent to each client. This enables custom strategies to evaluate, for example, different models on different clients, or use different hyperparameters on different clients (via the :code:`config` dict).


The :code:`aggregate_evaluate` method
-------------------------------------

:code:`aggregate_evaluate` is responsible for aggregating the results returned by the clients that were selected and asked to evaluate in :code:`configure_evaluate`.

.. code-block:: python

    @abstractmethod
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""

Of course, failures can happen, so there is no guarantee that the server will get results from all the clients it sent instructions to (via :code:`configure_evaluate`). :code:`aggregate_evaluate` therefore receives a list of :code:`results`, but also a list of :code:`failures`.

:code:`aggregate_evaluate` returns an optional :code:`float` (loss) and a dictionary of aggregated metrics. The :code:`float` return value is optional because :code:`aggregate_evaluate` might decide that the results provided are not sufficient for aggregation (e.g., too many failures).

The :code:`evaluate` method
---------------------------

:code:`evaluate` is responsible for evaluating model parameters on the server-side. Having :code:`evaluate` in addition to :code:`configure_evaluate`/:code:`aggregate_evaluate` enables strategies to perform both servers-side and client-side (federated) evaluation.

.. code-block:: python

    @abstractmethod
    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current model parameters."""

The return value is again optional because the strategy might not need to implement server-side evaluation or because the user-defined :code:`evaluate` method might not complete successfully (e.g., it might fail to load the server-side evaluation data).
