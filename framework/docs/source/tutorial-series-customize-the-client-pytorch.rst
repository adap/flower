Customize the client
====================

Welcome to the fourth part of the Flower federated learning tutorial. In the previous
parts of this tutorial, we introduced federated learning with PyTorch and Flower
(:doc:`part 1 <tutorial-series-get-started-with-flower-pytorch>`), we learned how
strategies can be used to customize the execution on both the server and the clients
(:doc:`part 2 <tutorial-series-use-a-federated-learning-strategy-pytorch>`) and we built
our own custom strategy from scratch (:doc:`part 3
<tutorial-series-build-a-strategy-from-scratch-pytorch>`).

In this final tutorial, we revisit ``NumPyClient`` and introduce a new baseclass for
building clients, simply named ``Client``. In previous parts of this tutorial, we've
based our client on ``NumPyClient``, a convenience class which makes it easy to work
with machine learning libraries that have good NumPy interoperability. With ``Client``,
we gain a lot of flexibility that we didn't have before, but we'll also have to do a few
things that we didn't have to do before.

    `Star Flower on GitHub <https://github.com/adap/flower>`__ ⭐️ and join the Flower
    community on Flower Discuss and the Flower Slack to connect, ask questions, and get
    help:

    - `Join Flower Discuss <https://discuss.flower.ai/>`__ We'd love to hear from you in
      the ``Introduction`` topic! If anything is unclear, post in ``Flower Help -
      Beginners``.
    - `Join Flower Slack <https://flower.ai/join-slack>`__ We'd love to hear from you in
      the ``#introductions`` channel! If anything is unclear, head over to the
      ``#questions`` channel.

Let's go deeper and see what it takes to move from ``NumPyClient`` to ``Client``! 🌼

Preparation
-----------

Before we begin with the actual code, let's make sure that we have everything we need.

Installing dependencies
~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    If you've completed part 1 of the tutorial, you can skip this step.

First, we install the Flower package ``flwr``:

.. code-block:: shell

    # In a new Python environment
    $ pip install -U "flwr[simulation]"

Then, we create a new Flower app called ``flower-tutorial`` using the PyTorch template.
We also specify a username (``flwrlabs``) for the project:

.. code-block:: shell

    $ flwr new flower-tutorial --framework pytorch --username flwrlabs

After running the command, a new directory called ``flower-tutorial`` will be created.
It should have the following structure:

.. code-block:: shell

    flower-tutorial
    ├── README.md
    ├── flower_tutorial
    │   ├── __init__.py
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

Next, we install the project and its dependencies, which are specified in the
``pyproject.toml`` file:

.. code-block:: shell

    $ cd flower-tutorial
    $ pip install -e .

Revisiting NumPyClient
----------------------

So far, we've implemented our client by subclassing ``flwr.client.NumPyClient``. The two
methods that were implemented in ``client_app.py`` are ``fit`` and ``evaluate``.

.. code-block:: python

    class FlowerClient(NumPyClient):
        def __init__(self, net, trainloader, valloader, local_epochs):
            self.net = net
            self.trainloader = trainloader
            self.valloader = valloader
            self.local_epochs = local_epochs
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.to(self.device)

        def fit(self, parameters, config):
            set_weights(self.net, parameters)
            train_loss = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                self.device,
            )
            return (
                get_weights(self.net),
                len(self.trainloader.dataset),
                {"train_loss": train_loss},
            )

        def evaluate(self, parameters, config):
            set_weights(self.net, parameters)
            loss, accuracy = test(self.net, self.valloader, self.device)
            return loss, len(self.valloader.dataset), {"accuracy": accuracy}

Then, we have the function ``client_fn`` that is used by Flower to create the
``FlowerClient`` instances on demand. Finally, we create the ``ClientApp`` and pass the
``client_fn`` to it.

.. code-block:: python

    def client_fn(context: Context):
        # Load model and data
        net = Net()
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        trainloader, valloader = load_data(partition_id, num_partitions)
        local_epochs = context.run_config["local-epochs"]

        # Return Client instance
        return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


    # Flower ClientApp
    app = ClientApp(
        client_fn,
    )

We've seen this before, there's nothing new so far. Next, in ``server_app.py``, the
number of federated learning rounds are preconfigured in the ``ServerConfig`` and in the
same module, the ``ServerApp`` is created with this config:

.. code-block:: python

    def server_fn(context: Context):
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]
        fraction_fit = context.run_config["fraction-fit"]

        # Initialize model parameters
        ndarrays = get_weights(Net())
        parameters = ndarrays_to_parameters(ndarrays)

        # Define strategy
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

Finally, we run the simulation to see the output we get:

.. code-block:: shell

    $ flwr run .

This works as expected, ten clients are training for three rounds of federated learning.

Let's dive a little bit deeper and discuss how Flower executes this simulation. Whenever
a client is selected to do some work, under the hood, Flower launches the ``ClientApp``
object which in turn calls the function ``client_fn`` to create an instance of our
``FlowerClient`` (along with loading the model and the data).

But here's the perhaps surprising part: Flower doesn't actually use the ``FlowerClient``
object directly. Instead, it wraps the object to makes it look like a subclass of
``flwr.client.Client``, not ``flwr.client.NumPyClient``. In fact, the Flower core
framework doesn't know how to handle ``NumPyClient``'s, it only knows how to handle
``Client``'s. ``NumPyClient`` is just a convenience abstraction built on top of
``Client``.

Instead of building on top of ``NumPyClient``, we can directly build on top of
``Client``.

Moving from ``NumPyClient`` to ``Client``
-----------------------------------------

Let's try to do the same thing using ``Client`` instead of ``NumPyClient``. Create a new
file called ``custom_client_app.py`` and copy the following code into it:

.. code-block:: python

    from typing import List

    import numpy as np
    import torch
    from flwr.client import Client, ClientApp
    from flwr.common import (
        Code,
        Context,
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
        GetParametersIns,
        GetParametersRes,
        Status,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )

    from flower_tutorial.task import Net, get_weights, load_data, set_weights, test, train


    class FlowerClient(Client):
        def __init__(self, partition_id, net, trainloader, valloader, local_epochs):
            self.partition_id = partition_id
            self.net = net
            self.trainloader = trainloader
            self.valloader = valloader
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.local_epochs = local_epochs

        def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
            print(f"[Client {self.partition_id}] get_parameters")

            # Get parameters as a list of NumPy ndarray's
            ndarrays: List[np.ndarray] = get_weights(self.net)

            # Serialize ndarray's into a Parameters object
            parameters = ndarrays_to_parameters(ndarrays)

            # Build and return response
            status = Status(code=Code.OK, message="Success")
            return GetParametersRes(
                status=status,
                parameters=parameters,
            )

        def fit(self, ins: FitIns) -> FitRes:
            print(f"[Client {self.partition_id}] fit, config: {ins.config}")

            # Deserialize parameters to NumPy ndarray's
            parameters_original = ins.parameters
            ndarrays_original = parameters_to_ndarrays(parameters_original)

            # Update local model, train, get updated parameters
            set_weights(self.net, ndarrays_original)
            train(self.net, self.trainloader, self.local_epochs, self.device)
            ndarrays_updated = get_weights(self.net)

            # Serialize ndarray's into a Parameters object
            parameters_updated = ndarrays_to_parameters(ndarrays_updated)

            # Build and return response
            status = Status(code=Code.OK, message="Success")
            return FitRes(
                status=status,
                parameters=parameters_updated,
                num_examples=len(self.trainloader),
                metrics={},
            )

        def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
            print(f"[Client {self.partition_id}] evaluate, config: {ins.config}")

            # Deserialize parameters to NumPy ndarray's
            parameters_original = ins.parameters
            ndarrays_original = parameters_to_ndarrays(parameters_original)

            set_weights(self.net, ndarrays_original)
            loss, accuracy = test(self.net, self.valloader, self.device)

            # Build and return response
            status = Status(code=Code.OK, message="Success")
            return EvaluateRes(
                status=status,
                loss=float(loss),
                num_examples=len(self.valloader),
                metrics={"accuracy": float(accuracy)},
            )


    def client_fn(context: Context) -> Client:
        net = Net()
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        local_epochs = context.run_config["local-epochs"]
        trainloader, valloader = load_data(partition_id, num_partitions)
        return FlowerClient(
            partition_id, net, trainloader, valloader, local_epochs
        ).to_client()


    # Create the ClientApp
    app = ClientApp(client_fn=client_fn)

Next, we update the ``pyproject.toml`` so that Flower uses the new module:

.. code-block:: toml

    [tool.flwr.app.components]
    serverapp = "flower_tutorial.server_app:app"
    clientapp = "flower_tutorial.custom_client_app:app"

Before we discuss the code in more detail, let's try to run it! Gotta make sure our new
``Client``-based client works, right? We run the simulation as follows:

.. code-block:: shell

    $ flwr run .

That's it, we're now using ``Client``. It probably looks similar to what we've done with
``NumPyClient``. So what's the difference?

First of all, it's more code. But why? The difference comes from the fact that
``Client`` expects us to take care of parameter serialization and deserialization. For
Flower to be able to send parameters over the network, it eventually needs to turn these
parameters into ``bytes``. Turning parameters (e.g., NumPy ``ndarray``'s) into raw bytes
is called serialization. Turning raw bytes into something more useful (like NumPy
``ndarray``'s) is called deserialization. Flower needs to do both: it needs to serialize
parameters on the server-side and send them to the client, the client needs to
deserialize them to use them for local training, and then serialize the updated
parameters again to send them back to the server, which (finally!) deserializes them
again in order to aggregate them with the updates received from other clients.

The only *real* difference between Client and NumPyClient is that NumPyClient takes care
of serialization and deserialization for you. It can do so because it expects you to
return parameters as NumPy ndarray's, and it knows how to handle these. This makes
working with machine learning libraries that have good NumPy support (most of them) a
breeze.

In terms of API, there's one major difference: all methods in Client take exactly one
argument (e.g., ``FitIns`` in ``Client.fit``) and return exactly one value (e.g.,
``FitRes`` in ``Client.fit``). The methods in ``NumPyClient`` on the other hand have
multiple arguments (e.g., ``parameters`` and ``config`` in ``NumPyClient.fit``) and
multiple return values (e.g., ``parameters``, ``num_example``, and ``metrics`` in
``NumPyClient.fit``) if there are multiple things to handle. These ``*Ins`` and ``*Res``
objects in ``Client`` wrap all the individual values you're used to from
``NumPyClient``.

Custom serialization
--------------------

Here we will explore how to implement custom serialization with a simple example.

But first what is serialization? Serialization is just the process of converting an
object into raw bytes, and equally as important, deserialization is the process of
converting raw bytes back into an object. This is very useful for network communication.
Indeed, without serialization, you could not just send a Python object through the
internet.

Federated Learning relies heavily on internet communication for training by sending
Python objects back and forth between the clients and the server. This means that
serialization is an essential part of Federated Learning.

In the following section, we will write a basic example where instead of sending a
serialized version of our ``ndarray``\ s containing our parameters, we will first
convert the ``ndarray`` into sparse matrices, before sending them. This technique can be
used to save bandwidth, as in certain cases where the weights of a model are sparse
(containing many 0 entries), converting them to a sparse matrix can greatly improve
their bytesize.

Our custom serialization/deserialization functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is where the real serialization/deserialization will happen, especially in
``ndarray_to_sparse_bytes`` for serialization and ``sparse_bytes_to_ndarray`` for
deserialization. First we add the following code to ``task.py``:

.. code-block:: python

    from io import BytesIO
    from typing import cast

    import numpy as np

    from flwr.common.typing import NDArray, NDArrays, Parameters


    def ndarrays_to_sparse_parameters(ndarrays: NDArrays) -> Parameters:
        """Convert NumPy ndarrays to parameters object."""
        tensors = [ndarray_to_sparse_bytes(ndarray) for ndarray in ndarrays]
        return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


    def sparse_parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
        """Convert parameters object to NumPy ndarrays."""
        return [sparse_bytes_to_ndarray(tensor) for tensor in parameters.tensors]


    def ndarray_to_sparse_bytes(ndarray: NDArray) -> bytes:
        """Serialize NumPy ndarray to bytes."""
        bytes_io = BytesIO()

        if len(ndarray.shape) > 1:
            # We convert our ndarray into a sparse matrix
            ndarray = torch.tensor(ndarray).to_sparse_csr()

            # And send it byutilizing the sparse matrix attributes
            # WARNING: NEVER set allow_pickle to true.
            # Reason: loading pickled data can execute arbitrary code
            # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
            np.savez(
                bytes_io,  # type: ignore
                crow_indices=ndarray.crow_indices(),
                col_indices=ndarray.col_indices(),
                values=ndarray.values(),
                allow_pickle=False,
            )
        else:
            # WARNING: NEVER set allow_pickle to true.
            # Reason: loading pickled data can execute arbitrary code
            # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
            np.save(bytes_io, ndarray, allow_pickle=False)
        return bytes_io.getvalue()


    def sparse_bytes_to_ndarray(tensor: bytes) -> NDArray:
        """Deserialize NumPy ndarray from bytes."""
        bytes_io = BytesIO(tensor)
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
        loader = np.load(bytes_io, allow_pickle=False)  # type: ignore

        if "crow_indices" in loader:
            # We convert our sparse matrix back to a ndarray, using the attributes we sent
            ndarray_deserialized = (
                torch.sparse_csr_tensor(
                    crow_indices=loader["crow_indices"],
                    col_indices=loader["col_indices"],
                    values=loader["values"],
                )
                .to_dense()
                .numpy()
            )
        else:
            ndarray_deserialized = loader
        return cast(NDArray, ndarray_deserialized)

Client-side
~~~~~~~~~~~

To be able to serialize our ``ndarray``\ s into sparse parameters, we will just have to
call our custom functions in our ``flwr.client.Client``.

Indeed, in ``get_parameters`` we need to serialize the parameters we got from our
network using our custom ``ndarrays_to_sparse_parameters`` defined above.

In ``fit``, we first need to deserialize the parameters coming from the server using our
custom ``sparse_parameters_to_ndarrays`` and then we need to serialize our local results
with ``ndarrays_to_sparse_parameters``.

In ``evaluate``, we will only need to deserialize the global parameters with our custom
function. In a new file called ``serde_client_app.py``, copy the following code into it:

.. code-block:: python

    from typing import List

    import numpy as np
    import torch
    from flwr.client import Client, ClientApp
    from flwr.common import (
        Code,
        Context,
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
        GetParametersIns,
        GetParametersRes,
        Status,
    )

    from flower_tutorial.task import (
        Net,
        get_weights,
        load_data,
        ndarrays_to_sparse_parameters,
        set_weights,
        sparse_parameters_to_ndarrays,
        test,
        train,
    )


    class FlowerClient(Client):
        def __init__(self, partition_id, net, trainloader, valloader, local_epochs):
            self.partition_id = partition_id
            self.net = net
            self.trainloader = trainloader
            self.valloader = valloader
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.local_epochs = local_epochs

        def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
            print(f"[Client {self.partition_id}] get_parameters")

            # Get parameters as a list of NumPy ndarray's
            ndarrays: List[np.ndarray] = get_weights(self.net)

            # Serialize ndarray's into a Parameters object using our custom function
            parameters = ndarrays_to_sparse_parameters(ndarrays)

            # Build and return response
            status = Status(code=Code.OK, message="Success")
            return GetParametersRes(
                status=status,
                parameters=parameters,
            )

        def fit(self, ins: FitIns) -> FitRes:
            print(f"[Client {self.partition_id}] fit, config: {ins.config}")

            # Deserialize parameters to NumPy ndarray's using our custom function
            parameters_original = ins.parameters
            ndarrays_original = sparse_parameters_to_ndarrays(parameters_original)

            # Update local model, train, get updated parameters
            set_weights(self.net, ndarrays_original)
            train(self.net, self.trainloader, self.local_epochs, self.device)
            ndarrays_updated = get_weights(self.net)

            # Serialize ndarray's into a Parameters object using our custom function
            parameters_updated = ndarrays_to_sparse_parameters(ndarrays_updated)

            # Build and return response
            status = Status(code=Code.OK, message="Success")
            return FitRes(
                status=status,
                parameters=parameters_updated,
                num_examples=len(self.trainloader),
                metrics={},
            )

        def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
            print(f"[Client {self.partition_id}] evaluate, config: {ins.config}")

            # Deserialize parameters to NumPy ndarray's using our custom function
            parameters_original = ins.parameters
            ndarrays_original = sparse_parameters_to_ndarrays(parameters_original)

            set_weights(self.net, ndarrays_original)
            loss, accuracy = test(self.net, self.valloader, self.device)

            # Build and return response
            status = Status(code=Code.OK, message="Success")
            return EvaluateRes(
                status=status,
                loss=float(loss),
                num_examples=len(self.valloader),
                metrics={"accuracy": float(accuracy)},
            )


    def client_fn(context: Context) -> Client:
        net = Net()
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        local_epochs = context.run_config["local-epochs"]
        trainloader, valloader = load_data(partition_id, num_partitions)
        return FlowerClient(
            partition_id, net, trainloader, valloader, local_epochs
        ).to_client()


    # Create the ClientApp
    app = ClientApp(client_fn=client_fn)

Server-side
~~~~~~~~~~~

For this example, we will just use ``FedAvg`` as a strategy. To change the serialization
and deserialization here, we only need to reimplement the ``evaluate`` and
``aggregate_fit`` functions of ``FedAvg``. The other functions of the strategy will be
inherited from the super class ``FedAvg``.

As you can see only one line as change in ``evaluate``:

.. code-block:: python

    parameters_ndarrays = sparse_parameters_to_ndarrays(parameters)

And for ``aggregate_fit``, we will first deserialize every result we received:

.. code-block:: python

    weights_results = [
        (sparse_parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        for _, fit_res in results
    ]

And then serialize the aggregated result:

.. code-block:: python

    parameters_aggregated = ndarrays_to_sparse_parameters(aggregate(weights_results))

In a new file called ``strategy.py``, copy the following code into it:

.. code-block:: python

    from logging import WARNING
    from typing import Callable, Dict, List, Optional, Tuple, Union

    from flwr.common import FitRes, MetricsAggregationFn, NDArrays, Parameters, Scalar
    from flwr.common.logger import log
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedAvg
    from flwr.server.strategy.aggregate import aggregate

    from flower_tutorial.task import (
        ndarrays_to_sparse_parameters,
        sparse_parameters_to_ndarrays,
    )

    WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
    Setting `min_available_clients` lower than `min_fit_clients` or
    `min_evaluate_clients` can cause the server to fail when there are too few clients
    connected to the server. `min_available_clients` must be set to a value larger
    than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
    """


    class FedSparse(FedAvg):
        def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        ) -> None:
            """Custom FedAvg strategy with sparse matrices.

            Parameters
            ----------
            fraction_fit : float, optional
                Fraction of clients used during training. Defaults to 0.1.
            fraction_evaluate : float, optional
                Fraction of clients used during validation. Defaults to 0.1.
            min_fit_clients : int, optional
                Minimum number of clients used during training. Defaults to 2.
            min_evaluate_clients : int, optional
                Minimum number of clients used during validation. Defaults to 2.
            min_available_clients : int, optional
                Minimum number of total clients in the system. Defaults to 2.
            evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
                Optional function used for validation. Defaults to None.
            on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
                Function used to configure training. Defaults to None.
            on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
                Function used to configure validation. Defaults to None.
            accept_failures : bool, optional
                Whether or not accept rounds containing failures. Defaults to True.
            initial_parameters : Parameters, optional
                Initial global model parameters.
            """

            if (
                min_fit_clients > min_available_clients
                or min_evaluate_clients > min_available_clients
            ):
                log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

            super().__init__(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=on_fit_config_fn,
                on_evaluate_config_fn=on_evaluate_config_fn,
                accept_failures=accept_failures,
                initial_parameters=initial_parameters,
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            )

        def evaluate(
            self, server_round: int, parameters: Parameters
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            """Evaluate model parameters using an evaluation function."""
            if self.evaluate_fn is None:
                # No evaluation function provided
                return None

            # We deserialize using our custom method
            parameters_ndarrays = sparse_parameters_to_ndarrays(parameters)

            eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
            if eval_res is None:
                return None
            loss, metrics = eval_res
            return loss, metrics

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate fit results using weighted average."""
            if not results:
                return None, {}
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}

            # We deserialize each of the results with our custom method
            weights_results = [
                (sparse_parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]

            # We serialize the aggregated result using our custom method
            parameters_aggregated = ndarrays_to_sparse_parameters(
                aggregate(weights_results)
            )

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            return parameters_aggregated, metrics_aggregated

We can now import our new ``FedSparse`` strategy into ``server_app.py`` and update our
``server_fn`` to use it:

.. code-block:: python

    from flower_tutorial.strategy import FedSparse


    def server_fn(context: Context):
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(
            strategy=FedSparse(), config=config  # <-- pass the new strategy here
        )


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

Finally, we run the simulation.

.. code-block:: shell

    $ flwr run .

Recap
-----

In this part of the tutorial, we've seen how we can build clients by subclassing either
``NumPyClient`` or ``Client``. ``NumPyClient`` is a convenience abstraction that makes
it easier to work with machine learning libraries that have good NumPy interoperability.
``Client`` is a more flexible abstraction that allows us to do things that are not
possible in ``NumPyClient``. In order to do so, it requires us to handle parameter
serialization and deserialization ourselves.

.. note::

    If you'd like to follow along with tutorial notebooks, check out the :doc:`Tutorial
    notebooks <notebooks/index>`. Note that the notebooks use the ``run_simulation``
    approach, whereas the recommended way to run simulations in Flower is using the
    ``flwr run`` approach as shown in this tutorial.

Next steps
----------

Before you continue, make sure to join the Flower community on Flower Discuss (`Join
Flower Discuss <https://discuss.flower.ai>`__) and on Slack (`Join Slack
<https://flower.ai/join-slack/>`__).

There's a dedicated ``#questions`` Slack channel if you need help, but we'd also love to
hear who you are in ``#introductions``!

This is the final part of the Flower tutorial (for now!), congratulations! You're now
well equipped to understand the rest of the documentation. There are many topics we
didn't cover in the tutorial, we recommend the following resources:

- `Read Flower Docs <https://flower.ai/docs/>`__
- `Check out Flower Code Examples <https://flower.ai/docs/examples/>`__
- `Use Flower Baselines for your research <https://flower.ai/docs/baselines/>`__
- `Watch Flower AI Summit 2025 videos
  <https://flower.ai/events/flower-ai-summit-2025/>`__
