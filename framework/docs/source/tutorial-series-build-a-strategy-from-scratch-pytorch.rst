Build a strategy from scratch
=============================

Welcome to the third part of the Flower federated learning tutorial. In previous parts
of this tutorial, we introduced federated learning with PyTorch and the Flower framework
(:doc:`part 1 <tutorial-series-get-started-with-flower-pytorch>`) and we learned how
strategies can be used to customize the execution on both the server and the clients
(:doc:`part 2 <tutorial-series-use-a-federated-learning-strategy-pytorch>`).

In this tutorial, we'll continue to customize the federated learning system we built
previously by creating a custom version of FedAvg using the Flower framework, Flower
Datasets, and PyTorch.

    `Star Flower on GitHub <https://github.com/adap/flower>`__ ⭐️ and join the Flower
    community on Flower Discuss and the Flower Slack to connect, ask questions, and get
    help:

    - `Join Flower Discuss <https://discuss.flower.ai/>`__ We'd love to hear from you in
      the ``Introduction`` topic! If anything is unclear, post in ``Flower Help -
      Beginners``.
    - `Join Flower Slack <https://flower.ai/join-slack>`__ We'd love to hear from you in
      the ``#introductions`` channel! If anything is unclear, head over to the
      ``#questions`` channel.

Let's build a new ``Strategy`` from scratch! 🌼

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

Build a Strategy from scratch
-----------------------------

Let's overwrite the ``configure_fit`` method such that it passes a higher learning rate
(potentially also other hyperparameters) to the optimizer of a fraction of the clients.
We will keep the sampling of the clients as it is in ``FedAvg`` and then change the
configuration dictionary (one of the ``FitIns`` attributes). Create a new module called
``strategy.py`` in the ``flower_tutorial`` directory. Next, we define a new class
``FedCustom`` that inherits from ``Strategy``. Copy and paste the following code into
``strategy.py``:

.. code-block:: python

    from typing import Dict, List, Optional, Tuple, Union

    from flwr.common import (
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
        Parameters,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.client_manager import ClientManager
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import Strategy
    from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

    from flower_tutorial.task import Net, get_weights


    class FedCustom(Strategy):
        def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
        ) -> None:
            super().__init__()
            self.fraction_fit = fraction_fit
            self.fraction_evaluate = fraction_evaluate
            self.min_fit_clients = min_fit_clients
            self.min_evaluate_clients = min_evaluate_clients
            self.min_available_clients = min_available_clients

            # Initialize model parameters
            ndarrays = get_weights(Net())
            self.initial_parameters = ndarrays_to_parameters(ndarrays)

        def __repr__(self) -> str:
            return "FedCustom"

        def initialize_parameters(
            self, client_manager: ClientManager
        ) -> Optional[Parameters]:
            """Initialize global model parameters."""
            initial_parameters = self.initial_parameters
            self.initial_parameters = None  # Don't keep initial parameters in memory
            return initial_parameters

        def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, FitIns]]:
            """Configure the next round of training."""

            # Sample clients
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )

            # Create custom configs
            n_clients = len(clients)
            half_clients = n_clients // 2
            standard_config = {"lr": 0.001}
            higher_lr_config = {"lr": 0.003}
            fit_configurations = []
            for idx, client in enumerate(clients):
                if idx < half_clients:
                    fit_configurations.append((client, FitIns(parameters, standard_config)))
                else:
                    fit_configurations.append(
                        (client, FitIns(parameters, higher_lr_config))
                    )
            return fit_configurations

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate fit results using weighted average."""

            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
            metrics_aggregated = {}
            return parameters_aggregated, metrics_aggregated

        def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, EvaluateIns]]:
            """Configure the next round of evaluation."""
            if self.fraction_evaluate == 0.0:
                return []
            config = {}
            evaluate_ins = EvaluateIns(parameters, config)

            # Sample clients
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )

            # Return client/config pairs
            return [(client, evaluate_ins) for client in clients]

        def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
            """Aggregate evaluation losses using weighted average."""

            if not results:
                return None, {}

            loss_aggregated = weighted_loss_avg(
                [
                    (evaluate_res.num_examples, evaluate_res.loss)
                    for _, evaluate_res in results
                ]
            )
            metrics_aggregated = {}
            return loss_aggregated, metrics_aggregated

        def evaluate(
            self, server_round: int, parameters: Parameters
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            """Evaluate global model parameters using an evaluation function."""

            # Let's assume we won't perform the global model evaluation on the server side.
            return None

        def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
            """Return sample size and required number of clients."""
            num_clients = int(num_available_clients * self.fraction_fit)
            return max(num_clients, self.min_fit_clients), self.min_available_clients

        def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
            """Use a fraction of available clients for evaluation."""
            num_clients = int(num_available_clients * self.fraction_evaluate)
            return max(num_clients, self.min_evaluate_clients), self.min_available_clients

The only thing left is to use the newly created custom Strategy ``FedCustom`` when
starting the experiment. In the ``server_app.py`` file, import the custom strategy and
use it in ``server_fn``:

.. code-block:: python

    from flower_tutorial.strategy import FedCustom


    def server_fn(context: Context):
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]

        # Define strategy
        strategy = FedCustom()
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

Finally, we run the simulation.

.. code-block:: shell

    $ flwr run .

Recap
-----

In this tutorial, we've seen how to implement a custom strategy. A custom strategy
enables granular control over client node configuration, result aggregation, and more.
To define a custom strategy, you only have to overwrite the abstract methods of the
(abstract) base class ``Strategy``. To make custom strategies even more powerful, you
can pass custom functions to the constructor of your new class (``__init__``) and then
call these functions whenever needed.

Next steps
----------

Before you continue, make sure to join the Flower community on Flower Discuss (`Join
Flower Discuss <https://discuss.flower.ai>`__) and on Slack (`Join Slack
<https://flower.ai/join-slack/>`__).

There's a dedicated ``#questions`` Slack channel if you need help, but we'd also love to
hear who you are in ``#introductions``!

The :doc:`Flower Federated Learning Tutorial - Part 4
<tutorial-series-customize-the-client-pytorch>` introduces ``Client``, the flexible API
underlying ``NumPyClient``.
