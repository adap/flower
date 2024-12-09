:og:description: Learn how to train a classification model on the Higgs dataset using federated learning with Flower and XGBoost in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a classification model on the Higgs dataset using federated learning with Flower and XGBoost in this step-by-step tutorial.

.. _quickstart-xgboost:

Quickstart XGBoost
==================

XGBoost
-------

EXtreme Gradient Boosting (**XGBoost**) is a robust and efficient implementation of
gradient-boosted decision tree (**GBDT**), that maximises the computational boundaries
for boosted tree methods. It's primarily designed to enhance both the performance and
computational speed of machine learning models. In XGBoost, trees are constructed
concurrently, unlike the sequential approach taken by GBDT.

Often, for tabular data on medium-sized datasets with fewer than 10k training examples,
XGBoost surpasses the results of deep learning techniques.

Why Federated XGBoost?
~~~~~~~~~~~~~~~~~~~~~~

As the demand for data privacy and decentralized learning grows, there's an increasing
requirement to implement federated XGBoost systems for specialised applications, like
survival analysis and financial fraud detection.

Federated learning ensures that raw data remains on the local device, making it an
attractive approach for sensitive domains where data privacy is paramount. Given the
robustness and efficiency of XGBoost, combining it with federated learning offers a
promising solution for these specific challenges.

Environment Setup
-----------------

In this tutorial, we learn how to train a federated XGBoost model on the HIGGS dataset
using Flower and the ``xgboost`` package to perform a binary classification task. We use
a simple example (`full code xgboost-quickstart
<https://github.com/adap/flower/tree/main/examples/xgboost-quickstart>`_) to demonstrate
how federated XGBoost works, and then we dive into a more complex comprehensive example
(`full code xgboost-comprehensive
<https://github.com/adap/flower/tree/main/examples/xgboost-comprehensive>`_) to run
various experiments.

It is recommended to create a virtual environment and run everything within a
:doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

We first need to install Flower and Flower Datasets. You can do this by running :

.. code-block:: shell

    # In a new Python environment
    $ pip install flwr flwr-datasets

Since we want to use ``xgboost`` package to build up XGBoost trees, let's go ahead and
install ``xgboost``:

.. code-block:: shell

    $ pip install xgboost

The Configurations
~~~~~~~~~~~~~~~~~~

We define all required configurations / hyper-parameters inside the ``pyproject.toml``
file:

.. code-block:: toml

    [tool.flwr.app.config]
    # ServerApp
    num-server-rounds = 3
    fraction-fit = 0.1
    fraction-evaluate = 0.1

    # ClientApp
    local-epochs = 1
    params.objective = "binary:logistic"
    params.eta = 0.1  # Learning rate
    params.max-depth = 8
    params.eval-metric = "auc"
    params.nthread = 16
    params.num-parallel-tree = 1
    params.subsample = 1
    params.tree-method = "hist"

The ``local-epochs`` represents the number of iterations for local tree boost. We use
CPU for the training in default. One can assign it to a GPU by setting ``tree_method``
to ``gpu_hist``. We use AUC as evaluation metric.

The Data
~~~~~~~~

This tutorial uses `Flower Datasets <https://flower.ai/docs/datasets/>`_ to easily
download and partition the `HIGGS` dataset.

.. code-block:: python

    # Load (HIGGS) dataset and partition.
    # We use a small subset (num_partitions=20) of the dataset for demonstration to speed up the data loading process.
    partitioner = IidPartitioner(num_partitions=20)
    fds = FederatedDataset(dataset="jxie/higgs", partitioners={"train": partitioner})

    # Load the partition for this `partition_id`
    partition = fds.load_partition(partition_id, split="train")
    partition.set_format("numpy")

In this example, we split the dataset into 20 partitions with uniform distribution
(`IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_).
Then, we load the partition for the given client based on ``partition_id``.

Subsequently, we train/test split using the given partition (client's local data), and
reformat data to DMatrix for the ``xgboost`` package.

.. code-block:: python

    # Train/test splitting
    train_data, valid_data, num_train, num_val = train_test_split(
        partition, test_fraction=0.2, seed=42
    )

    # Reformat data to DMatrix for xgboost
    train_dmatrix = transform_dataset_to_dmatrix(train_data)
    valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

The functions of ``train_test_split`` and ``transform_dataset_to_dmatrix`` are defined
as below:

.. code-block:: python

    def train_test_split(partition, test_fraction, seed):
        """Split the data into train and validation set given split rate."""
        train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
        partition_train = train_test["train"]
        partition_test = train_test["test"]

        num_train = len(partition_train)
        num_test = len(partition_test)

        return partition_train, partition_test, num_train, num_test


    def transform_dataset_to_dmatrix(data):
        """Transform dataset to DMatrix format for xgboost."""
        x = data["inputs"]
        y = data["label"]
        new_data = xgb.DMatrix(x, label=y)
        return new_data

The ClientApp
~~~~~~~~~~~~~

*Clients* are responsible for generating individual weight-updates for the model based
on their local datasets. Let's first see how we define Flower client for XGBoost. We
follow the general rule to define ``FlowerClient`` class inherited from
``fl.client.Client``.

.. code-block:: python

    # Define Flower Client and client_fn
    class FlowerClient(Client):
        def __init__(
            self,
            train_dmatrix,
            valid_dmatrix,
            num_train,
            num_val,
            num_local_round,
            params,
        ):
            self.train_dmatrix = train_dmatrix
            self.valid_dmatrix = valid_dmatrix
            self.num_train = num_train
            self.num_val = num_val
            self.num_local_round = num_local_round
            self.params = params

All required parameters defined above are passed to ``FlowerClient``'s constructor.

Then, we override ``fit`` and ``evaluate`` methods insides ``FlowerClient`` class as
follows.

.. code-block:: python

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

In ``fit``, at the first round, we call ``xgb.train()`` to build up the first set of
trees. From the second round, we load the global model sent from server to new build
Booster object, and then update model weights on local training data with function
``_local_boost`` as follows:

.. code-block:: python

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds()
            - self.num_local_round : bst_input.num_boosted_rounds()
        ]

        return bst

Given ``num_local_round``, we update trees by calling ``bst_input.update`` method. After
training, the last ``N=num_local_round`` trees will be extracted to send to the server.

.. code-block:: python

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"AUC": auc},
        )

In ``evaluate``, after loading the global model, we call ``bst.eval_set`` function to
conduct evaluation on valid set. The AUC value will be returned.

The ServerApp
~~~~~~~~~~~~~

After the local training on clients, clients' model updates are sent to the *server*,
which aggregates them to produce a better model. Finally, the *server* sends this
improved model version back to each *client* to complete a federated round.

In the file named ``server_app.py``, we define a strategy for XGBoost bagging
aggregation:

.. code-block:: python

    # Define strategy
    strategy = FedXgbBagging(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=config_func,
        on_fit_config_fn=config_func,
        initial_parameters=parameters,
    )


    def evaluate_metrics_aggregation(eval_metrics):
        """Return an aggregated metric (AUC) for evaluation."""
        total_num = sum([num for num, _ in eval_metrics])
        auc_aggregated = (
            sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
        )
        metrics_aggregated = {"AUC": auc_aggregated}
        return metrics_aggregated


    def config_func(rnd: int) -> Dict[str, str]:
        """Return a configuration with global epochs."""
        config = {
            "global_round": str(rnd),
        }
        return config

An ``evaluate_metrics_aggregation`` function is defined to collect and wighted average
the AUC values from clients. The ``config_func`` function is to return the current FL
round number to client's ``fit()`` and ``evaluate()`` methods.

Tree-based Bagging Aggregation
++++++++++++++++++++++++++++++

You must be curious about how bagging aggregation works. Let's look into the details.

In file ``flwr.server.strategy.fedxgb_bagging.py``, we define ``FedXgbBagging``
inherited from ``flwr.server.strategy.FedAvg``. Then, we override the ``aggregate_fit``,
``aggregate_evaluate`` and ``evaluate`` methods as follows:

.. code-block:: python

    import json
    from logging import WARNING
    from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

    from flwr.common import EvaluateRes, FitRes, Parameters, Scalar
    from flwr.common.logger import log
    from flwr.server.client_proxy import ClientProxy

    from .fedavg import FedAvg


    class FedXgbBagging(FedAvg):
        """Configurable FedXgbBagging strategy implementation."""

        def __init__(
            self,
            evaluate_function: Optional[
                Callable[
                    [int, Parameters, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            **kwargs: Any,
        ):
            self.evaluate_function = evaluate_function
            self.global_model: Optional[bytes] = None
            super().__init__(**kwargs)

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate fit results using bagging."""
            if not results:
                return None, {}
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}

            # Aggregate all the client trees
            global_model = self.global_model
            for _, fit_res in results:
                update = fit_res.parameters.tensors
                for bst in update:
                    global_model = aggregate(global_model, bst)

            self.global_model = global_model

            return (
                Parameters(tensor_type="", tensors=[cast(bytes, global_model)]),
                {},
            )

        def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, Scalar]]:
            """Aggregate evaluation metrics using average."""
            if not results:
                return None, {}
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated = {}
            if self.evaluate_metrics_aggregation_fn:
                eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No evaluate_metrics_aggregation_fn provided")

            return 0, metrics_aggregated

        def evaluate(
            self, server_round: int, parameters: Parameters
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            """Evaluate model parameters using an evaluation function."""
            if self.evaluate_function is None:
                # No evaluation function provided
                return None
            eval_res = self.evaluate_function(server_round, parameters, {})
            if eval_res is None:
                return None
            loss, metrics = eval_res
            return loss, metrics

In ``aggregate_fit``, we sequentially aggregate the clients' XGBoost trees by calling
``aggregate()`` function:

.. code-block:: python

    def aggregate(
        bst_prev_org: Optional[bytes],
        bst_curr_org: bytes,
    ) -> bytes:
        """Conduct bagging aggregation for given trees."""
        if not bst_prev_org:
            return bst_curr_org

        # Get the tree numbers
        tree_num_prev, _ = _get_tree_nums(bst_prev_org)
        _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)

        bst_prev = json.loads(bytearray(bst_prev_org))
        bst_curr = json.loads(bytearray(bst_curr_org))

        bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ] = str(tree_num_prev + paral_tree_num_curr)
        iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
            "iteration_indptr"
        ]
        bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
            iteration_indptr[-1] + paral_tree_num_curr
        )

        # Aggregate new trees
        trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
        for tree_count in range(paral_tree_num_curr):
            trees_curr[tree_count]["id"] = tree_num_prev + tree_count
            bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
                trees_curr[tree_count]
            )
            bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

        bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")

        return bst_prev_bytes


    def _get_tree_nums(xgb_model_org: bytes) -> Tuple[int, int]:
        xgb_model = json.loads(bytearray(xgb_model_org))
        # Get the number of trees
        tree_num = int(
            xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                "num_trees"
            ]
        )
        # Get the number of parallel trees
        paral_tree_num = int(
            xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                "num_parallel_tree"
            ]
        )
        return tree_num, paral_tree_num

In this function, we first fetch the number of trees and the number of parallel trees
for the current and previous model by calling ``_get_tree_nums``. Then, the fetched
information will be aggregated. After that, the trees (containing model weights) are
aggregated to generate a new tree model.

After traversal of all clients' models, a new global model is generated, followed by
serialisation, and sending the global model back to each client.

Launch Federated XGBoost!
-------------------------

To run the project, do:

.. code-block:: shell

    # Run with default arguments
    $ flwr run .

With default arguments you will see an output like this one:

.. code-block:: shell

    Loading project configuration...
    Success
    INFO :      Starting Flower ServerApp, config: num_rounds=3, no round_timeout
    INFO :
    INFO :      [INIT]
    INFO :      Using initial global parameters provided by strategy
    INFO :      Starting evaluation of initial global parameters
    INFO :      Evaluation returned no results (`None`)
    INFO :
    INFO :      [ROUND 1]
    INFO :      configure_fit: strategy sampled 2 clients (out of 20)
    INFO :      aggregate_fit: received 2 results and 0 failures
    INFO :      configure_evaluate: strategy sampled 2 clients (out of 20)
    INFO :      aggregate_evaluate: received 2 results and 0 failures
    INFO :
    INFO :      [ROUND 2]
    INFO :      configure_fit: strategy sampled 2 clients (out of 20)
    INFO :      aggregate_fit: received 2 results and 0 failures
    INFO :      configure_evaluate: strategy sampled 2 clients (out of 20)
    INFO :      aggregate_evaluate: received 2 results and 0 failures
    INFO :
    INFO :      [ROUND 3]
    INFO :      configure_fit: strategy sampled 2 clients (out of 20)
    INFO :      aggregate_fit: received 2 results and 0 failures
    INFO :      configure_evaluate: strategy sampled 2 clients (out of 20)
    INFO :      aggregate_evaluate: received 2 results and 0 failures
    INFO :
    INFO :      [SUMMARY]
    INFO :      Run finished 3 round(s) in 145.42s
    INFO :              History (loss, distributed):
    INFO :                      round 1: 0
    INFO :                      round 2: 0
    INFO :                      round 3: 0
    INFO :              History (metrics, distributed, evaluate):
    INFO :              {'AUC': [(1, 0.7664), (2, 0.77595), (3, 0.7826)]}
    INFO :

Congratulations! You've successfully built and run your first federated XGBoost system.
The AUC values can be checked in ``History (metrics, distributed, evaluate)``. One can
see that the average AUC increases over FL rounds.

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config "num-server-rounds=5 params.eta=0.05"

.. note::

    Check the full `source code
    <https://github.com/adap/flower/blob/main/examples/xgboost-quickstart>`_ for this
    example in ``examples/xgboost-quickstart`` in the Flower GitHub repository.

Comprehensive Federated XGBoost
-------------------------------

Now that you know how federated XGBoost works with Flower, it's time to run some more
comprehensive experiments by customising the experimental settings. In the
xgboost-comprehensive example (`full code
<https://github.com/adap/flower/tree/main/examples/xgboost-comprehensive>`_), we provide
more options to define various experimental setups, including aggregation strategies,
data partitioning and centralised / distributed evaluation. Let's take a look!

Cyclic Training
~~~~~~~~~~~~~~~

In addition to bagging aggregation, we offer a cyclic training scheme, which performs FL
in a client-by-client fashion. Instead of aggregating multiple clients, there is only
one single client participating in the training per round in the cyclic training
scenario. The trained local XGBoost trees will be passed to the next client as an
initialised model for next round's boosting.

To do this, we first customise a ``ClientManager`` in ``server_app.py``:

.. code-block:: python

    class CyclicClientManager(SimpleClientManager):
        """Provides a cyclic client selection rule."""

        def sample(
            self,
            num_clients: int,
            min_num_clients: Optional[int] = None,
            criterion: Optional[Criterion] = None,
        ) -> List[ClientProxy]:
            """Sample a number of Flower ClientProxy instances."""

            # Block until at least num_clients are connected.
            if min_num_clients is None:
                min_num_clients = num_clients
            self.wait_for(min_num_clients)

            # Sample clients which meet the criterion
            available_cids = list(self.clients)
            if criterion is not None:
                available_cids = [
                    cid for cid in available_cids if criterion.select(self.clients[cid])
                ]

            if num_clients > len(available_cids):
                log(
                    INFO,
                    "Sampling failed: number of available clients"
                    " (%s) is less than number of requested clients (%s).",
                    len(available_cids),
                    num_clients,
                )
                return []

            # Return all available clients
            return [self.clients[cid] for cid in available_cids]

The customised ``ClientManager`` samples all available clients in each FL round based on
the order of connection to the server. Then, we define a new strategy ``FedXgbCyclic``
in ``flwr.server.strategy.fedxgb_cyclic.py``, in order to sequentially select only one
client in given round and pass the received model to the next client.

.. code-block:: python

    class FedXgbCyclic(FedAvg):
        """Configurable FedXgbCyclic strategy implementation."""

        # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
        def __init__(
            self,
            **kwargs: Any,
        ):
            self.global_model: Optional[bytes] = None
            super().__init__(**kwargs)

        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate fit results using bagging."""
            if not results:
                return None, {}
            # Do not aggregate if there are failures and failures are not accepted
            if not self.accept_failures and failures:
                return None, {}

            # Fetch the client model from last round as global model
            for _, fit_res in results:
                update = fit_res.parameters.tensors
                for bst in update:
                    self.global_model = bst

            return (
                Parameters(tensor_type="", tensors=[cast(bytes, self.global_model)]),
                {},
            )

Unlike the original ``FedAvg``, we don't perform aggregation here. Instead, we just make
a copy of the received client model as global model by overriding ``aggregate_fit``.

Also, the customised ``configure_fit`` and ``configure_evaluate`` methods ensure the
clients to be sequentially selected given FL round:

.. code-block:: python

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        # Sample the clients sequentially given server_round
        sampled_idx = (server_round - 1) % len(clients)
        sampled_clients = [clients[sampled_idx]]

        # Return client/config pairs
        return [(client, fit_ins) for client in sampled_clients]

Customised Data Partitioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``task.py``, we use the ``instantiate_fds`` function to instantiate Flower Datasets
and the data partitioner based on the given ``partitioner_type`` and ``num_partitions``.
Currently, we provide four supported partitioner type to simulate the
uniformity/non-uniformity in data quantity (uniform, linear, square, exponential).

.. code-block:: python

    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import (
        IidPartitioner,
        LinearPartitioner,
        SquarePartitioner,
        ExponentialPartitioner,
    )

    CORRELATION_TO_PARTITIONER = {
        "uniform": IidPartitioner,
        "linear": LinearPartitioner,
        "square": SquarePartitioner,
        "exponential": ExponentialPartitioner,
    }


    def instantiate_fds(partitioner_type, num_partitions):
        """Initialize FederatedDataset."""
        # Only initialize `FederatedDataset` once
        global fds
        if fds is None:
            partitioner = CORRELATION_TO_PARTITIONER[partitioner_type](
                num_partitions=num_partitions
            )
            fds = FederatedDataset(
                dataset="jxie/higgs",
                partitioners={"train": partitioner},
                preprocessor=resplit,
            )
        return fds

Customised Centralised / Distributed Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To facilitate centralised evaluation, we define a function in ``server_app.py``:

.. code-block:: python

    def get_evaluate_fn(test_data, params):
        """Return a function for centralised evaluation."""

        def evaluate_fn(
            server_round: int, parameters: Parameters, config: Dict[str, Scalar]
        ):
            # If at the first round, skip the evaluation
            if server_round == 0:
                return 0, {}
            else:
                bst = xgb.Booster(params=params)
                for para in parameters.tensors:
                    para_b = bytearray(para)

                # Load global model
                bst.load_model(para_b)
                # Run evaluation
                eval_results = bst.eval_set(
                    evals=[(test_data, "valid")],
                    iteration=bst.num_boosted_rounds() - 1,
                )
                auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

                return 0, {"AUC": auc}

        return evaluate_fn

This function returns an evaluation function, which instantiates a ``Booster`` object
and loads the global model weights to it. The evaluation is conducted by calling
``eval_set()`` method, and the tested AUC value is reported.

As for distributed evaluation on the clients, it's same as the quick-start example by
overriding the ``evaluate()`` method insides the ``XgbClient`` class in
``client_app.py``.

Arguments Explainer
~~~~~~~~~~~~~~~~~~~

We define all hyper-parameters under ``[tool.flwr.app.config]`` entry in
``pyproject.toml``:

.. code-block:: toml

    [tool.flwr.app.config]
    # ServerApp
    train-method = "bagging"  # Choose from [bagging, cyclic]
    num-server-rounds = 3
    fraction-fit = 1.0
    fraction-evaluate = 1.0
    centralised-eval = false

    # ClientApp
    partitioner-type = "uniform"  # Choose from [uniform, linear, square, exponential]
    test-fraction = 0.2
    seed = 42
    centralised-eval-client = false
    local-epochs = 1
    scaled-lr = false
    params.objective = "binary:logistic"
    params.eta = 0.1  # Learning rate
    params.max-depth = 8
    params.eval-metric = "auc"
    params.nthread = 16
    params.num-parallel-tree = 1
    params.subsample = 1
    params.tree-method = "hist"

On the server side, we allow user to specify training strategies / FL rounds /
participating clients / clients for evaluation, and evaluation fashion. Note that with
``centralised-eval = true``, the sever will do centralised evaluation and all
functionalities for client evaluation will be disabled.

On the client side, we can define various options for client data partitioning. Besides,
clients also have an option to conduct evaluation on centralised test set by setting
``centralised-eval = true``, as well as an option to perform scaled learning rate based
on the number of clients by setting ``scaled-lr = true``.

Example Commands
~~~~~~~~~~~~~~~~

To run bagging aggregation for 5 rounds evaluated on centralised test set:

.. code-block:: shell

    flwr run . --run-config "train-method='bagging' num-server-rounds=5 centralised-eval=true"

To run cyclic training with linear partitioner type evaluated on centralised test set:

.. code-block:: shell

    flwr run . --run-config "train-method='cyclic' partitioner-type='linear'
    centralised-eval-client=true"

.. note::

    The full `code
    <https://github.com/adap/flower/blob/main/examples/xgboost-comprehensive/>`_ for
    this comprehensive example can be found in ``examples/xgboost-comprehensive`` in the
    Flower GitHub repository.

Video Tutorial
--------------

.. note::

    The video shown below shows how to setup a XGBoost + Flower project using our
    previously recommended APIs. A new video tutorial will be released that shows the
    new APIs (as the content above does)

.. meta::
    :description: Check out this Federated Learning quickstart tutorial for using Flower with XGBoost to train classification models on trees.

.. youtube:: AY1vpXUpesc
    :width: 100%
