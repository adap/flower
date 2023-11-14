.. _quickstart-xgboost:


Quickstart XGBoost
==================

.. meta::
   :description: Check out this Federated Learning quickstart tutorial for using Flower with XGBoost to train classification models on trees.

Federated XGBoost
-------------
EXtreme Gradient Boosting (**XGBoost**) is a robust and efficient implementation of gradient-boosted decision tree (**GBDT**), that maximises the computational boundaries for boosted tree methods.
It's primarily designed to enhance both the performance and computational speed of machine learning models.
In XGBoost, trees are constructed concurrently, unlike the sequential approach taken by GBDT.

Often, for tabular data on medium-sized datasets with fewer than 10k training examples, XGBoost surpasses the results of deep learning techniques.

Why federated XGBoost?
~~~~~~~~
Indeed, as the demand for data privacy and decentralized learning grows, there's an increasing requirement to implement federated XGBoost systems for specialised applications, like survival analysis and financial fraud detection.

Federated learning ensures that raw data remains on the local device, making it an attractive approach for sensitive domains where data security and privacy are paramount.
Given the robustness and efficiency of XGBoost, combining it with federated learning offers a promising solution for these specific challenges.

In this tutorial we will learn how to train a federated XGBoost model on HIGGS dataset using Flower and :code:`xgboost` package.
Our example consists of two *clients* and one *server*, where the local trees are aggregated based on bagging criterion on the server.

Environment Setup
-------------
First of all, it is recommended to create a virtual environment and run everything within a `virtualenv <https://flower.dev/docs/recommended-env-setup.html>`_.

We first need to install Flower and Flower Datasets. You can do this by running :

.. code-block:: shell

  $ pip install flwr flwr-datasets

Since we want to use :code:`xgboost` package to build up XGBoost trees, let's go ahead and install :code:`xgboost`:

.. code-block:: shell

  $ pip install xgboost


Flower Client
-------------

*Clients* are responsible for generating individual weight-updates for the model based on their local datasets.
Now that we have all our dependencies installed, let's run a simple distributed training with two clients and one server.

In a file called :code:`client.py`, import xgboost, Flower and related functions from :code:`dataset.py`:

.. code-block:: python

    import xgboost as xgb

    import flwr as fl
    from flwr_datasets import FederatedDataset
    from flwr.common import (
        Code,
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
        GetParametersIns,
        GetParametersRes,
        Parameters,
        Status,
    )

    from dataset import instantiate_partitioner, train_test_split, transform_dataset_to_dmatrix

Dataset partition and hyper-parameter selection
~~~~~~~~
Prior to local training, we require loading the HIGGS dataset from Flower Datasets and conduct data partitioning for FL.
Currently, we provide four options to split the dataset to simulate the non-uniformity in data quantity (uniform, linear, square, exponential) based on the number of samples.

The implementation details can be found in :code:`dataset.py` from `full code example <https://github.com/adap/flower/tree/main/examples/quickstart-xgboost>`_.

.. code-block:: python

    # Load (HIGGS) dataset and conduct partitioning
    num_partitions = 20
    # partitioner type is chosen from ["uniform", "linear", "square", "exponential"]
    partitioner_type = "uniform"

    # instantiate partitioner
    partitioner = instantiate_partitioner(partitioner_type=partitioner_type, num_partitions=num_partitions)
    fds = FederatedDataset(dataset="jxie/higgs", partitioners={"train": partitioner})

    # let's use the first partition as an example
    partition_id = 0
    partition = fds.load_partition(idx=partition_id, split="train")
    partition.set_format("numpy")

    # train/test splitting and data re-formatting
    SEED = 42
    test_fraction = 0.2
    train_data, valid_data, num_train, num_val = train_test_split(partition, test_fraction=test_fraction, seed=SEED)

    # reformat data to DMatrix for xgboost
    train_dmatrix = transform_dataset_to_dmatrix(train_data)
    valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

Then, we define the hyper-parameters used for XGBoost training.

.. code-block:: python

    num_local_round = 1
    params = {
        "objective": "binary:logistic",
        "eta": 0.1,  # lr
        "max_depth": 8,
        "eval_metric": "auc",
        "nthread": 16,
        "num_parallel_tree": 1,
        "subsample": 1,
        "tree_method": "hist",
    }

The :code:`num_local_round` represents the number of iterations for local tree boost.
We use CPU for the training in default.
One can shift it to GPU by setting :code:`tree_method` to :code:`gpu_hist`.
We use AUC as evaluation metric.

Flower client definition for XGBoost
~~~~~~~~
After loading the dataset we define the Flower client.
We follow the general rule to define :code:`XGBoostClient` class inherited from :code:`fl.client.Client`.

.. code-block:: python

    class XGBoostClient(fl.client.Client):
        def __init__(self):
            self.bst = None

The :code:`self.bst` is used to keep the Booster objects that remain consistent across rounds,
allowing them to store predictions from trees integrated in earlier rounds and maintain other essential data structures for training.

Then, we override :code:`get_parameters`, :code:`fit` and :code:`evaluate` methods insides :code:`XGBoostClient` class as follows.

.. code-block:: python

        def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
            _ = (self, ins)
            return GetParametersRes(
                status=Status(
                    code=Code.OK,
                    message="OK",
                ),
                parameters=Parameters(tensor_type="", tensors=[]),
            )

Unlike neural network training, XGBoost trees are not started from a specified random weights.
In this case, we do not use :code:`get_parameters` and :code:`set_parameters` to initialise model parameters for XGBoost.
As a result, let's return an empty tensor in :code:`get_parameters` when it is called by the server at the first round.

.. code-block:: python

        def fit(self, ins: FitIns) -> FitRes:
            if not self.bst:
                # first round local training
                print("Start training at round 1")
                bst = xgb.train(
                    params,
                    train_dmatrix,
                    num_boost_round=num_local_round,
                    evals=[(valid_dmatrix, "validate"), (train_dmatrix, "train")],
                )
                self.config = bst.save_config()
                self.bst = bst
            else:
                print("load global model")
                for item in ins.parameters.tensors:
                    global_model = bytearray(item)

                # load global model into booster
                self.bst.load_model(global_model)
                self.bst.load_config(self.config)

                bst = self.local_boost()

            local_model = bst.save_raw("json")
            local_model_bytes = bytes(local_model)

            return FitRes(
                status=Status(
                    code=Code.OK,
                    message="OK",
                ),
                parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
                num_examples=num_train,
                metrics={},
            )

In :code:`fit`, at the first round, we call :code:`xgb.train()` to build up the first set of trees.
the returned Booster object and config are stored in :code:`self.bst` and :code:`self.config`, respectively.
From the second round, we load the global model sent from server to :code:`self.bst`,
and then update model weights on local training data with function :code:`local_boost` as follows.

.. code-block:: python

    def local_boost(self):
        # update trees based on local training data.
        for i in range(num_local_round):
            self.bst.update(train_dmatrix, self.bst.num_boosted_rounds())

        # extract the last N=num_local_round trees as new local model
        bst = self.bst[
            self.bst.num_boosted_rounds()
            - num_local_round : self.bst.num_boosted_rounds()
        ]
        return bst

Given :code:`num_local_round`, we update trees by calling :code:`self.bst.update` method.
After training, the last :code:`N=num_local_round` trees will be extracted as the new local model.

.. code-block:: python

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        eval_results = self.bst.eval_set(
            evals=[(train_dmatrix, "train"), (valid_dmatrix, "valid")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[2].split(":")[1]), 4)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=num_val,
            metrics={"AUC": auc},
        )

In :code:`evaluate`, we call :code:`self.bst.eval_set` function to conduct evaluation on valid set.
The AUC value will be returned.

Now, we can create an instance of our class :code:`XGBoostClient` and add one line
to actually run this client:

.. code-block:: python

    fl.client.start_client(server_address="[::]:8080", client=FlowerClient().to_client())

That's it for the client. We only have to implement :code:`Client`and call :code:`fl.client.start_client()`.
The string :code:`"[::]:8080"` tells the client which server to connect to.
In our case we can run the server and the client on the same machine, therefore we use
:code:`"[::]:8080"`. If we run a truly federated workload with the server and
clients running on different machines, all that needs to change is the
:code:`server_address` we point the client at.


Flower Server
-------------

These updates are then sent to the *server* which will aggregate them to produce a better model.
Finally, the *server* sends this improved version of the model back to each *client* to finish a complete FL round.

In a file named :code:`server.py`, import Flower and XGbBagging from :code:`strategy.py`.

We first define a strategy for XGBoost bagging aggregation.

.. code-block:: python

    # Define strategy
    strategy = XGbBagging(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        fraction_evaluate=1.0,
        min_evaluate_clients=2,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )

    def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    auc_aggregated = sum([metrics["AUC"] for _, metrics in eval_metrics]) / len(
        eval_metrics
    )
    metrics_aggregated = {"AUC": auc_aggregated}
    return metrics_aggregated

We use two clients for this example.
A :code:`evaluate_metrics_aggregation` function is defined to collect and average AUC values from clients.

Then, we start the server:

.. code-block:: python

    # Start Flower server
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

Tree-based bagging aggregation
~~~~~~~~
You must be curious about how bagging aggregation works.
Let's look into the details.

In file :code:`strategy.py`, we define :code:`XGbBagging` inherited from :code:`fl.server.strategy.FedAvg`.
Then, we override the :code:`aggregate_fit` method.

.. code-block:: python

    from typing import Dict, List, Optional, Tuple, Union
    import flwr as fl
    import json

    from flwr.common import (
        FitRes,
        Parameters,
        Scalar,
    )
    from flwr.server.client_proxy import ClientProxy


    class XGbBagging(fl.server.strategy.FedAvg):
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
            global_model = None
            for _, fit_res in results:
                update = fit_res.parameters.tensors
                for item in update:
                    global_model = aggregate(global_model, json.loads(bytearray(item)))

            weights_avg = json.dumps(global_model)

            return (
                Parameters(
                    tensor_type="", tensors=[bytes(weights_avg, "utf-8")]
                ),
                {},
            )

We sequentially aggregate the clients' XGBoost trees by calling :code:`aggregate()` function:

.. code-block:: python

    def aggregate(bst_prev, bst_curr):
        if not bst_prev:
            return bst_curr
        else:
            # get the tree numbers
            tree_num_prev, paral_tree_num_prev = _get_tree_nums(bst_prev)
            tree_num_curr, paral_tree_num_curr = _get_tree_nums(bst_curr)

            bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                "num_trees"
            ] = str(tree_num_prev + paral_tree_num_curr)
            iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
                "iteration_indptr"
            ]
            bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
                iteration_indptr[-1] + 1 * paral_tree_num_curr
            )

            # aggregate new trees
            trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
            for tree_count in range(paral_tree_num_curr):
                trees_curr[tree_count]["id"] = tree_num_prev + tree_count
                bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
                    trees_curr[tree_count]
                )
                bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)
            return bst_prev

    def _get_tree_nums(xgb_model):
        # get the number of trees
        tree_num = int(
            xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                "num_trees"
            ]
        )
        # get the number of parallel trees
        paral_tree_num = int(
            xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
                "num_parallel_tree"
            ]
        )
        return tree_num, paral_tree_num

In this function, we first fetch the number of trees and the number of parallel trees for the current and previous model
by calling :code:`_get_tree_nums`.
Then, the fetched information will be aggregated.
After that, the trees (containing model weights) are aggregated to generate a new tree model.

After traversal of all clients' models, a new global model is generated,
followed by the serialisation, and sending back each client.


Launch federated XGBoost!
---------------------------

With both client and server ready, we can now run everything and see federated
learning in action. FL systems usually have a server and multiple clients. We
therefore have to start the server first:

.. code-block:: shell

    $ python server.py

Once the server is running we can start the clients in different terminals.
Open a new terminal and start the first client:

.. code-block:: shell

    $ python client.py

Open another terminal and start the second client:

.. code-block:: shell

    $ python client.py

Each client will have its own dataset.
You should now see how the training does in the very first terminal (the one that started the server):

.. code-block:: shell

    INFO flwr 2023-11-06 10:50:38,755 | app.py:162 | Starting Flower server, config: ServerConfig(num_rounds=5, round_timeout=None)
    INFO flwr 2023-11-06 10:50:39,293 | app.py:175 | Flower ECE: gRPC server running (5 rounds), SSL is disabled
    INFO flwr 2023-11-06 10:50:39,294 | server.py:89 | Initializing global parameters
    INFO flwr 2023-11-06 10:50:39,294 | server.py:276 | Requesting initial parameters from one random client
    INFO flwr 2023-11-06 10:52:16,328 | server.py:280 | Received initial parameters from one random client
    INFO flwr 2023-11-06 10:52:16,329 | server.py:91 | Evaluating initial parameters
    INFO flwr 2023-11-06 10:52:16,329 | server.py:104 | FL starting
    DEBUG flwr 2023-11-06 10:52:16,331 | server.py:222 | fit_round 1: strategy sampled 2 clients (out of 2)
    DEBUG flwr 2023-11-06 10:52:16,935 | server.py:236 | fit_round 1 received 2 results and 0 failures
    DEBUG flwr 2023-11-06 10:52:16,943 | server.py:173 | evaluate_round 1: strategy sampled 2 clients (out of 2)
    DEBUG flwr 2023-11-06 10:52:17,023 | server.py:187 | evaluate_round 1 received 2 results and 0 failures
    DEBUG flwr 2023-11-06 10:52:17,023 | server.py:222 | fit_round 2: strategy sampled 2 clients (out of 2)
    DEBUG flwr 2023-11-06 10:52:17,125 | server.py:236 | fit_round 2 received 2 results and 0 failures
    DEBUG flwr 2023-11-06 10:52:17,132 | server.py:173 | evaluate_round 2: strategy sampled 2 clients (out of 2)
    DEBUG flwr 2023-11-06 10:52:17,195 | server.py:187 | evaluate_round 2 received 2 results and 0 failures
    DEBUG flwr 2023-11-06 10:52:17,195 | server.py:222 | fit_round 3: strategy sampled 2 clients (out of 2)
    DEBUG flwr 2023-11-06 10:52:17,304 | server.py:236 | fit_round 3 received 2 results and 0 failures
    DEBUG flwr 2023-11-06 10:52:17,310 | server.py:173 | evaluate_round 3: strategy sampled 2 clients (out of 2)
    DEBUG flwr 2023-11-06 10:52:17,383 | server.py:187 | evaluate_round 3 received 2 results and 0 failures
    DEBUG flwr 2023-11-06 10:52:17,383 | server.py:222 | fit_round 4: strategy sampled 2 clients (out of 2)
    DEBUG flwr 2023-11-06 10:52:17,496 | server.py:236 | fit_round 4 received 2 results and 0 failures
    DEBUG flwr 2023-11-06 10:52:17,503 | server.py:173 | evaluate_round 4: strategy sampled 2 clients (out of 2)
    DEBUG flwr 2023-11-06 10:52:17,570 | server.py:187 | evaluate_round 4 received 2 results and 0 failures
    DEBUG flwr 2023-11-06 10:52:17,570 | server.py:222 | fit_round 5: strategy sampled 2 clients (out of 2)
    DEBUG flwr 2023-11-06 10:52:17,683 | server.py:236 | fit_round 5 received 2 results and 0 failures
    DEBUG flwr 2023-11-06 10:52:17,689 | server.py:173 | evaluate_round 5: strategy sampled 2 clients (out of 2)
    DEBUG flwr 2023-11-06 10:52:17,756 | server.py:187 | evaluate_round 5 received 2 results and 0 failures
    INFO flwr 2023-11-06 10:52:17,756 | server.py:153 | FL finished in 1.4270430290000036
    INFO flwr 2023-11-06 10:52:17,759 | app.py:225 | app_fit: losses_distributed [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    INFO flwr 2023-11-06 10:52:17,759 | app.py:226 | app_fit: metrics_distributed_fit {}
    INFO flwr 2023-11-06 10:52:17,759 | app.py:227 | app_fit: metrics_distributed {'AUC': [(1, 0.7587), (2, 0.7681), (3, 0.771), (4, 0.7705), (5, 0.7712)]}
    INFO flwr 2023-11-06 10:52:17,759 | app.py:228 | app_fit: losses_centralized []
    INFO flwr 2023-11-06 10:52:17,759 | app.py:229 | app_fit: metrics_centralized {}

Congratulations!
You've successfully built and run your first federated XGBoost system.
The AUC values can be checked in :code:`metrics_distributed`.
One can see that the average AUC increases over FL rounds.

The full `source code <https://github.com/adap/flower/blob/main/examples/quickstart-xgboost/>`_ for this example can be found in :code:`examples/quickstart-xgboost`.
