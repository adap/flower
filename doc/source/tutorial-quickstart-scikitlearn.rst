.. _quickstart-scikitlearn:


Quickstart scikit-learn
=======================

.. meta::
   :description: Check out this Federated Learning quickstart tutorial for using Flower with scikit-learn to train a linear regression model.

In this tutorial, we will learn how to train a :code:`Logistic Regression` model on MNIST using Flower and scikit-learn.

First of all, it is recommended to create a virtual environment and run everything within a :doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

Our example consists of one *server* and two *clients* all having the same model.

*Clients* are responsible for generating individual model parameter updates for the model based on their local datasets.
These updates are then sent to the *server* which will aggregate them to produce an updated global model. Finally, the *server* sends this improved version of the model back to each *client*.
A complete cycle of parameters updates is called a *round*.

Now that we have a rough idea of what is going on, let's get started. We first need to install Flower and Flower Datasets:

.. code-block:: shell

  $ pip install flwr flwr-datasets

Since we want to use scikit-learn, let's go ahead and install it:

.. code-block:: shell

  $ pip install scikit-learn


Flower Client
-------------

Now that we have all our dependencies installed, let's run a simple distributed training with two clients and one server.
However, before setting up the client and server, we will define all functionalities that we need for our federated learning setup within :code:`utils.py`. The :code:`utils.py` contains different functions defining all the machine learning basics:

* :code:`get_model_parameters()`
    * Returns the parameters of a :code:`sklearn` LogisticRegression model
* :code:`set_model_params()`
    * Sets the parameters of a :code:`sklearn` LogisticRegression model
* :code:`set_initial_params()`
    * Initializes the model parameters that the Flower server will ask for

Please check out :code:`utils.py` `here <https://github.com/adap/flower/blob/main/examples/sklearn-logreg-mnist/utils.py>`_ for more details.
The pre-defined functions are used in the :code:`client.py` and imported.

Now, in a file called :code:`client.py`, we import several packages such as Flower, Flower Datasets, and scikit-learn:

.. code-block:: python

  import warnings
  
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import log_loss
  
  import flwr as fl
  import utils
  from flwr_datasets import FederatedDataset

Prior to local training, we need to load the MNIST dataset, a popular image classification dataset of handwritten digits for machine learning, and partition the dataset for FL. This can be conveniently achieved using `Flower Datasets <https://flower.ai/docs/datasets>`_.
The :code:`FederatedDataset.load_partition()` method loads the partitioned training set for each partition ID set in the `partition_id` variable. We assign an integer to each `partition_id` for each client in our federated learning example, starting from 0.

.. code-block:: python

    fds = FederatedDataset(dataset="mnist", partitioners={"train": N_CLIENTS})
    
    dataset = fds.load_partition(partition_id, "train").with_format("numpy")
    X, y = dataset["image"].reshape((len(dataset), -1)), dataset["label"]
    
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]


Next, we define the logistic regression model and initialize it with :code:`utils.set_initial_params()`.

.. code-block:: python

    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    utils.set_initial_params(model)

The Flower server interacts with clients through an interface called
:code:`Client`. When the server selects a particular client for training, it
sends training instructions over the network. The client receives those
instructions and calls one of the :code:`Client` methods to run your code
(i.e., to fit the logistic regression we defined earlier).

Flower provides a convenience class called :code:`NumPyClient` which makes it
easier to implement the :code:`Client` interface when your workload uses scikit-learn.
Implementing :code:`NumPyClient` usually means defining the following methods
(:code:`set_parameters` is optional though):

#. :code:`get_parameters`
    * return the model weight as a list of NumPy ndarrays
#. :code:`set_parameters` (optional)
    * update the local model weights with the parameters received from the server
    * is directly imported with :code:`utils.set_model_params()`
#. :code:`fit`
    * set the local model weights
    * train the local model
    * receive the updated local model weights
#. :code:`evaluate`
    * test the local model

The :code:`NumPyClient` interface defines the three methods which can be implemented in the following way:

.. code-block:: python

    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):
            utils.set_model_params(model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}


Next, we create a client function that returns instances of :code:`MnistClient` on-demand when called:

.. code-block:: python

    def client_fn(cid: str):
        return CifarClient().to_client()

Finally, we create a :code:`ClientApp()` object that uses this client function:

.. code-block:: python

    app = ClientApp(client_fn=client_fn)

That's it for the client. We only have to implement :code:`Client` or
:code:`NumPyClient`, create a :code:`ClientApp`, and pass the client function to it. If we implement a client of type :code:`NumPyClient` we'll need to first call its :code:`to_client()` method.


Flower Server
-------------

The following Flower server is a little bit more advanced and returns an evaluation function for the server-side evaluation.
First, in a file named :code:`server.py`, we import all required libraries such as Flower, Flower Datasets, and scikit-learn:

.. code-block:: python

    import flwr as fl
    import utils
    from flwr.common import NDArrays, Scalar
    from sklearn.metrics import log_loss
    from sklearn.linear_model import LogisticRegression
    from typing import Dict
    
    from flwr_datasets import FederatedDataset

The number of federated learning rounds is set in :code:`fit_round()` and the evaluation is defined in :code:`get_evaluate_fn()`.
The evaluation function is called after each federated learning round and gives you information about loss and accuracy.
Note that we also make use of Flower Datasets here to load the test split of the MNIST dataset for server-side evaluation.

.. code-block:: python

    def fit_round(server_round: int) -> Dict:
        """Send round number to client."""
        return {"server_round": server_round}


    def get_evaluate_fn(model: LogisticRegression):
        """Return an evaluation function for server-side evaluation."""

        fds = FederatedDataset(dataset="mnist", partitioners={"train": 10})
        dataset = fds.load_split("test").with_format("numpy")
        X_test, y_test = dataset["image"].reshape((len(dataset), -1)), dataset["label"]

        def evaluate(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, {"accuracy": accuracy}

        return evaluate

We set the `ServerConfig` with `num_rounds=3` to train the `Logistic Regression` model for 3 rounds.

.. code-block:: python

    config = fl.server.ServerConfig(num_rounds=3)

Next, we initialize the server-side parameters for :code:`LogisticRegression()` using :code:`utils.set_initial_params()` and set the aggregation strategy :code:`fl.server.strategy:FedAvg()`. The strategy is the default one, federated averaging (or FedAvg), with two clients and evaluation after each federated learning round. In the last line, we create a `ServerApp` using the config and strategy.

.. code-block:: python

    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    app = ServerApp(
        config=config,
        strategy=strategy,
    )


Train the model, federated!
---------------------------

With both client and server ready, we can now run everything and see federated
learning in action. First, we run the :code:`flower-superlink` command in one terminal to start the infrastructure. This step only needs to be run once.

.. admonition:: Note
    :class: note

    In this example, the :code:`--insecure` command line argument starts Flower without HTTPS and is only used for prototyping. To run with HTTPS, we instead use the arguments :code:`--ssl-ca-certfile`, :code:`--ssl-certfile`, and :code:`--ssl-keyfile` and pass the paths to the certificates. Please refer to `Flower CLI reference <ref-api-cli.html#flower-superlink>`_ for implementation details.

.. code-block:: shell

    $ flower-superlink --insecure

FL systems usually have a server and multiple clients. We therefore need to start multiple `SuperNodes`, one for each client, respectively. First, we open a new terminal and start the first `SuperNode` using the :code:`flower-client-app` command.

.. code-block:: shell

    $ flower-client-app client:app --insecure

In the above, we launch the :code:`app` object in the :code:`client.py` module.
Open another terminal and start the second `SuperNode`:

.. code-block:: shell

    $ flower-client-app client:app --insecure

Finally, in another terminal window, we run the `ServerApp`. This starts the actual training run:

.. code-block:: shell

    $ flower-server-app server:app --insecure

We should now see how the training does in the last terminal (the one that started the :code:`ServerApp`):

.. code-block:: shell

    WARNING :   Option `--insecure` was set. Starting insecure HTTP client connected to 0.0.0.0:9091.
    INFO :      Starting Flower ServerApp, config: num_rounds=3, no round_timeout
    INFO :
    INFO :      [INIT]
    INFO :      Requesting initial parameters from one random client
    INFO :      Received initial parameters from one random client
    INFO :      Evaluating initial global parameters
    INFO :      initial parameters (loss, other metrics): 2.3025850929940455, {'accuracy': 0.098}
    INFO :
    INFO :      [ROUND 1]
    INFO :      configure_fit: strategy sampled 2 clients (out of 2)
    INFO :      aggregate_fit: received 2 results and 0 failures
    WARNING :   No fit_metrics_aggregation_fn provided
    INFO :      fit progress: (1, 1.4140462685358515, {'accuracy': 0.6752}, 4.125828707939945)
    INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
    INFO :      aggregate_evaluate: received 2 results and 0 failures
    WARNING :   No evaluate_metrics_aggregation_fn provided
    INFO :
    INFO :      [ROUND 2]
    INFO :      configure_fit: strategy sampled 2 clients (out of 2)
    INFO :      aggregate_fit: received 2 results and 0 failures
    INFO :      fit progress: (2, 0.7323360226502517, {'accuracy': 0.7706}, 10.23554670799058)
    INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
    INFO :      aggregate_evaluate: received 2 results and 0 failures
    INFO :
    INFO :      [ROUND 3]
    INFO :      configure_fit: strategy sampled 2 clients (out of 2)
    INFO :      aggregate_fit: received 2 results and 0 failures
    INFO :      fit progress: (3, 0.5672925184955843, {'accuracy': 0.8202}, 16.32356683292892)
    INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
    INFO :      aggregate_evaluate: received 2 results and 0 failures
    INFO :
    INFO :      [SUMMARY]
    INFO :      Run finished 3 rounds in 19.34s
    INFO :      History (loss, distributed):
    INFO :          ('\tround 1: 1.3345516917477076\n'
    INFO :           '\tround 2: 0.6896191223254897\n'
    INFO :           '\tround 3: 0.5527833946909323\n')History (loss, centralized):
    INFO :          ('\tround 0: 2.3025850929940455\n'
    INFO :           '\tround 1: 1.4140462685358515\n'
    INFO :           '\tround 2: 0.7323360226502517\n'
    INFO :           '\tround 3: 0.5672925184955843\n')History (metrics, centralized):
    INFO :          {'accuracy': [(0, 0.098), (1, 0.6752), (2, 0.7706), (3, 0.8202)]}

Congratulations!
You've successfully built and run your first federated learning system.
The full source code for this example can be found in |quickstart_sklearn_link|_.

.. |quickstart_sklearn_link| replace:: :code:`examples/sklearn-logreg-mnist` 
.. _quickstart_sklearn_link: https://github.com/adap/flower/tree/main/examples/sklearn-logreg-mnist