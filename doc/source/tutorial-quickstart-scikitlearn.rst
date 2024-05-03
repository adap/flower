.. _quickstart-scikitlearn:


Quickstart scikit-learn
=======================

.. meta::
   :description: Check out this Federated Learning quickstart tutorial for using Flower with scikit-learn to train a linear regression model.

In this tutorial, we will learn how to train a :code:`Logistic Regression` model on MNIST using Flower and scikit-learn.

It is recommended to create a virtual environment and run everything within this :doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

Our example consists of one *server* and two *clients* all having the same model.

*Clients* are responsible for generating individual model parameter updates for the model based on their local datasets.
These updates are then sent to the *server* which will aggregate them to produce an updated global model. Finally, the *server* sends this improved version of the model back to each *client*.
A complete cycle of parameters updates is called a *round*.

Now that we have a rough idea of what is going on, let's get started. We first need to install Flower. You can do this by running:

.. code-block:: shell

  $ pip install flwr

Since we want to use scikit-learn, let's go ahead and install it:

.. code-block:: shell

  $ pip install scikit-learn

Or simply install all dependencies using Poetry:

.. code-block:: shell

  $ poetry install


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
The pre-defined functions are used in the :code:`client.py` and imported. The :code:`client.py` also requires to import several packages such as Flower and scikit-learn:

.. code-block:: python

  import argparse
  import warnings
  
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import log_loss
  
  import flwr as fl
  import utils
  from flwr_datasets import FederatedDataset

Prior to local training, we need to load the MNIST dataset, a popular image classification dataset of handwritten digits for machine learning, and partition the dataset for FL. This can be conveniently achieved using `Flower Datasets <https://flower.ai/docs/datasets>`_.
The :code:`FederatedDataset.load_partition()` method loads the partitioned training set for each partition ID defined in the :code:`--partition-id` argument.

.. code-block:: python

    if __name__ == "__main__":
        N_CLIENTS = 10
    
        parser = argparse.ArgumentParser(description="Flower")
        parser.add_argument(
            "--partition-id",
            type=int,
            choices=range(0, N_CLIENTS),
            required=True,
            help="Specifies the artificial data partition",
        )
        args = parser.parse_args()
        partition_id = args.partition_id
    
        fds = FederatedDataset(dataset="mnist", partitioners={"train": N_CLIENTS})
    
        dataset = fds.load_partition(partition_id, "train").with_format("numpy")
        X, y = dataset["image"].reshape((len(dataset), -1)), dataset["label"]
        
        X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
        y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]


Next, the logistic regression model is defined and initialized with :code:`utils.set_initial_params()`.

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

The methods can be implemented in the following way:

.. code-block:: python

    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, len(X_test), {"accuracy": accuracy}


We can now create an instance of our class :code:`MnistClient` and add one line
to actually run this client:

.. code-block:: python

    fl.client.start_client("0.0.0.0:8080", client=MnistClient().to_client())

That's it for the client. We only have to implement :code:`Client` or
:code:`NumPyClient` and call :code:`fl.client.start_client()`. If you implement a client of type :code:`NumPyClient` you'll need to first call its :code:`to_client()` method. The string :code:`"0.0.0.0:8080"` tells the client which server to connect to. In our case we can run the server and the client on the same machine, therefore we use
:code:`"0.0.0.0:8080"`. If we run a truly federated workload with the server and
clients running on different machines, all that needs to change is the
:code:`server_address` we pass to the client.

Flower Server
-------------

The following Flower server is a little bit more advanced and returns an evaluation function for the server-side evaluation.
First, we import again all required libraries such as Flower and scikit-learn.

:code:`server.py`, import Flower and start the server:

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

The :code:`main` contains the server-side parameter initialization :code:`utils.set_initial_params()` as well as the aggregation strategy :code:`fl.server.strategy:FedAvg()`. The strategy is the default one, federated averaging (or FedAvg), with two clients and evaluation after each federated learning round. The server can be started with the command :code:`fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=3))`.

.. code-block:: python

    # Start Flower server for three rounds of federated learning
    if __name__ == "__main__":
        model = LogisticRegression()
        utils.set_initial_params(model)
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=2,
            evaluate_fn=get_evaluate_fn(model),
            on_fit_config_fn=fit_round,
        )
        fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=3))


Train the model, federated!
---------------------------

With both client and server ready, we can now run everything and see federated
learning in action. Federated learning systems usually have a server and multiple clients. We, therefore, have to start the server first:

.. code-block:: shell

    $ python3 server.py

Once the server is running we can start the clients in different terminals.
Open a new terminal and start the first client:

.. code-block:: shell

    $ python3 client.py

Open another terminal and start the second client:

.. code-block:: shell

    $ python3 client.py

Each client will have its own dataset.
You should now see how the training does in the very first terminal (the one that started the server):

.. code-block:: shell

    INFO flower 2022-01-13 13:43:14,859 | app.py:73 | Flower server running (insecure, 3 rounds)
    INFO flower 2022-01-13 13:43:14,859 | server.py:118 | Getting initial parameters
    INFO flower 2022-01-13 13:43:17,903 | server.py:306 | Received initial parameters from one random client
    INFO flower 2022-01-13 13:43:17,903 | server.py:120 | Evaluating initial parameters
    INFO flower 2022-01-13 13:43:17,992 | server.py:123 | initial parameters (loss, other metrics): 2.3025850929940455, {'accuracy': 0.098}
    INFO flower 2022-01-13 13:43:17,992 | server.py:133 | FL starting
    DEBUG flower 2022-01-13 13:43:19,814 | server.py:251 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2022-01-13 13:43:20,046 | server.py:260 | fit_round received 2 results and 0 failures
    INFO flower 2022-01-13 13:43:20,220 | server.py:148 | fit progress: (1, 1.3365667871792377, {'accuracy': 0.6605}, 2.227397900000142)
    INFO flower 2022-01-13 13:43:20,220 | server.py:199 | evaluate_round: no clients selected, cancel
    DEBUG flower 2022-01-13 13:43:20,220 | server.py:251 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2022-01-13 13:43:20,456 | server.py:260 | fit_round received 2 results and 0 failures
    INFO flower 2022-01-13 13:43:20,603 | server.py:148 | fit progress: (2, 0.721620492535375, {'accuracy': 0.7796}, 2.6108531999998377)
    INFO flower 2022-01-13 13:43:20,603 | server.py:199 | evaluate_round: no clients selected, cancel
    DEBUG flower 2022-01-13 13:43:20,603 | server.py:251 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2022-01-13 13:43:20,837 | server.py:260 | fit_round received 2 results and 0 failures
    INFO flower 2022-01-13 13:43:20,967 | server.py:148 | fit progress: (3, 0.5843629244915138, {'accuracy': 0.8217}, 2.9750180000010005)
    INFO flower 2022-01-13 13:43:20,968 | server.py:199 | evaluate_round: no clients selected, cancel
    INFO flower 2022-01-13 13:43:20,968 | server.py:172 | FL finished in 2.975252800000817
    INFO flower 2022-01-13 13:43:20,968 | app.py:109 | app_fit: losses_distributed []
    INFO flower 2022-01-13 13:43:20,968 | app.py:110 | app_fit: metrics_distributed {}
    INFO flower 2022-01-13 13:43:20,968 | app.py:111 | app_fit: losses_centralized [(0, 2.3025850929940455), (1, 1.3365667871792377), (2, 0.721620492535375), (3, 0.5843629244915138)]
    INFO flower 2022-01-13 13:43:20,968 | app.py:112 | app_fit: metrics_centralized {'accuracy': [(0, 0.098), (1, 0.6605), (2, 0.7796), (3, 0.8217)]}
    DEBUG flower 2022-01-13 13:43:20,968 | server.py:201 | evaluate_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2022-01-13 13:43:21,232 | server.py:210 | evaluate_round received 2 results and 0 failures
    INFO flower 2022-01-13 13:43:21,232 | app.py:121 | app_evaluate: federated loss: 0.5843629240989685
    INFO flower 2022-01-13 13:43:21,232 | app.py:122 | app_evaluate: results [('ipv4:127.0.0.1:53980', EvaluateRes(loss=0.5843629240989685, num_examples=10000, accuracy=0.0, metrics={'accuracy': 0.8217})), ('ipv4:127.0.0.1:53982', EvaluateRes(loss=0.5843629240989685, num_examples=10000, accuracy=0.0, metrics={'accuracy': 0.8217}))]
    INFO flower 2022-01-13 13:43:21,232 | app.py:127 | app_evaluate: failures []

Congratulations!
You've successfully built and run your first federated learning system.
The full `source code <https://github.com/adap/flower/tree/main/examples/sklearn-logreg-mnist>`_ for this example can be found in :code:`examples/sklearn-logreg-mnist`.
