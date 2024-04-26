.. _quickstart-tensorflow:


Quickstart TensorFlow
=====================

.. meta::
   :description: Check out this Federated Learning quickstart tutorial for using Flower with TensorFlow to train a MobileNetV2 model on CIFAR-10.

..  youtube:: FGTc2TQq7VM
   :width: 100%

.. admonition:: Disclaimer
    :class: important

    The Quickstart TensorFlow video uses slightly different Flower commands than this tutorial. Please follow the :doc:`Upgrade to Flower Next <how-to-upgrade-to-flower-next>` guide to convert commands shown in the video.

In this tutorial we will learn how to train a MobileNetV2 model on CIFAR10 using the Flower framework and TensorFlow.

First of all, it is recommended to create a virtual environment and run everything within a :doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

Before Flower can be imported we have to install it:

.. code-block:: shell

  $ pip install flwr flwr-datasets[vision]

Since we want to use the Keras API of TensorFlow (TF), we have to install TF as well: 

.. code-block:: shell

  $ pip install tensorflow


Flower Client
-------------

Next, in a file called :code:`client.py`, import Flower, Flower Datasets, and TensorFlow:

.. code-block:: python

    import flwr as fl
    import tensorflow as tf
    from flwr_datasets import FederatedDataset

We use `Flower Datasets <https://flower.ai/docs/datasets/>`_ to load CIFAR10, a popular colored image classification
dataset for machine learning. The call to
:code:`FederatedDataset(dataset="cifar10")` downloads CIFAR10, partitions it, and caches it locally.
We assign an integer to `partition_id` for each client in our federated learning example, starting from 0.

.. code-block:: python

    fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2, seed=42)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]

Next, we need a model. For the purpose of this tutorial, we use MobileNetV2 with 10 output classes:

.. code-block:: python

    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

The Flower server interacts with clients through an interface called
:code:`Client`. When the server selects a particular client for training, it
sends training instructions over the network. The client receives those
instructions and calls one of the :code:`Client` methods to run your code
(i.e., to train the neural network we defined earlier).

Flower provides a convenience class called :code:`NumPyClient` which makes it
easier to implement the :code:`Client` interface when your workload uses Keras.
The :code:`NumPyClient` interface defines three methods which can be
implemented in the following way:

.. code-block:: python

    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), {"accuracy": float(accuracy)}


Next, we create a client function that returns instances of :code:`CifarClient` on-demand when called:

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

For simple workloads, we create a :code:`ServerApp` and leave all the
configuration possibilities at their default values. In a file named
:code:`server.py`, import Flower and create a :code:`ServerApp`:

.. code-block:: python

    from flwr.server import ServerApp

    app = ServerApp()


Train the model, federated!
---------------------------

With both client and server ready, we can now run everything and see federated
learning in action. First, we run the :code:`flower-superlink` command in one terminal to start the infrastructure. This step only needs to be run once.

.. admonition:: Note
    :class: note

    In this example, the :code:`--insecure` command line argument starts Flower without HTTPS and is only used for prototyping. To run with HTTPS, we instead use the argument :code:`--certificates` and pass the paths to the certificates. Please refer to `Flower CLI reference <ref-api-cli.html>`_ for implementation details.

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
    INFO :      Starting Flower ServerApp, config: num_rounds=1, no round_timeout
    INFO :
    INFO :      [INIT]
    INFO :      Requesting initial parameters from one random client
    INFO :      Received initial parameters from one random client
    INFO :      Evaluating initial global parameters
    INFO :
    INFO :      [ROUND 1]
    INFO :      configure_fit: strategy sampled 2 clients (out of 2)
    INFO :      aggregate_fit: received 2 results and 0 failures
    WARNING :   No fit_metrics_aggregation_fn provided
    INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
    INFO :      aggregate_evaluate: received 2 results and 0 failures
    WARNING :   No evaluate_metrics_aggregation_fn provided
    INFO :
    INFO :      [SUMMARY]
    INFO :      Run finished 1 rounds in 7.20s
    INFO :      History (loss, distributed):
    INFO :          '\tround 1: 2.302561044692993\n'
    INFO :

Congratulations! You've successfully built and run your first federated
learning system. The full source code for this can be found in
|quickstart_tf_link|_.

.. |quickstart_tf_link| replace:: :code:`examples/quickstart-tensorflow/client.py`
.. _quickstart_tf_link: https://github.com/adap/flower/blob/main/examples/quickstart-tensorflow/client.py