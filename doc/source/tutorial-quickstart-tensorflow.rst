.. _quickstart-tensorflow:


Quickstart TensorFlow
=====================

.. meta::
   :description: Check out this Federated Learning quickstart tutorial for using Flower with TensorFlow to train a MobilNetV2 model on CIFAR-10.

..  youtube:: FGTc2TQq7VM
   :width: 100%

Let's build a federated learning system in less than 20 lines of code!

Before Flower can be imported we have to install it:

.. code-block:: shell

  $ pip install flwr

Since we want to use the Keras API of TensorFlow (TF), we have to install TF as well: 

.. code-block:: shell

  $ pip install tensorflow


Flower Client
-------------

Next, in a file called :code:`client.py`, import Flower and TensorFlow:

.. code-block:: python

    import flwr as fl
    import tensorflow as tf

We use the Keras utilities of TF to load CIFAR10, a popular colored image classification
dataset for machine learning. The call to
:code:`tf.keras.datasets.cifar10.load_data()` downloads CIFAR10, caches it locally,
and then returns the entire training and test set as NumPy ndarrays.

.. code-block:: python

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

Next, we need a model. For the purpose of this tutorial, we use MobilNetV2 with 10 output classes:

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


We can now create an instance of our class :code:`CifarClient` and add one line
to actually run this client:

.. code-block:: python

     fl.client.start_client(server_address="[::]:8080", client=CifarClient().to_client())


That's it for the client. We only have to implement :code:`Client` or
:code:`NumPyClient` and call :code:`fl.client.start_client()`. If you implement a client of type :code:`NumPyClient` you'll need to first call its :code:`to_client()` method. The string :code:`"[::]:8080"` tells the client which server to connect to. In our case we can run the server and the client on the same machine, therefore we use
:code:`"[::]:8080"`. If we run a truly federated workload with the server and
clients running on different machines, all that needs to change is the
:code:`server_address` we point the client at.


Flower Server
-------------

For simple workloads we can start a Flower server and leave all the
configuration possibilities at their default values. In a file named
:code:`server.py`, import Flower and start the server:

.. code-block:: python

    import flwr as fl

    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))


Train the model, federated!
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

You should now see how the training does in the very first terminal (the one
that started the server):

.. code-block:: shell

    INFO flower 2021-02-25 14:15:46,741 | app.py:76 | Flower server running (insecure, 3 rounds)
    INFO flower 2021-02-25 14:15:46,742 | server.py:72 | Getting initial parameters
    INFO flower 2021-02-25 14:16:01,770 | server.py:74 | Evaluating initial parameters
    INFO flower 2021-02-25 14:16:01,770 | server.py:87 | [TIME] FL starting
    DEBUG flower 2021-02-25 14:16:12,341 | server.py:165 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2021-02-25 14:21:17,235 | server.py:177 | fit_round received 2 results and 0 failures
    DEBUG flower 2021-02-25 14:21:17,512 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-02-25 14:21:29,628 | server.py:149 | evaluate received 2 results and 0 failures
    DEBUG flower 2021-02-25 14:21:29,696 | server.py:165 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2021-02-25 14:25:59,917 | server.py:177 | fit_round received 2 results and 0 failures
    DEBUG flower 2021-02-25 14:26:00,227 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-02-25 14:26:11,457 | server.py:149 | evaluate received 2 results and 0 failures
    DEBUG flower 2021-02-25 14:26:11,530 | server.py:165 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2021-02-25 14:30:43,389 | server.py:177 | fit_round received 2 results and 0 failures
    DEBUG flower 2021-02-25 14:30:43,630 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-02-25 14:30:53,384 | server.py:149 | evaluate received 2 results and 0 failures
    INFO flower 2021-02-25 14:30:53,384 | server.py:122 | [TIME] FL finished in 891.6143046000007
    INFO flower 2021-02-25 14:30:53,385 | app.py:109 | app_fit: losses_distributed [(1, 2.3196680545806885), (2, 2.3202896118164062), (3, 2.1818180084228516)]
    INFO flower 2021-02-25 14:30:53,385 | app.py:110 | app_fit: accuracies_distributed []
    INFO flower 2021-02-25 14:30:53,385 | app.py:111 | app_fit: losses_centralized []
    INFO flower 2021-02-25 14:30:53,385 | app.py:112 | app_fit: accuracies_centralized []
    DEBUG flower 2021-02-25 14:30:53,442 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-02-25 14:31:02,848 | server.py:149 | evaluate received 2 results and 0 failures
    INFO flower 2021-02-25 14:31:02,848 | app.py:121 | app_evaluate: federated loss: 2.1818180084228516
    INFO flower 2021-02-25 14:31:02,848 | app.py:125 | app_evaluate: results [('ipv4:127.0.0.1:57158', EvaluateRes(loss=2.1818180084228516, num_examples=10000, accuracy=0.0, metrics={'accuracy': 0.21610000729560852})), ('ipv4:127.0.0.1:57160', EvaluateRes(loss=2.1818180084228516, num_examples=10000, accuracy=0.0, metrics={'accuracy': 0.21610000729560852}))]
    INFO flower 2021-02-25 14:31:02,848 | app.py:127 | app_evaluate: failures [] flower 2020-07-15 10:07:56,396 | app.py:77 | app_evaluate: failures []

Congratulations! You've successfully built and run your first federated
learning system. The full `source code <https://github.com/adap/flower/blob/main/examples/quickstart-tensorflow/client.py>`_ for this can be found in
:code:`examples/quickstart-tensorflow/client.py`.
