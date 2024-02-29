.. _quickstart-mxnet:


Quickstart MXNet
================

.. warning:: MXNet is no longer maintained and has been moved into `Attic <https://attic.apache.org/projects/mxnet.html>`_. As a result, we would encourage you to use other ML frameworks alongside Flower, for example, PyTorch. This tutorial might be removed in future versions of Flower.

.. meta::
   :description: Check out this Federated Learning quickstart tutorial for using Flower with MXNet to train a Sequential model on MNIST.

In this tutorial, we will learn how to train a :code:`Sequential` model on MNIST using Flower and MXNet.

It is recommended to create a virtual environment and run everything within this :doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

Our example consists of one *server* and two *clients* all having the same model.

*Clients* are responsible for generating individual model parameter updates for the model based on their local datasets.
These updates are then sent to the *server* which will aggregate them to produce an updated global model. Finally, the *server* sends this improved version of the model back to each *client*.
A complete cycle of parameters updates is called a *round*.

Now that we have a rough idea of what is going on, let's get started. We first need to install Flower. You can do this by running:

.. code-block:: shell

  $ pip install flwr

Since we want to use MXNet, let's go ahead and install it:

.. code-block:: shell

  $ pip install mxnet


Flower Client
-------------

Now that we have all our dependencies installed, let's run a simple distributed training with two clients and one server. Our training procedure and network architecture are based on MXNet´s `Hand-written Digit Recognition tutorial <https://mxnet.apache.org/api/python/docs/tutorials/packages/gluon/image/mnist.html>`_.

In a file called :code:`client.py`, import Flower and MXNet related packages:

.. code-block:: python

    import flwr as fl

    import numpy as np

    import mxnet as mx
    from mxnet import nd
    from mxnet import gluon
    from mxnet.gluon import nn
    from mxnet import autograd as ag
    import mxnet.ndarray as F

In addition, define the device allocation in MXNet with:

.. code-block:: python

    DEVICE = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]

We use MXNet to load MNIST, a popular image classification dataset of handwritten digits for machine learning. The MXNet utility :code:`mx.test_utils.get_mnist()` downloads the training and test data.

.. code-block:: python

    def load_data():
        print("Download Dataset")
        mnist = mx.test_utils.get_mnist()
        batch_size = 100
        train_data = mx.io.NDArrayIter(
            mnist["train_data"], mnist["train_label"], batch_size, shuffle=True
        )
        val_data = mx.io.NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size)
        return train_data, val_data

Define the training and loss with MXNet. We train the model by looping over the dataset, measure the corresponding loss, and optimize it.

.. code-block:: python

    def train(net, train_data, epoch):
        trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.03})
        trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.01})
        accuracy_metric = mx.metric.Accuracy()
        loss_metric = mx.metric.CrossEntropy()
        metrics = mx.metric.CompositeEvalMetric()
        for child_metric in [accuracy_metric, loss_metric]:
            metrics.add(child_metric)
        softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        for i in range(epoch):
            train_data.reset()
            num_examples = 0
            for batch in train_data:
                data = gluon.utils.split_and_load(
                    batch.data[0], ctx_list=DEVICE, batch_axis=0
                )
                label = gluon.utils.split_and_load(
                    batch.label[0], ctx_list=DEVICE, batch_axis=0
                )
                outputs = []
                with ag.record():
                    for x, y in zip(data, label):
                        z = net(x)
                        loss = softmax_cross_entropy_loss(z, y)
                        loss.backward()
                        outputs.append(z.softmax())
                        num_examples += len(x)
                metrics.update(label, outputs)
                trainer.step(batch.data[0].shape[0])
            trainings_metric = metrics.get_name_value()
            print("Accuracy & loss at epoch %d: %s" % (i, trainings_metric))
        return trainings_metric, num_examples


Next, we define the validation of our machine learning model. We loop over the test set and measure both loss and accuracy on the test set.

.. code-block:: python

    def test(net, val_data):
        accuracy_metric = mx.metric.Accuracy()
        loss_metric = mx.metric.CrossEntropy()
        metrics = mx.metric.CompositeEvalMetric()
        for child_metric in [accuracy_metric, loss_metric]:
            metrics.add(child_metric)
        val_data.reset()
        num_examples = 0
        for batch in val_data:
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=DEVICE, batch_axis=0)
            label = gluon.utils.split_and_load(
                batch.label[0], ctx_list=DEVICE, batch_axis=0
            )
            outputs = []
            for x in data:
                outputs.append(net(x).softmax())
                num_examples += len(x)
            metrics.update(label, outputs)
        return metrics.get_name_value(), num_examples

After defining the training and testing of a MXNet machine learning model, we use these functions to implement a Flower client.

Our Flower clients will use a simple :code:`Sequential` model:

.. code-block:: python

    def main():
        def model():
            net = nn.Sequential()
            net.add(nn.Dense(256, activation="relu"))
            net.add(nn.Dense(64, activation="relu"))
            net.add(nn.Dense(10))
            net.collect_params().initialize()
            return net

        train_data, val_data = load_data()

        model = model()
        init = nd.random.uniform(shape=(2, 784))
        model(init)

After loading the dataset with :code:`load_data()` we perform one forward propagation to initialize the model and model parameters with :code:`model(init)`. Next, we implement a Flower client.

The Flower server interacts with clients through an interface called
:code:`Client`. When the server selects a particular client for training, it
sends training instructions over the network. The client receives those
instructions and calls one of the :code:`Client` methods to run your code
(i.e., to train the neural network we defined earlier).

Flower provides a convenience class called :code:`NumPyClient` which makes it
easier to implement the :code:`Client` interface when your workload uses MXNet.
Implementing :code:`NumPyClient` usually means defining the following methods
(:code:`set_parameters` is optional though):

#. :code:`get_parameters`
    * return the model weight as a list of NumPy ndarrays
#. :code:`set_parameters` (optional)
    * update the local model weights with the parameters received from the server
#. :code:`fit`
    * set the local model weights
    * train the local model
    * receive the updated local model weights
#. :code:`evaluate`
    * test the local model

They can be implemented in the following way:

.. code-block:: python

    class MNISTClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            param = []
            for val in model.collect_params(".*weight").values():
                p = val.data()
                param.append(p.asnumpy())
            return param

        def set_parameters(self, parameters):
            params = zip(model.collect_params(".*weight").keys(), parameters)
            for key, value in params:
                model.collect_params().setattr(key, value)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            [accuracy, loss], num_examples = train(model, train_data, epoch=2)
            results = {"accuracy": float(accuracy[1]), "loss": float(loss[1])}
            return self.get_parameters(config={}), num_examples, results

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            [accuracy, loss], num_examples = test(model, val_data)
            print("Evaluation accuracy & loss", accuracy, loss)
            return float(loss[1]), val_data.batch_size, {"accuracy": float(accuracy[1])}


We can now create an instance of our class :code:`MNISTClient` and add one line
to actually run this client:

.. code-block:: python

     fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=MNISTClient())

That's it for the client. We only have to implement :code:`Client` or
:code:`NumPyClient` and call :code:`fl.client.start_client()` or :code:`fl.client.start_numpy_client()`. The string :code:`"0.0.0.0:8080"` tells the client which server to connect to. In our case we can run the server and the client on the same machine, therefore we use
:code:`"0.0.0.0:8080"`. If we run a truly federated workload with the server and
clients running on different machines, all that needs to change is the
:code:`server_address` we pass to the client.

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
learning in action. Federated learning systems usually have a server and multiple clients. We
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

    INFO flower 2021-03-11 11:59:04,512 | app.py:76 | Flower server running (insecure, 3 rounds)
    INFO flower 2021-03-11 11:59:04,512 | server.py:72 | Getting initial parameters
    INFO flower 2021-03-11 11:59:09,089 | server.py:74 | Evaluating initial parameters
    INFO flower 2021-03-11 11:59:09,089 | server.py:87 | [TIME] FL starting
    DEBUG flower 2021-03-11 11:59:11,997 | server.py:165 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2021-03-11 11:59:14,652 | server.py:177 | fit_round received 2 results and 0 failures
    DEBUG flower 2021-03-11 11:59:14,656 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-03-11 11:59:14,811 | server.py:149 | evaluate received 2 results and 0 failures
    DEBUG flower 2021-03-11 11:59:14,812 | server.py:165 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2021-03-11 11:59:18,499 | server.py:177 | fit_round received 2 results and 0 failures
    DEBUG flower 2021-03-11 11:59:18,503 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-03-11 11:59:18,784 | server.py:149 | evaluate received 2 results and 0 failures
    DEBUG flower 2021-03-11 11:59:18,786 | server.py:165 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2021-03-11 11:59:22,551 | server.py:177 | fit_round received 2 results and 0 failures
    DEBUG flower 2021-03-11 11:59:22,555 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-03-11 11:59:22,789 | server.py:149 | evaluate received 2 results and 0 failures
    INFO flower 2021-03-11 11:59:22,789 | server.py:122 | [TIME] FL finished in 13.700094900001204
    INFO flower 2021-03-11 11:59:22,790 | app.py:109 | app_fit: losses_distributed [(1, 1.5717803835868835), (2, 0.6093432009220123), (3, 0.4424773305654526)]
    INFO flower 2021-03-11 11:59:22,790 | app.py:110 | app_fit: accuracies_distributed []
    INFO flower 2021-03-11 11:59:22,791 | app.py:111 | app_fit: losses_centralized []
    INFO flower 2021-03-11 11:59:22,791 | app.py:112 | app_fit: accuracies_centralized []
    DEBUG flower 2021-03-11 11:59:22,793 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-03-11 11:59:23,111 | server.py:149 | evaluate received 2 results and 0 failures
    INFO flower 2021-03-11 11:59:23,112 | app.py:121 | app_evaluate: federated loss: 0.4424773305654526
    INFO flower 2021-03-11 11:59:23,112 | app.py:125 | app_evaluate: results [('ipv4:127.0.0.1:44344', EvaluateRes(loss=0.443320095539093, num_examples=100, accuracy=0.0, metrics={'accuracy': 0.8752475247524752})), ('ipv4:127.0.0.1:44346', EvaluateRes(loss=0.44163456559181213, num_examples=100, accuracy=0.0, metrics={'accuracy': 0.8761386138613861}))]
    INFO flower 2021-03-11 11:59:23,112 | app.py:127 | app_evaluate: failures []

Congratulations!
You've successfully built and run your first federated learning system.
The full `source code <https://github.com/adap/flower/blob/main/examples/quickstart-mxnet/client.py>`_ for this example can be found in :code:`examples/quickstart-mxnet`.
