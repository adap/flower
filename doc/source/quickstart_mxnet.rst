Quickstart (MXNet)
==================

In this tutorial we will learn how to train a Sequential Model on MNIST using Flower and MXNet. 

First of all, it is recommended to create a virtual environment and run everything within a `virtualenv <https://flower.dev/docs/recommended-env-setup.html>`_. 

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

We use MXNet to load MNIST, a popular image classification dataset of handrwritten digits for machine learning. The MXNet :code:`mx.test_utils.get_mnist()` downloads the training and test data. 

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

Define the training and loss with MXNet. We train the model by looping over the dataset, measure the corresponding loss and optimize it. 

.. code-block:: python

    def train(net, train_data, epoch):
        trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.03})
        metric = mx.metric.Accuracy()
        softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        for i in range(epoch):
            train_data.reset()
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
                        outputs.append(z)
                metric.update(label, outputs)
                trainer.step(batch.data[0].shape[0])
            name, acc = metric.get()
            metric.reset()
            print("training acc at epoch %d: %s=%f" % (i, name, acc))


Define then the validation of the  machine learning model. We loop over the test set and measure the loss and accuracy on the test set. 

.. code-block:: python

    def test(net, val_data):
        metric = mx.metric.Accuracy()
        loss_metric = mx.metric.Loss()
        loss = 0.0
        val_data.reset()
        for batch in val_data:
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=DEVICE, batch_axis=0)
            label = gluon.utils.split_and_load(
                batch.label[0], ctx_list=DEVICE, batch_axis=0
            )
            outputs = []
            for x in data:
                outputs.append(net(x))
                loss_metric.update(label, outputs)
                loss += loss_metric.get()[1]
            metric.update(label, outputs)
        print("validation acc: %s=%f" % metric.get())
        print("validation loss:", loss)
        accuracy = metric.get()[1]
        return loss, accuracy

After defining the training and testing of a MXNet machine learning model, we use the functions for the Flower clients.

The Flower clients will use a simple Sequential model:

.. code-block:: python

    def main():
        def model():
            net = nn.Sequential()
            net.add(nn.Dense(256, activation="relu"))
            net.add(nn.Dense(10))
            net.collect_params().initialize()
            return net

        train_data, val_data = load_data()

        model = model()
        init = nd.random.uniform(shape=(2, 784))
        model(init)

After loading the dataset with :code:`load_data()` we implement a Flower client. 

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

which can be implemented in the following way:

.. code-block:: python

    class MNISTClient(fl.client.NumPyClient):
        def get_parameters(self):
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
            train(model, train_data, epoch=1)
            return self.get_parameters(), train_data.batch_size, {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(model, val_data)
            return float(loss),  val_data.batch_size, {"accuracy":float(accuracy)}
    

We can now create an instance of our class :code:`MNISTClient` and add one line
to actually run this client:

.. code-block:: python

     fl.client.start_numpy_client("0.0.0.0:8080", client=MNISTClient())

That's it for the client. We only have to implement :code:`Client` or
:code:`NumPyClient` and call :code:`fl.client.start_client()` or :code:`fl.client.start_numpy_client()`. The string :code:`"[::]:8080"` tells the client which server to connect to. In our case we can run the server and the client on the same machine, therefore we use
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

    fl.server.start_server(config={"num_rounds": 3})

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
You should now see how the training does in the very first terminal (the one that started the server):

.. code-block:: shell

    INFO flower 2021-03-02 11:03:45,534 | app.py:76 | Flower server running (insecure, 3 rounds)
    INFO flower 2021-03-02 11:03:45,534 | server.py:72 | Getting initial parameters
    INFO flower 2021-03-02 11:03:53,639 | server.py:74 | Evaluating initial parameters
    INFO flower 2021-03-02 11:03:53,639 | server.py:87 | [TIME] FL starting
    DEBUG flower 2021-03-02 11:04:00,162 | server.py:165 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2021-03-02 11:04:04,979 | server.py:177 | fit_round received 2 results and 0 failures
    DEBUG flower 2021-03-02 11:04:04,985 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-03-02 11:04:05,242 | server.py:149 | evaluate received 2 results and 0 failures
    DEBUG flower 2021-03-02 11:04:05,244 | server.py:165 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2021-03-02 11:04:10,510 | server.py:177 | fit_round received 2 results and 0 failures
    DEBUG flower 2021-03-02 11:04:10,515 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-03-02 11:04:10,855 | server.py:149 | evaluate received 2 results and 0 failures
    DEBUG flower 2021-03-02 11:04:10,856 | server.py:165 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2021-03-02 11:04:15,432 | server.py:177 | fit_round received 2 results and 0 failures
    DEBUG flower 2021-03-02 11:04:15,436 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-03-02 11:04:15,730 | server.py:149 | evaluate received 2 results and 0 failures
    INFO flower 2021-03-02 11:04:15,730 | server.py:122 | [TIME] FL finished in 22.09073099998932
    INFO flower 2021-03-02 11:04:15,731 | app.py:109 | app_fit: losses_distributed [(1, 12.912875175476074), (2, 14.816988945007324), (3, 15.702619552612305)]
    INFO flower 2021-03-02 11:04:15,731 | app.py:110 | app_fit: accuracies_distributed []
    INFO flower 2021-03-02 11:04:15,731 | app.py:111 | app_fit: losses_centralized []
    INFO flower 2021-03-02 11:04:15,731 | app.py:112 | app_fit: accuracies_centralized []
    DEBUG flower 2021-03-02 11:04:15,733 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-03-02 11:04:16,010 | server.py:149 | evaluate received 2 results and 0 failures
    INFO flower 2021-03-02 11:04:16,010 | app.py:121 | app_evaluate: federated loss: 15.702619552612305
    INFO flower 2021-03-02 11:04:16,011 | app.py:125 | app_evaluate: results [('ipv4:127.0.0.1:59960', EvaluateRes(loss=15.706217765808105, num_examples=100, accuracy=0.0, metrics={'accuracy': 0.9222})), ('ipv4:127.0.0.1:59962', EvaluateRes(loss=15.699021339416504, num_examples=100, accuracy=0.0, metrics={'accuracy': 0.9218}))]
    INFO flower 2021-03-02 11:04:16,011 | app.py:127 | app_evaluate: failures []

Congratulations!
You've successfully built and run your first federated learning system.
The full `source code <https://github.com/adap/flower/blob/main/examples/quickstart_mxnet/client.py>`_ for this example can be found in :code:`examples/quickstart_mxnet`.
