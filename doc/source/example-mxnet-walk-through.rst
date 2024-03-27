Example: MXNet - Run MXNet Federated
====================================

This tutorial will show you how to use Flower to build a federated version of an existing MXNet workload.
We are using MXNet to train a Sequential model on the MNIST dataset.
We will structure the example similar to our `PyTorch - From Centralized To Federated <https://github.com/adap/flower/blob/main/examples/pytorch-from-centralized-to-federated>`_ walkthrough. MXNet and PyTorch are very similar and a very good comparison between MXNet and PyTorch is given `here <https://mxnet.apache.org/versions/1.7.0/api/python/docs/tutorials/getting-started/to-mxnet/pytorch.html>`_.
First, we build a centralized training approach based on the `Handwritten Digit Recognition <https://mxnet.apache.org/versions/1.7.0/api/python/docs/tutorials/packages/gluon/image/mnist.html>`_ tutorial.
Then, we build upon the centralized training code to run the training in a federated fashion.

Before we start setting up our MXNet example, we install the :code:`mxnet` and :code:`flwr` packages:

.. code-block:: shell

  $ pip install mxnet
  $ pip install flwr


MNIST Training with MXNet
-------------------------

We begin with a brief description of the centralized training code based on a :code:`Sequential` model.
If you want a more in-depth explanation of what's going on then have a look at the official `MXNet tutorial <https://mxnet.apache.org/versions/1.7.0/api/python/docs/tutorials/>`_.

Let's create a new file called:code:`mxnet_mnist.py` with all the components required for a traditional (centralized) MNIST training. 
First, the MXNet package :code:`mxnet` needs to be imported.
You can see that we do not yet import the :code:`flwr` package for federated learning. This will be done later. 

.. code-block:: python

    from __future__ import print_function
    from typing import Tuple
    import mxnet as mx
    from mxnet import gluon
    from mxnet.gluon import nn
    from mxnet import autograd as ag
    import mxnet.ndarray as F
    from mxnet import nd

    # Fixing the random seed
    mx.random.seed(42)

The :code:`load_data()` function loads the MNIST training and test sets.

.. code-block:: python

    def load_data() -> Tuple[mx.io.NDArrayIter, mx.io.NDArrayIter]:
        print("Download Dataset")
        # Download MNIST data
        mnist = mx.test_utils.get_mnist()
        batch_size = 100
        train_data = mx.io.NDArrayIter(
            mnist["train_data"], mnist["train_label"], batch_size, shuffle=True
        )
        val_data = mx.io.NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size)
        return train_data, val_data

As already mentioned, we will use the MNIST dataset for this machine learning workload. The model architecture (a very simple :code:`Sequential` model) is defined in :code:`model()`.

.. code-block:: python

    def model():
        # Define simple Sequential model
        net = nn.Sequential()
        net.add(nn.Dense(256, activation="relu"))
        net.add(nn.Dense(64, activation="relu"))
        net.add(nn.Dense(10))
        net.collect_params().initialize()
        return net

We now need to define the training (function :code:`train()`) which loops over the training set and measures the loss for each batch of training examples.

.. code-block:: python

    def train(
        net: mx.gluon.nn, train_data: mx.io.NDArrayIter, epoch: int, device: mx.context
    ) -> None:
        trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.03})
        # Use Accuracy and Cross Entropy Loss as the evaluation metric.
        accuracy_metric = mx.metric.Accuracy()
        loss_metric = mx.metric.CrossEntropy()
        metrics = mx.metric.CompositeEvalMetric()
        for child_metric in [accuracy_metric, loss_metric]:
            metrics.add(child_metric)
        softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        for i in range(epoch):
            # Reset the train data iterator.
            train_data.reset()
            # Calculate number of samples
            num_examples = 0
            # Loop over the train data iterator.
            for batch in train_data:
                # Splits train data into multiple slices along batch_axis
                # and copy each slice into a context.
                data = gluon.utils.split_and_load(
                    batch.data[0], ctx_list=device, batch_axis=0
                )
                # Splits train labels into multiple slices along batch_axis
                # and copy each slice into a context.
                label = gluon.utils.split_and_load(
                    batch.label[0], ctx_list=device, batch_axis=0
                )
                outputs = []
                # Inside training scope
                with ag.record():
                    for x, y in zip(data, label):
                        z = net(x)
                        # Computes softmax cross entropy loss.
                        loss = softmax_cross_entropy_loss(z, y)
                        # Backpropogate the error for one iteration.
                        loss.backward()
                        outputs.append(z.softmax())
                        num_examples += len(x)
                # Updates internal evaluation
                metric.update(label, outputs)
                # Make one step of parameter update. Trainer needs to know the
                # batch size of data to normalize the gradient by 1/batch_size.
                trainer.step(batch.data[0].shape[0])
            # Gets the evaluation result.
            trainings_metric = metrics.get_name_value()
            print("Accuracy & loss at epoch %d: %s" % (i, trainings_metric))
        return trainings_metric, num_examples

The evaluation of the model is defined in function :code:`test()`. The function loops over all test samples and measures the loss and accuracy of the model based on the test dataset. 

.. code-block:: python

    def test(
        net: mx.gluon.nn, val_data: mx.io.NDArrayIter, device: mx.context
    ) -> Tuple[float, float]:
        # Use Accuracy and Cross Entropy Loss as the evaluation metric.
        accuracy_metric = mx.metric.Accuracy()
        loss_metric = mx.metric.CrossEntropy()
        metrics = mx.metric.CompositeEvalMetric()
        for child_metric in [accuracy_metric, loss_metric]:
            metrics.add(child_metric)
        # Reset the validation data iterator.
        val_data.reset()
        # Get number of samples for val_dat
        num_examples = 0
        # Loop over the validation data iterator.
        for batch in val_data:
            # Splits validation data into multiple slices along batch_axis
            # and copy each slice into a context.
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=device, batch_axis=0)
            # Splits validation label into multiple slices along batch_axis
            # and copy each slice into a context.
            label = gluon.utils.split_and_load(
                batch.label[0], ctx_list=device, batch_axis=0
            )
            outputs = []
            for x in data:
                outputs.append(net(x).softmax())
                num_examples += len(x) 
            # Updates internal evaluation
            metrics.update(label, outputs)
        return metrics.get_name_value(), num_examples

Having defined the data loading, model architecture, training, and evaluation we can put everything together and train our model on MNIST. Note that the GPU/CPU device for the training and testing is defined within the :code:`ctx` (context).  

.. code-block:: python

    def main():
        # Setup context to GPU and if not available to CPU
        DEVICE = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
        # Load train and validation data
        train_data, val_data = load_data()
        # Define sequential model
        net = model()
        # Start forward propagation to initialize model parameters (optional) 
        init = nd.random.uniform(shape=(2, 784))
        net(init)
        # Start model training based on training set
        train(net=net, train_data=train_data, epoch=5, device=DEVICE)
        # Evaluate model using loss and accuracy
        eval_metric, _ = test(net=net, val_data=val_data, device=DEVICE)
        acc = eval_metric[0]
        loss = eval_metric[1]
        print("Evaluation Loss: ", loss)
        print("Evaluation Accuracy: ", acc)

    if __name__ == "__main__":
            main()

You can now run your (centralized) MXNet machine learning workload:

.. code-block:: python

    python3 mxnet_mnist.py

So far this should all look fairly familiar if you've used MXNet (or even PyTorch) before.
Let's take the next step and use what we've built to create a simple federated learning system consisting of one server and two clients.

MXNet meets Flower
------------------

So far, it was not easily possible to use MXNet workloads for federated learning because federated learning is not supported in MXNet. Since Flower is fully agnostic towards the underlying machine learning framework, it can be used to federated arbitrary machine learning workloads. This section will show you how Flower can be used to federate our centralized MXNet workload.

The concept to federate an existing workload is always the same and easy to understand.
We have to start a *server* and then use the code in :code:`mxnet_mnist.py` for the *clients* that are connected to the *server*.
The *server* sends model parameters to the clients. The *clients* run the training and update the parameters.
The updated parameters are sent back to the *server* which averages all received parameter updates.
This describes one round of the federated learning process and we repeat this for multiple rounds.

Our example consists of one *server* and two *clients*. Let's set up :code:`server.py` first. The *server* needs to import the Flower package :code:`flwr`.
Next, we use the :code:`start_server` function to start a server and tell it to perform three rounds of federated learning.

.. code-block:: python

    import flwr as fl

    if __name__ == "__main__":
        fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=3))

We can already start the *server*:

.. code-block:: python

    python3 server.py

Finally, we will define our *client* logic in :code:`client.py` and build upon the previously defined MXNet training in :code:`mxnet_mnist.py`.
Our *client* needs to import :code:`flwr`, but also :code:`mxnet` to update the parameters on our MXNet model:

.. code-block:: python

    from typing import Dict, List, Tuple

    import flwr as fl
    import numpy as np
    import mxnet as mx
    from mxnet import nd

    import mxnet_mnist


Implementing a Flower *client* basically means implementing a subclass of either :code:`flwr.client.Client` or :code:`flwr.client.NumPyClient`.
Our implementation will be based on :code:`flwr.client.NumPyClient` and we'll call it :code:`MNISTClient`.
:code:`NumPyClient` is slightly easier to implement than :code:`Client` if you use a framework with good NumPy interoperability (like PyTorch or MXNet) because it avoids some of the boilerplate that would otherwise be necessary.
:code:`MNISTClient` needs to implement four methods, two methods for getting/setting model parameters, one method for training the model, and one method for testing the model:

#. :code:`set_parameters (optional)`
    * set the model parameters on the local model that are received from the server
    * transform MXNet :code:`NDArray`'s to NumPy :code:`ndarray`'s
    * loop over the list of model parameters received as NumPy :code:`ndarray`'s (think list of neural network layers)
#. :code:`get_parameters`
    * get the model parameters and return them as a list of NumPy :code:`ndarray`'s (which is what :code:`flwr.client.NumPyClient` expects)
#. :code:`fit`
    * update the parameters of the local model with the parameters received from the server
    * train the model on the local training set
    * get the updated local model weights and return them to the server
#. :code:`evaluate`
    * update the parameters of the local model with the parameters received from the server
    * evaluate the updated model on the local test set
    * return the local loss and accuracy to the server

The challenging part is to transform the MXNet parameters from :code:`NDArray` to :code:`NumPy Arrays` to make it readable for Flower. 

The two :code:`NumPyClient` methods :code:`fit` and :code:`evaluate` make use of the functions :code:`train()` and :code:`test()` previously defined in :code:`mxnet_mnist.py`.
So what we really do here is we tell Flower through our :code:`NumPyClient` subclass which of our already defined functions to call for training and evaluation.
We included type annotations to give you a better understanding of the data types that get passed around.

.. code-block:: python

    class MNISTClient(fl.client.NumPyClient):
        """Flower client implementing MNIST classification using MXNet."""

        def __init__(
            self,
            model: mxnet_mnist.model(),
            train_data: mx.io.NDArrayIter,
            val_data: mx.io.NDArrayIter,
            device: mx.context,
        ) -> None:
            self.model = model
            self.train_data = train_data
            self.val_data = val_data
            self.device = device

        def get_parameters(self, config) -> List[np.ndarray]:
            # Return model parameters as a list of NumPy Arrays
            param = []
            for val in self.model.collect_params(".*weight").values():
                p = val.data()
                # convert parameters from MXNet NDArray to Numpy Array required by Flower Numpy Client
                param.append(p.asnumpy())
            return param

        def set_parameters(self, parameters: List[np.ndarray]) -> None:
            # Collect model parameters and set new weight values
            params = zip(self.model.collect_params(".*weight").keys(), parameters)
            for key, value in params:
                self.model.collect_params().setattr(key, value)

        def fit(
            self, parameters: List[np.ndarray], config: Dict[str, str]
        ) -> Tuple[List[np.ndarray], int]:
            # Set model parameters, train model, return updated model parameters
            self.set_parameters(parameters)
            [accuracy, loss], num_examples = mxnet_mnist.train(
            self.model, self.train_data, epoch=2, device=self.device
            )
            results = {"accuracy": accuracy[1], "loss": loss[1]}
            return self.get_parameters(config={}), num_examples, results

        def evaluate(
            self, parameters: List[np.ndarray], config: Dict[str, str]
        ) -> Tuple[int, float, float]:
            # Set model parameters, evaluate model on local test dataset, return result
            self.set_parameters(parameters)
            [accuracy, loss], num_examples = mxnet_mnist.test(
            self.model, self.val_data, device=self.device
            )
            print("Evaluation accuracy & loss", accuracy, loss)
            return (
                float(loss[1]),
                num_examples,
                {"accuracy": float(accuracy[1])},
            )

Having defined data loading, model architecture, training, and evaluation we can put everything together and train our :code:`Sequential` model on MNIST.

.. code-block:: python

    def main() -> None:
        """Load data, start MNISTClient."""

        # Setup context to GPU and if not available to CPU
        DEVICE = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
        
        # Load data
        train_data, val_data = mxnet_mnist.load_data()
        
        # Define model from centralized training
        model = mxnet_mnist.model()
        
        # Make one forward propagation to initialize parameters
        init = nd.random.uniform(shape=(2, 784))
        model(init)

        # Start Flower client
        client = MNISTClient(model, train_data, val_data, DEVICE)
        fl.client.start_numpy_client(server_address="0.0.0.0:8080", client)


    if __name__ == "__main__":
        main()

And that's it. You can now open two additional terminal windows and run

.. code-block:: python

    python3 client.py

in each window (make sure that the server is still running before you do so) and see your MXNet project run federated learning across two clients. Congratulations!

Next Steps
----------

The full source code for this example: `MXNet: From Centralized To Federated (Code) <https://github.com/adap/flower/blob/main/examples/mxnet-from-centralized-to-federated>`_.
Our example is of course somewhat over-simplified because both clients load the exact same dataset, which isn't realistic.
You're now prepared to explore this topic further. How about using a CNN or using a different dataset? How about adding more clients?
