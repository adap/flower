Example: MXNet - Run MXNet Federated
====================================

This tutorial will show you how to use Flower to build a federated version of a MXNet workload.
We are using MXNet to train a Sequential model on a MNIST dataset. We will setup the example similar to our `PyTorch - From Centralized To Federated <https://github.com/adap/flower/blob/main/examples/pytorch_from_centralized_to_federated>`_ walk through. MXNet and PyTorch are very similar and a very good comparison between MXNet and PyTorch is given `here <https://mxnet.apache.org/versions/1.7.0/api/python/docs/tutorials/getting-started/to-mxnet/pytorch.html>`_.
First, we build a centralized training approach based on the `Handwritten Digit Recognition <https://mxnet.apache.org/versions/1.7.0/api/python/docs/tutorials/packages/gluon/image/mnist.html>`_ tutorial.
Then, we build upon the centralized training code to run the training in a federated fashion.

Before we start setting up our MXNet example we install the :code:`mxnet` and :code:`flwr` packages:

.. code-block:: shell

  $ pip install mxnet
  $ pip install flwr


MNIST Training with MXNet
-------------------------

We begin with a brief description of the centralized training code based on a Sequential model.
If you want a more in-depth explanation of what's going on then have a look at the official `MXNet tutorial <https://mxnet.apache.org/versions/1.7.0/api/python/docs/tutorials/>`_.

Let's create an new file called :code:`mxnet_mnist.py` with all the components required for a traditional (centralized) MNIST training. 
First, the MXNet package :code:`mxnet` needs to be imported.
You can see that we do not import any :code:`flwr` package for federated learning. This will be done later. 

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

As already mentioned we will use the MNIST dataset for this machine learning workload. The model architecture (a very simple Sequential model) is defined in :code:`model()`.

.. code-block:: python

    def model():
        # Define simple Sequential model
        net = nn.Sequential()
        net.add(nn.Dense(256, activation="relu"))
        net.add(nn.Dense(10))
        net.collect_params().initialize()
        return net

We now need to define the training (function :code:`train()`) which loops over the training set and measures the the loss for each batch of training examples.

.. code-block:: python

    def train(
        net: mx.gluon.nn, train_data: mx.io.NDArrayIter, epoch: int, device: mx.context
    ) -> None:
        trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.03})
        # Use Accuracy as the evaluation metric.
        metric = mx.metric.Accuracy()
        softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        for i in range(epoch):
            # Reset the train data iterator.
            train_data.reset()
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
                        outputs.append(z)
                # Updates internal evaluation
                metric.update(label, outputs)
                # Make one step of parameter update. Trainer needs to know the
                # batch size of data to normalize the gradient by 1/batch_size.
                trainer.step(batch.data[0].shape[0])
            # Gets the evaluation result.
            name, acc = metric.get()
            # Reset evaluation result to initial state.
            metric.reset()
            print("training acc at epoch %d: %s=%f" % (i, name, acc))

The evalution of the model is defined in function :code:`test()`. The function loops over all test samples and measures the loss and accuracy of the model based on the test dataset. 

.. code-block:: python

    def test(
        net: mx.gluon.nn, val_data: mx.io.NDArrayIter, device: mx.context
    ) -> Tuple[float, float]:
        # Use Accuracy as the evaluation metric.
        metric = mx.metric.Accuracy()
        loss_metric = mx.metric.Loss()
        loss = 0.0
        # Reset the validation data iterator.
        val_data.reset()
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
                outputs.append(net(x))
                loss_metric.update(label, outputs)
                loss += loss_metric.get()[1]
            # Updates internal evaluation
            metric.update(label, outputs)
        print("validation acc: %s=%f" % metric.get())
        print("validation loss:", loss)
        accuracy = metric.get()[1]
        return loss, accuracy

Having defined defining the data loading, model architecture, training, and evaluation we can put everything together and train our model on MNIST. Note that the GPU/CPU device for the training and testing is defined within the :code:`ctx` (context).  

.. code-block:: python

    def main():
        # Setup context to GPU and if not available to CPU
        DEVICE = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
        # Load train and validation data
        train_data, val_data = load_data()
        # Define sequential model
        net = model()
        init = nd.random.uniform(shape=(2, 784))
        net(init)
        # Start model training based on training set
        train(net=net, train_data=train_data, epoch=5, device=DEVICE)
        # Evaluate model using loss and accuracy
        loss, acc = test(net=net, val_data=val_data, device=DEVICE)
        print("Loss: ", loss)
        print("Accuracy: ", acc)

    if __name__ == "__main__":
            main()

You can now run your MXNet machine learning workload:

.. code-block:: python

    python3 mxnet_mnist.py

So far this should all look fairly familiar if you've used MXNet or even PyTorch before.
Let's take the next step and use what we've built to create a simple federated learning system consisting of one server and two clients.

MXNet meets Flower
------------------