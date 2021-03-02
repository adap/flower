Example: MXNet - Run MXNet Federated
====================================

This tutorial will show you how to use Flower to build a federated version of a MXNet workload.
We are using MXNet to train a Sequential model on a MNIST dataset. We will setup the example similar to our `PyTorch - From Centralized To Federated <https://github.com/adap/flower/blob/main/examples/pytorch_from_centralized_to_federated>`_ walk through. 
First, we build a centralized training approach based on the `Handwritten Digit Recognition <https://mxnet.apache.org/versions/1.7.0/api/python/docs/tutorials/packages/gluon/image/mnist.html>`_ tutorial.
Then, we build upon the centralized training code to run the training in a federated fashion.

Before we start setting up our MXNet example we install the :code:`mxnet` and :code:`flwr` packages:

.. code-block:: shell

  $ pip install mxnet
  $ pip install flwr


Centralized Training
--------------------

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

As already mentioned we will use the MNIST dataset for this machine learning workload. The model architecture (a very simple Sequential model) is defined in :code:`model()`.

.. code-block:: python

    def model():
        # Define simple Sequential model
        net = nn.Sequential()
        net.add(nn.Dense(256, activation="relu"))
        net.add(nn.Dense(10))
        net.collect_params().initialize()
        return net
