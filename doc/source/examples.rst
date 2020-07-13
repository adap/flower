Usage Examples
==============

Flower comes with a number of usage examples. The examples demonstrate how
Flower can be used to federate different kinds of existing machine learning
pipelines, usually leveraging popular machine learning frameworks such as
`PyTorch <https://pytorch.org/>`_ or
`TensorFlow <https://www.tensorflow.org/>`_.


Extra Dependencies
------------------

The core Flower framework tries to keep a minimal set of dependencies. Because
the examples demonstrate Flower in the context of differnt machine learning
frameworks, one needs to install those additional dependencies before running
an example::

  $ pip install git+https://github.com/adap/flower.git#egg=flower[examples-tensorflow]

The previous command installs the extra :code:`examples-tensorflow`. Please
consult :code:`pyproject.toml` for a full list of possible extras (section
:code:`[tool.poetry.extras]`).


PyTorch Examples
----------------

Our PyTorch examples are based on PyTorch 1.5.1. They should work with other
releases as well. So far, we provide the follow examples.

CIFAR-10 Image Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`CIFAR-10 and CIFAR-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ are
popular RGB image datasets. The Flower CIFAR-10 example uses PyTorch to train a
simple CNN classifier in a federated learning setup with two clients.

First, start a Flower server:

  $ ./src/flower_example/pytorch/run-server.sh

Then, start the two clients in a new terminal window:

  $ ./src/flower_example/pytorch/run-clients.sh

For more details, see :code:`src/flower_example/pytorch`.


TensorFlow Examples
-------------------

Our TensorFlow examples are based on TensorFlow 2.0 or newer. So far, we
provide the following examples.

MNIST Image Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

`MNIST <http://yann.lecun.com/exdb/mnist/>`_ is often used as the "Hello,
world!" of machine learning. We follow this tradition and provide an example
which samples random local datasets from MNIST and trains a simple image
classification model over those partitions.

First, start a Flower server:

  $ ./src/flower_example/tf_fashion_mnist/run-server.sh

Then, start the two clients in a new terminal window:

  $ ./src/flower_example/tf_fashion_mnist/run-clients.sh


For more details, see :code:`src/flower_example/tf_fashion_mnist`.
