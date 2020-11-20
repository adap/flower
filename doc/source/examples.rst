Usage Examples
==============

Flower comes with a number of usage examples. The examples demonstrate how
Flower can be used to federate different kinds of existing machine learning
pipelines, usually leveraging popular machine learning frameworks such as
`PyTorch <https://pytorch.org/>`_ or
`TensorFlow <https://www.tensorflow.org/>`_.


Extra Dependencies
------------------

The core Flower framework keeps a minimal set of dependencies. The examples
demonstrate Flower in the context of different machine learning frameworks, so
additional dependencies need to be installed before an example can be run.

For PyTorch examples::

  $ pip install flwr[examples-pytorch]

For TensorFlow examples::

  $ pip install flwr[examples-tensorflow]

For both PyTorch and TensorFlow examples::

  $ pip install flwr[examples-pytorch,examples-tensorflow]

Please consult :code:`pyproject.toml` for a full list of possible extras
(section :code:`[tool.poetry.extras]`).


PyTorch Examples
----------------

Our PyTorch examples are based on PyTorch 1.7. They should work with other
releases as well. So far, we provide the following examples.

CIFAR-10 Image Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`CIFAR-10 and CIFAR-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ are
popular RGB image datasets. The Flower CIFAR-10 example uses PyTorch to train a
simple CNN classifier in a federated learning setup with two clients.

First, start a Flower server:

  $ ./src/py/flwr_example/pytorch_cifar/run-server.sh

Then, start the two clients in a new terminal window:

  $ ./src/py/flwr_example/pytorch_cifar/run-clients.sh

For more details, see :code:`src/py/flwr_example/pytorch_cifar`.

ImageNet-2012 Image Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`ImageNet-2012 <http://www.image-net.org/>`_ is one of the major computer
vision datasets. The Flower ImageNet example uses PyTorch to train a ResNet-18
classifier in a federated learning setup with ten clients.

First, start a Flower server:

  $ ./src/py/flwr_example/pytorch_imagenet/run-server.sh

Then, start the two clients in a new terminal window:

  $ ./src/py/flwr_example/pytorch_imagenet/run-clients.sh

For more details, see :code:`src/py/flwr_example/pytorch_imagenet`.


TensorFlow Examples
-------------------

Our TensorFlow examples are based on TensorFlow 2.0 or newer. So far, we
provide the following examples.

Fashion-MNIST Image Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ is often
used as the "Hello, world!" of machine learning. We follow this tradition and
provide an example which samples random local datasets from Fashion-MNIST and
trains a simple image classification model over those partitions.

First, start a Flower server:

  $ ./src/py/flwr_example/tensorflow_fashion_mnist/run-server.sh

Then, start the two clients in a new terminal window:

  $ ./src/py/flwr_example/tensorflow_fashion_mnist/run-clients.sh

For more details, see :code:`src/py/flwr_example/tensorflow_fashion_mnist`.
