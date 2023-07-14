Code examples
=============

Flower comes with a number of usage examples. The examples demonstrate how
Flower can be used to federate different kinds of existing machine learning
pipelines, usually leveraging popular machine learning frameworks such as
`PyTorch <https://pytorch.org/>`_ or
`TensorFlow <https://www.tensorflow.org/>`_.

.. note::
   Flower usage examples used to be bundled with Flower in a package called
   ``flwr_example``. We are migrating those examples to standalone projects to
   make them easier to use. All new examples are based in the directory
   `examples <https://github.com/adap/flower/tree/main/examples>`_.

The following examples are available as standalone projects.


Quickstart TensorFlow/Keras
---------------------------

The TensorFlow/Keras quickstart example shows CIFAR-10 image classification
with MobileNetV2:

- `Quickstart TensorFlow (Code) <https://github.com/adap/flower/tree/main/examples/quickstart_tensorflow>`_
- `Quickstart TensorFlow (Tutorial) <https://flower.dev/docs/quickstart-tensorflow.html>`_
- `Quickstart TensorFlow (Blog Post) <https://flower.dev/blog/2020-12-11-federated-learning-in-less-than-20-lines-of-code>`_


Quickstart PyTorch
------------------

The PyTorch quickstart example shows CIFAR-10 image classification
with a simple Convolutional Neural Network:

- `Quickstart PyTorch (Code) <https://github.com/adap/flower/tree/main/examples/quickstart_pytorch>`_
- `Quickstart PyTorch (Tutorial) <https://flower.dev/docs/quickstart-pytorch.html>`_


PyTorch: From Centralized To Federated
--------------------------------------

This example shows how a regular PyTorch project can be federated using Flower:

- `PyTorch: From Centralized To Federated (Code) <https://github.com/adap/flower/tree/main/examples/pytorch_from_centralized_to_federated>`_
- `PyTorch: From Centralized To Federated (Tutorial) <https://flower.dev/docs/example-pytorch-from-centralized-to-federated.html>`_


Federated Learning on Raspberry Pi and Nvidia Jetson
----------------------------------------------------

This example shows how Flower can be used to build a federated learning system that run across Raspberry Pi and Nvidia Jetson:

- `Federated Learning on Raspberry Pi and Nvidia Jetson (Code) <https://github.com/adap/flower/tree/main/examples/embedded_devices>`_
- `Federated Learning on Raspberry Pi and Nvidia Jetson (Blog Post) <https://flower.dev/blog/2020-12-16-running_federated_learning_applications_on_embedded_devices_with_flower>`_



Legacy Examples (`flwr_example`)
--------------------------------

.. warning::
   The useage examples in `flwr_example` are deprecated and will be removed in
   the future. New examples are provided as standalone projects in
   `examples <https://github.com/adap/flower/tree/main/examples>`_.


Extra Dependencies
~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~

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
