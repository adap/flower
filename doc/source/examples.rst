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
demonstrate Flower in the context of differnt machine learning frameworks, so 
additional dependencies need to be installed before an example can be run.

For PyTorch examples::

  $ pip install flwr[examples-pytorch]

For TensorFlow examples::

  $ pip install flwr[examples-tensorflow]

For both PyTorch and TensorFlow examples::

  $ pip install flwr[examples-pytorch,examples-tensorflow]

Please consult :code:`pyproject.toml` for a full list of possible extras
(section :code:`[tool.poetry.extras]`).


Run Examples Using Docker
-------------------------

Flower examples can also be run through Docker without the need for most of the
setup steps that are otherwise necessary::

  # Create docker network `flwr` so that containers can reach each other by name
  $ docker network create flwr
  
  # Build the Flower docker containers
  $ ./dev/docker_build.sh

  # Run the docker containers (will tail a logfile created by a central logserver)
  $ ./src/py/flwr_example/tensorflow/run-docker.sh

This will start a slightly smaller workload with only four clients.


PyTorch Examples
----------------

Our PyTorch examples are based on PyTorch 1.6. They should work with other
releases as well. So far, we provide the follow examples.

CIFAR-10 Image Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`CIFAR-10 and CIFAR-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ are
popular RGB image datasets. The Flower CIFAR-10 example uses PyTorch to train a
simple CNN classifier in a federated learning setup with two clients.

First, start a Flower server:

  $ ./src/py/flwr_example/pytorch/run-server.sh

Then, start the two clients in a new terminal window:

  $ ./src/py/flwr_example/pytorch/run-clients.sh

For more details, see :code:`src/py/flwr_example/pytorch`.


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

  $ ./src/py/flwr_example/tensorflow/run-server.sh

Then, start the two clients in a new terminal window:

  $ ./src/py/flwr_example/tensorflow/run-clients.sh

For more details, see :code:`src/py/flwr_example/tensorflow`.
