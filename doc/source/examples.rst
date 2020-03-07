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

PyTorch examples will follow shortly. Stay tuned!


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

  $ python -m flower_examples.tf_mnist_grpc_server

For each Flower client that should participate in the trainig, open a new
terminal window and type:

  $ python -m flower_examples.tf_mnist_grpc_client

For more details, see :code:`src/flower_examples/tf_mnist_grpc_server.py` and
:code:`src/flower_examples/tf_mnist_grpc_client.py`. 

MNIST Image Classification (single-machine)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is also a single-machine version of the previous MNIST image
classification example. It executes both client and server in a single process,
which might be handy for some types of experimentation. To run it, type::

  $ python src/flower_examples/mnist.py
