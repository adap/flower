.. _quickstart-pytorch:


Quickstart PyTorch
==================

In this tutorial we will learn how to train a Neural Network on MNIST using Flower and CoreML. 

First of all, it is recommended to create a virtual environment and run everything within a `virtualenv <https://flower.dev/docs/recommended-env-setup.html>`_. 

Our example consists of one *server* and *clients* that all have the same model. 

*Clients* are responsible for generating individual weight-updates for the model based on their local datasets. 
These updates are then sent to the *server* which will aggregate them to produce a better model. Finally, the *server* sends this improved version of the model back to each *client*.
A complete cycle of weight updates is called a *round*.

Now that we have a rough idea of what is going on, let's get started. We first need to install Flower. You can do this by running :

.. code-block:: shell

  $ pip install flwr

Or simply install all dependencies using Poetry:

.. code-block:: shell

  $ poetry install

Flower Client
-------------

Now that we have all our dependencies installed, let's run a simple distributed training.
Please refer to the `full code example <https://github.com/adap/flower/tree/main/examples/ios>`_ to learn more.


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
learning in action. FL systems usually have a server and multiple clients. We
therefore have to start the server first:

.. code-block:: shell

    $ python server.py

Once the server is running we can start the clients in different terminals.
Build and run the client through your Xcode.

Congratulations!
You've successfully built and run your first federated learning system in your ios device.
The full `source code <https://github.com/adap/flower/blob/main/examples/quickstart_pytorch/client.py>`_ for this example can be found in :code:`examples/ios`.
