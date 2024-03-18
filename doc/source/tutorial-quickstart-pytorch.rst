.. _quickstart-pytorch:


Quickstart PyTorch
==================

.. meta::
   :description: Check out this Federated Learning quickstart tutorial for using Flower with PyTorch to train a CNN model on MNIST.

..  youtube:: jOmmuzMIQ4c
   :width: 100%

In this tutorial we will learn how to train a Convolutional Neural Network on CIFAR10 using Flower and PyTorch.

First of all, it is recommended to create a virtual environment and run everything within a :doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

Our example consists of one *server* and two *clients* all having the same model.

*Clients* are responsible for generating individual weight-updates for the model based on their local datasets.
These updates are then sent to the *server* which will aggregate them to produce a better model. Finally, the *server* sends this improved version of the model back to each *client*.
A complete cycle of weight updates is called a *round*.

Now that we have a rough idea of what is going on, let's get started. We first need to install Flower. You can do this by running :

.. code-block:: shell

  $ pip install flwr

Since we want to use PyTorch to solve a computer vision task, let's go ahead and install PyTorch and the **torchvision** library:

.. code-block:: shell

  $ pip install torch torchvision


Flower Client
-------------

Now that we have all our dependencies installed, let's run a simple distributed training with two clients and one server. Our training procedure and network architecture are based on PyTorch's `Deep Learning with PyTorch <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_.

In a file called :code:`client.py`, import Flower and PyTorch related packages:

.. code-block:: python

    from collections import OrderedDict

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10

    import flwr as fl

In addition, we define the device allocation in PyTorch with:

.. code-block:: python

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

We use PyTorch to load CIFAR10, a popular colored image classification dataset for machine learning. The PyTorch :code:`DataLoader()` downloads the training and test data that are then normalized.

.. code-block:: python

    def load_data():
        """Load CIFAR-10 (training and test set)."""
        transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        trainset = CIFAR10(".", train=True, download=True, transform=transform)
        testset = CIFAR10(".", train=False, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
        testloader = DataLoader(testset, batch_size=32)
        num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
        return trainloader, testloader, num_examples

Define the loss and optimizer with PyTorch. The training of the dataset is done by looping over the dataset, measure the corresponding loss and optimize it.

.. code-block:: python

    def train(net, trainloader, epochs):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        for _ in range(epochs):
            for images, labels in trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(net(images), labels)
                loss.backward()
                optimizer.step()

Define then the validation of the  machine learning network. We loop over the test set and measure the loss and accuracy of the test set.

.. code-block:: python

    def test(net, testloader):
        """Validate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy

After defining the training and testing of a PyTorch machine learning model, we use the functions for the Flower clients.

The Flower clients will use a simple CNN adapted from 'PyTorch: A 60 Minute Blitz':

.. code-block:: python

    class Net(nn.Module):
        def __init__(self) -> None:
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Load model and data
    net = Net().to(DEVICE)
    trainloader, testloader, num_examples = load_data()

After loading the data set with :code:`load_data()` we define the Flower interface.

The Flower server interacts with clients through an interface called
:code:`Client`. When the server selects a particular client for training, it
sends training instructions over the network. The client receives those
instructions and calls one of the :code:`Client` methods to run your code
(i.e., to train the neural network we defined earlier).

Flower provides a convenience class called :code:`NumPyClient` which makes it
easier to implement the :code:`Client` interface when your workload uses PyTorch.
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

    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, trainloader, epochs=1)
            return self.get_parameters(config={}), num_examples["trainset"], {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}

We can now create an instance of our class :code:`CifarClient` and add one line
to actually run this client:

.. code-block:: python

     fl.client.start_client(server_address="[::]:8080", client=CifarClient().to_client())

That's it for the client. We only have to implement :code:`Client` or
:code:`NumPyClient` and call :code:`fl.client.start_client()`. If you implement a client of type :code:`NumPyClient` you'll need to first call its :code:`to_client()` method. The string :code:`"[::]:8080"` tells the client which server to connect to. In our case we can run the server and the client on the same machine, therefore we use
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

    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))

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

    INFO flower 2021-02-25 14:00:27,227 | app.py:76 | Flower server running (insecure, 3 rounds)
    INFO flower 2021-02-25 14:00:27,227 | server.py:72 | Getting initial parameters
    INFO flower 2021-02-25 14:01:15,881 | server.py:74 | Evaluating initial parameters
    INFO flower 2021-02-25 14:01:15,881 | server.py:87 | [TIME] FL starting
    DEBUG flower 2021-02-25 14:01:41,310 | server.py:165 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2021-02-25 14:02:00,256 | server.py:177 | fit_round received 2 results and 0 failures
    DEBUG flower 2021-02-25 14:02:00,262 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-02-25 14:02:03,047 | server.py:149 | evaluate received 2 results and 0 failures
    DEBUG flower 2021-02-25 14:02:03,049 | server.py:165 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2021-02-25 14:02:23,908 | server.py:177 | fit_round received 2 results and 0 failures
    DEBUG flower 2021-02-25 14:02:23,915 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-02-25 14:02:27,120 | server.py:149 | evaluate received 2 results and 0 failures
    DEBUG flower 2021-02-25 14:02:27,122 | server.py:165 | fit_round: strategy sampled 2 clients (out of 2)
    DEBUG flower 2021-02-25 14:03:04,660 | server.py:177 | fit_round received 2 results and 0 failures
    DEBUG flower 2021-02-25 14:03:04,671 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-02-25 14:03:09,273 | server.py:149 | evaluate received 2 results and 0 failures
    INFO flower 2021-02-25 14:03:09,273 | server.py:122 | [TIME] FL finished in 113.39180790000046
    INFO flower 2021-02-25 14:03:09,274 | app.py:109 | app_fit: losses_distributed [(1, 650.9747924804688), (2, 526.2535400390625), (3, 473.76959228515625)]
    INFO flower 2021-02-25 14:03:09,274 | app.py:110 | app_fit: accuracies_distributed []
    INFO flower 2021-02-25 14:03:09,274 | app.py:111 | app_fit: losses_centralized []
    INFO flower 2021-02-25 14:03:09,274 | app.py:112 | app_fit: accuracies_centralized []
    DEBUG flower 2021-02-25 14:03:09,276 | server.py:139 | evaluate: strategy sampled 2 clients
    DEBUG flower 2021-02-25 14:03:11,852 | server.py:149 | evaluate received 2 results and 0 failures
    INFO flower 2021-02-25 14:03:11,852 | app.py:121 | app_evaluate: federated loss: 473.76959228515625
    INFO flower 2021-02-25 14:03:11,852 | app.py:122 | app_evaluate: results [('ipv6:[::1]:36602', EvaluateRes(loss=351.4906005859375, num_examples=10000, accuracy=0.0, metrics={'accuracy': 0.6067})), ('ipv6:[::1]:36604', EvaluateRes(loss=353.92742919921875, num_examples=10000, accuracy=0.0, metrics={'accuracy': 0.6005}))]
    INFO flower 2021-02-25 14:03:27,514 | app.py:127 | app_evaluate: failures []

Congratulations!
You've successfully built and run your first federated learning system.
The full `source code <https://github.com/adap/flower/blob/main/examples/quickstart-pytorch/client.py>`_ for this example can be found in :code:`examples/quickstart-pytorch`.
