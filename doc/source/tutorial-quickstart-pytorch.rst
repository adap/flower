.. _quickstart-pytorch:


Quickstart PyTorch
==================

.. meta::
   :description: Check out this Federated Learning quickstart tutorial for using Flower with PyTorch to train a CNN model on MNIST.

..  youtube:: jOmmuzMIQ4c
   :width: 100%

.. admonition:: Disclaimer
    :class: important

    The Quickstart PyTorch video uses slightly different Flower commands than this tutorial. Please follow the :doc:`Upgrade to Flower Next <how-to-upgrade-to-flower-next>` guide to convert commands shown in the video.

In this tutorial we will learn how to train a Convolutional Neural Network on CIFAR10 using the Flower framework and PyTorch.

First of all, it is recommended to create a virtual environment and run everything within a :doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

Our example consists of one *server* and two *clients* all having the same model.

*Clients* are responsible for generating individual weight-updates for the model based on their local datasets.
These updates are then sent to the *server* which will aggregate them to produce a better model. Finally, the *server* sends this improved version of the model back to each *client*.
A complete cycle of weight updates is called a *round*.

Now that we have a rough idea of what is going on, let's get started. We first need to install Flower and Flower Datasets. You can do this by running:

.. code-block:: shell

  $ pip install flwr flwr-datasets[vision]

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

    import flwr as fl
    from flwr_datasets import FederatedDataset

In addition, we define the device allocation in PyTorch with:

.. code-block:: python

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

We use `Flower Datasets <https://flower.ai/docs/datasets/>`_ to load CIFAR10, a popular colored image classification dataset for machine learning. The :code:`FederatedDataset()` module downloads, partitions, and preprocesses the dataset. The :code:`torchvision.transforms` modules are then used to normalize the training and test data.

.. code-block:: python

    def load_data(partition_id):
        """Load CIFAR-10 (training and test set)."""
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": 3})
        partition = fds.load_partition(partition_id)
        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        def apply_transforms(batch):
            """Apply transforms to the partition from FederatedDataset."""
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch

        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
        testloader = DataLoader(partition_train_test["test"], batch_size=32)
        return trainloader, testloader

We define the loss and optimizer with PyTorch. The training of the dataset is done by looping over the dataset, measure the corresponding loss, and optimize it.

.. code-block:: python

    def train(net, trainloader, epochs):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        net.train()
        for _ in range(epochs):
            for batch in trainloader:
                images = batch["img"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(net(images), labels)
                loss.backward()
                optimizer.step()

We then define the validation of the  machine learning network. We loop over the test set and measure the loss and accuracy of the test set.

.. code-block:: python

    def test(net, testloader):
        """Validate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        net.eval()
        with torch.no_grad():
            for batch in testloader:
                images = batch["img"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
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
    trainloader, testloader = load_data(partition_id=partition_id)

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
            return self.get_parameters(config={}), len(trainloader.dataset), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}

Next, we create a client function that returns instances of :code:`CifarClient` on-demand when called:

.. code-block:: python

    def client_fn(cid: str):
        return CifarClient().to_client()

Finally, we create a :code:`ClientApp()` object that uses this client function:

.. code-block:: python

    app = ClientApp(client_fn=client_fn)

That's it for the client. We only have to implement :code:`Client` or :code:`NumPyClient`, create a :code:`ClientApp`, and pass the client function to it. If we implement a client of type :code:`NumPyClient` we'll need to first call its :code:`to_client()` method.


Flower Server
-------------

For simple workloads, we create a :code:`ServerApp` and leave all the
configuration possibilities at their default values. In a file named
:code:`server.py`, import Flower and create a :code:`ServerApp`:

.. code-block:: python

    from flwr.server import ServerApp

    app = ServerApp()


Train the model, federated!
---------------------------

With both :code:`ClientApps` and :code:`ServerApp` ready, we can now run everything and see federated
learning in action. First, we run the :code:`flower-superlink` command in one terminal to start the infrastructure. This step only needs to be run once.

.. admonition:: Note
    :class: note

    In this example, the :code:`--insecure` command line argument starts Flower without HTTPS and is only used for prototyping. To run with HTTPS, we instead use the argument :code:`--certificates` and pass the paths to the certificates. Please refer to `Flower CLI reference <ref-api-cli.html>`_ for implementation details.

.. code-block:: shell

    $ flower-superlink --insecure

FL systems usually have a server and multiple clients. We therefore need to start multiple `SuperNode`s, one for each client, respectively. First, we open a new terminal and start the first `SuperNode` using the :code:`flower-client-app` command.

.. code-block:: shell

    $ flower-client-app client:app --insecure

In the above, we launch the :code:`app` object in the :code:`client.py` module.
Open another terminal and start the second `SuperNode`:

.. code-block:: shell

    $ flower-client-app client:app --insecure

Finally, in another terminal window, we run the `ServerApp`. This starts the actual training run:

.. code-block:: shell

    $ flower-server-app server:app --insecure

We should now see how the training does in the last terminal (the one that started the :code:`ServerApp`):

.. code-block:: shell

    WARNING :   Option `--insecure` was set. Starting insecure HTTP client connected to 0.0.0.0:9091.
    INFO :      Starting Flower ServerApp, config: num_rounds=1, no round_timeout
    INFO :
    INFO :      [INIT]
    INFO :      Requesting initial parameters from one random client
    INFO :      Received initial parameters from one random client
    INFO :      Evaluating initial global parameters
    INFO :
    INFO :      [ROUND 1]
    INFO :      configure_fit: strategy sampled 2 clients (out of 2)
    INFO :      aggregate_fit: received 2 results and 0 failures
    WARNING :   No fit_metrics_aggregation_fn provided
    INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
    INFO :      aggregate_evaluate: received 2 results and 0 failures
    WARNING :   No evaluate_metrics_aggregation_fn provided
    INFO :
    INFO :      [SUMMARY]
    INFO :      Run finished 1 rounds in 15.08s
    INFO :      History (loss, distributed):
    INFO :          '\tround 1: 241.32427430152893\n'
    INFO :

Congratulations!
You've successfully built and run your first federated learning system.
The full source code for this example can be found in |quickstart_pt_link|_.

.. |quickstart_pt_link| replace:: :code:`examples/quickstart-pytorch/client.py`
.. _quickstart_pt_link: https://github.com/adap/flower/blob/main/examples/quickstart-pytorch/client.py
