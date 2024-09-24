Example: PyTorch - From Centralized To Federated
================================================

This tutorial will show you how to use Flower to build a federated version of an existing machine learning workload.
We are using PyTorch to train a Convolutional Neural Network on the CIFAR-10 dataset.
First, we introduce this machine learning task with a centralized training approach based on the `Deep Learning with PyTorch <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_ tutorial.
Then, we build upon the centralized training code to run the training in a federated fashion.

Centralized Training
--------------------

We begin with a brief description of the centralized CNN training code.
If you want a more in-depth explanation of what's going on then have a look at the official `PyTorch tutorial <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_.

Let's create a new file called :code:`cifar.py` with all the components required for a traditional (centralized) training on CIFAR-10. 
First, all required packages (such as :code:`torch` and :code:`torchvision`) need to be imported.
You can see that we do not import any package for federated learning.
You can keep all these imports as they are even when we add the federated learning components at a later point.

.. code-block:: python

    from typing import Tuple, Dict

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from torch import Tensor
    from torchvision.datasets import CIFAR10

As already mentioned we will use the CIFAR-10 dataset for this machine learning workload. The model architecture (a very simple Convolutional Neural Network) is defined in :code:`class Net()`.

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

        def forward(self, x: Tensor) -> Tensor:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

The :code:`load_data()` function loads the CIFAR-10 training and test sets. The :code:`transform` normalized the data after loading. 

.. code-block:: python

    DATA_ROOT = "~/data/cifar-10"

    def load_data() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
        """Load CIFAR-10 (training and test set)."""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        trainset = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
        testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
        return trainloader, testloader, num_examples

We now need to define the training (function :code:`train()`) which loops over the training set, measures the loss, backpropagates it, and then takes one optimizer step for each batch of training examples.

The evaluation of the model is defined in the function :code:`test()`. The function loops over all test samples and measures the loss of the model based on the test dataset. 

.. code-block:: python

    def train(
        net: Net,
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        device: torch.device,
    ) -> None:
        """Train the network."""
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

        # Train the network
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                images, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0


    def test(
        net: Net,
        testloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Tuple[float, float]:
        """Validate the network on the entire test set."""
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy

Having defined the data loading, model architecture, training, and evaluation we can put everything together and train our CNN on CIFAR-10.

.. code-block:: python

    def main():
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Centralized PyTorch training")
        print("Load data")
        trainloader, testloader, _ = load_data()
        print("Start training")
        net=Net().to(DEVICE)
        train(net=net, trainloader=trainloader, epochs=2, device=DEVICE)
        print("Evaluate model")
        loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
        print("Loss: ", loss)
        print("Accuracy: ", accuracy)


    if __name__ == "__main__":
        main()

You can now run your machine learning workload:

.. code-block:: python

    python3 cifar.py

So far, this should all look fairly familiar if you've used PyTorch before.
Let's take the next step and use what we've built to create a simple federated learning system consisting of one server and two clients.

Federated Training
------------------

The simple machine learning project discussed in the previous section trains the model on a single dataset (CIFAR-10), we call this centralized learning.
This concept of centralized learning, as shown in the previous section, is probably known to most of you, and many of you have used it previously.
Normally, if you'd want to run machine learning workloads in a federated fashion, then you'd have to change most of your code and set everything up from scratch. This can be a considerable effort. 

However, with Flower you can evolve your pre-existing code into a federated learning setup without the need for a major rewrite.

The concept is easy to understand.
We have to start a *server* and then use the code in :code:`cifar.py` for the *clients* that are connected to the *server*.
The *server* sends model parameters to the clients. The *clients* run the training and update the parameters.
The updated parameters are sent back to the *server* which averages all received parameter updates.
This describes one round of the federated learning process and we repeat this for multiple rounds. 

Our example consists of one *server* and two *clients*. Let's set up :code:`server.py` first. The *server* needs to import the Flower package :code:`flwr`.
Next, we use the :code:`start_server` function to start a server and tell it to perform three rounds of federated learning.

.. code-block:: python

    import flwr as fl

    if __name__ == "__main__":
        fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=3))

We can already start the *server*:

.. code-block:: python

    python3 server.py

Finally, we will define our *client* logic in :code:`client.py` and build upon the previously defined centralized training in :code:`cifar.py`.
Our *client* needs to import :code:`flwr`, but also :code:`torch` to update the parameters on our PyTorch model:

.. code-block:: python

    from collections import OrderedDict
    from typing import Dict, List, Tuple

    import numpy as np
    import torch

    import cifar
    import flwr as fl

    DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Implementing a Flower *client* basically means implementing a subclass of either :code:`flwr.client.Client` or :code:`flwr.client.NumPyClient`.
Our implementation will be based on :code:`flwr.client.NumPyClient` and we'll call it :code:`CifarClient`.
:code:`NumPyClient` is slightly easier to implement than :code:`Client` if you use a framework with good NumPy interoperability (like PyTorch or TensorFlow/Keras) because it avoids some of the boilerplate that would otherwise be necessary.
:code:`CifarClient` needs to implement four methods, two methods for getting/setting model parameters, one method for training the model, and one method for testing the model:

#. :code:`set_parameters`
    * set the model parameters on the local model that are received from the server
    * loop over the list of model parameters received as NumPy :code:`ndarray`'s (think list of neural network layers)
#. :code:`get_parameters`
    * get the model parameters and return them as a list of NumPy :code:`ndarray`'s (which is what :code:`flwr.client.NumPyClient` expects)
#. :code:`fit`
    * update the parameters of the local model with the parameters received from the server
    * train the model on the local training set
    * get the updated local model weights and return them to the server
#. :code:`evaluate`
    * update the parameters of the local model with the parameters received from the server
    * evaluate the updated model on the local test set
    * return the local loss and accuracy to the server

The two :code:`NumPyClient` methods :code:`fit` and :code:`evaluate` make use of the functions :code:`train()` and :code:`test()` previously defined in :code:`cifar.py`.
So what we really do here is we tell Flower through our :code:`NumPyClient` subclass which of our already defined functions to call for training and evaluation.
We included type annotations to give you a better understanding of the data types that get passed around.

.. code-block:: python

    class CifarClient(fl.client.NumPyClient):
        """Flower client implementing CIFAR-10 image classification using
        PyTorch."""

        def __init__(
            self,
            model: cifar.Net,
            trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader,
            num_examples: Dict,
        ) -> None:
            self.model = model
            self.trainloader = trainloader
            self.testloader = testloader
            self.num_examples = num_examples

        def get_parameters(self, config) -> List[np.ndarray]:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        def set_parameters(self, parameters: List[np.ndarray]) -> None:
            # Set model parameters from a list of NumPy ndarrays
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

        def fit(
            self, parameters: List[np.ndarray], config: Dict[str, str]
        ) -> Tuple[List[np.ndarray], int, Dict]:
            # Set model parameters, train model, return updated model parameters
            self.set_parameters(parameters)
            cifar.train(self.model, self.trainloader, epochs=1, device=DEVICE)
            return self.get_parameters(config={}), self.num_examples["trainset"], {}

        def evaluate(
            self, parameters: List[np.ndarray], config: Dict[str, str]
        ) -> Tuple[float, int, Dict]:
            # Set model parameters, evaluate model on local test dataset, return result
            self.set_parameters(parameters)
            loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
            return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

All that's left to do it to define a function that loads both model and data, creates a :code:`CifarClient`, and starts this client.
You load your data and model by using :code:`cifar.py`. Start :code:`CifarClient` with the function :code:`fl.client.start_client()` by pointing it at the same IP address we used in :code:`server.py`: 

.. code-block:: python

    def main() -> None:
        """Load data, start CifarClient."""

        # Load model and data
        model = cifar.Net()
        model.to(DEVICE)
        trainloader, testloader, num_examples = cifar.load_data()

        # Start client
        client = CifarClient(model, trainloader, testloader, num_examples)
        fl.client.start_client(server_address="0.0.0.0:8080", client.to_client())


    if __name__ == "__main__":
        main()

And that's it. You can now open two additional terminal windows and run

.. code-block:: python

    python3 client.py

in each window (make sure that the server is running before you do so) and see your (previously centralized) PyTorch project run federated learning across two clients. Congratulations!

Next Steps
----------

The full source code for this example: `PyTorch: From Centralized To Federated (Code) <https://github.com/adap/flower/blob/main/examples/pytorch-from-centralized-to-federated>`_.
Our example is, of course, somewhat over-simplified because both clients load the exact same dataset, which isn't realistic.
You're now prepared to explore this topic further. How about using different subsets of CIFAR-10 on each client? How about adding more clients?
