Example: PyTorch - From Centralized To Federated
================================================

Before you start with this tutorial we recommend to setup a virtual environment as described `here <https://flower.dev/docs/recommended-env-setup.html>`_. 
This tutorial will show you how to build Flower on top of an existing machine learning workload. We are using PyTorch to train a Convolutional Neural Network on the CIFAR-10 dataset. First, we introduce this machine learning task with a centralized training approach based on the `Deep Learning with PyTorch <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_ tutorial. Then, we build on the centralized training to run it in a federated fashion.

Centralized Training
--------------------

We will post the complete centralized training here and explain it shortly. If you have question about the centralized training have a look to the `PyTorch tutorial <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_. 

Let's now create a :code:`cifar.py` to train your CIFAR-10 dataset centralized. 

First, all required packages as :code:`torch` and :code:`torchvision` need to be implemented. You can see that we do not implement any package for federated learning. You can keep all these imports as they are even for the federated learning process at a later point.

.. code-block:: python

    from typing import Tuple

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from torch import Tensor

As already mentioned we will use the CIFAR-10 dataset for the machine learning workload. The model setup is defined in the :code:`class Net()` and will be a Convolutional Neural Network.

.. code-block:: python

    DATA_ROOT = "~/data/cifar-10"

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

The :code:`load_data()` loads the CIFAR-10 training and test data. As soon as the data is loaded it is also normalized. 

.. code-block:: python

    def load_data() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Load CIFAR-10 (training and test set)."""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        # Training set
        trainset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=True, download=True, transform=transform
        )
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

        # Test set
        testset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

        return trainloader, testloader

We now need to define the training :code:`train()` by running through all training data samples and measuring the the loss and optimize it. 

The evalution of the model is done by :code:`test()`. The function loops over all test samples and measures the loss of the model based on the test dataset. 

.. code-block:: python

    def train(
        net: Net,
        trainloader: torch.utils.data.DataLoader,
        epochs: int,
        device: torch.device,  # pylint: disable=no-member
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
        device: torch.device,  # pylint: disable=no-member
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
                _, predicted = torch.max(outputs.data, 1)  # pylint: disable-msg=no-member
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy

After defining the data loader, training and evaluation we can start to centrally train the CIFAR-10 dataset as you may have done it before.

.. code-block:: python

    def main():
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Central PyTorch Training")
        print("Load data")
        trainloader, testloader = load_data()
        print("Start training")
        train(net=Net(), trainloader=trainloader, epochs=2, device=DEVICE)
        print("Start Testing")
        loss, accuracy = test(net=Net(), testloader=testloader, device=DEVICE)
        print("Loss: ", loss)
        print("Accuracy: ", accuracy)


    if __name__ == "__main__":
        main()

You can now run your machine learning workload with:

.. code-block:: python

    python3 cifar.py


Federated Training
------------------

The concept of centralized learning as shown in the previous section is known for most of you and many of you have set it up already. Normally, if you want to run machine learning workloads federated you have to change your complete code and set up everything from scratch. This is quite a big effort. 

However, with Flower you can evolve your pre-existing code into a federated learning setup without the need for a major rewrite.

The concept is easy to understand. We have to set up a *server* and take the :code:`cifar.py` for the *clients* that are connected to the *server*. The *server* sends model parameters to the clients. The *clients* running the training and updating the paramters. The updated parameters are evaluated and send back to the *server* that averages all received paramters. This is one round of a federated learning process. 

Our example consists of one *server* and two *clients* all having the same model. 

Let us set up the :code:`server.py` first. The *server* needs first the flower package. Then, you define the IP address and how many federated learning rounds you need. 

.. code-block:: python

    import flwr as fl

    if __name__ == "__main__":
        fl.server.start_server("[::]:8080", config={"num_rounds": 3})

You can already start the *server* with:

.. code-block:: python

    python3 server.py

Finally, we will setup the *clients* with :code:`client.py` and use the previously defined centralized training in :code:`cifar.py`. In order to update the model parameters on the *server* and *client* we also need to implement :code:`flwr`, :code:`torch` and :code:`torchvision`.

.. code-block:: python

    from collections import OrderedDict
    from typing import Dict, List, Tuple

    import numpy as np
    import torch
    import torchvision

    import flwr as fl

    from . import cifar

    DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

The implemenation of the Flower *client* is done with the :code:`CifarClient()`. This *client* has two paramter definition and two running  functions:

#. :code:`set_parameters`
    * set the model weights on the local model that are received from the server
    * loop over all model parameters
#. :code:`get_parameters`
    * encapsulates the model weights into Flower parameters
#. :code:`fit`
    * set the local model weights
    * train the local model
    * receive the updated local model weights
#. :code:`evaluate`
    * test the local model
    * measure loss and accuracy based on the test set

The main *Client* functions :code:`train()` and :code:`evaluate()` make use of the previously created :code:`cifar.py` where your model, training and evaluation setup is already defined. 

.. code-block:: python

    # Flower Client
    class CifarClient(fl.client.NumPyClient):

        def __init__(
            self,
            model: cifar.Net,
            trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader,
        ) -> None:
            self.model = model
            self.trainloader = trainloader
            self.testloader = testloader

        def get_parameters(self) -> List[np.ndarray]:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        def set_parameters(self, parameters: List[np.ndarray]) -> None:
            # Set model parameters from a list of NumPy ndarrays.
            state_dict = OrderedDict(
                {
                    k: torch.Tensor(v)
                    for k, v in zip(self.model.state_dict().keys(), parameters)
                }
            )
            self.model.load_state_dict(state_dict, strict=True)

        def fit(
            self, parameters: List[np.ndarray], config: Dict[str, str]
        ) -> Tuple[List[np.ndarray], int]:
            # Set model parameters
            self.set_parameters(parameters)

            # Train model
            cifar.train(self.model, self.trainloader, epochs=1, device=DEVICE)

            # Return the updated model parameters
            return self.get_parameters(), len(self.trainloader)

        def evaluate(
            self, parameters: List[np.ndarray], config: Dict[str, str]
        ) -> Tuple[int, float, float]:
            # Use provided parameters to update the local model
            self.set_parameters(parameters)

            # Evaluate the updated model on the local dataset
            loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)

            # Return the number of evaluation examples and the evaluation result (loss)
            return len(self.testloader), float(loss), float(accuracy)

After you setup the Flower *Client* you can start your federated training and connect the client to the server. 

You load your data and model by using :code:`cifar.py`. Start :code:`CifarClient()` with Flower :code:`fl.client.start_numpy_client()` by setting the IP adress as done in the :code:`server.py`. 

.. code-block:: python

    def main() -> None:
        """Load data, start CifarClient."""

        # Load model and data
        model = cifar.Net()
        model.to(DEVICE)
        trainloader, testloader = cifar.load_data()

        # Start client
        client = CifarClient(model, trainloader, testloader)
        fl.client.start_numpy_client("[::]:8080", client)


    if __name__ == "__main__":
        main()

That's it. You can now run

.. code-block:: python

    python client.py

in two different terminals and your centralized PyTorch example is running federated without touching your central training. The full `source code <https://github.com/adap/flower/blob/main/examples/pytorch_minimal/client.py>`_ for this can be found in :code:`examples/pytorch_minimal/client.py`.
