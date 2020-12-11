Example: PyTorch Centralized and Federated
==========================================

Before you start with this tutorial we recommend to setup a virtual environment as described `here <https://flower.dev/docs/recommended-env-setup.html>`_. 
This tutorial will show you how to build Flower on top of an existing machine learning workload. We are using PyTorch to train a Convolutional Neural Network on a CIFAR10 dataset. First, we setup this machine learning task with a centralized training approach based on `Deep Learning with PyTorch <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_ tutorial. Secondly, we use the centralized training and run it federated.

Centralized Training
-------------------- 

We will post the complete centralized training here and explain it shortly. If you have question about the centralized training have a look to the `PyTorch tutorial <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_. 

Let's now create a :code:`cifar.py` to train your CIFAR10 dataset centralized. 

First, all required packages as :code:`torch` and :code:`torchvision` need to be implemented. You can see that we do not implement any package for federated learning. You can keep all these import as they are even for the federated learning process at a later point.

.. code-block:: python

    from typing import Tuple

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from torch import Tensor

As already mentioned we will use the CIFAR10 dataset for the machine learning workload. The model setup is defined in the :code:`class Net()` and will be a Convolutional Neural Network.

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

The :code:`load_data()` loads the CIFAR10 training and test data. As soon as the data is loaded it is also normalized. 

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

After defining the data loader, training and evaluation we can start to centrally train the CIFAR10 dataset as you may have done it before.

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

However, if you use Flower you can keep your code and put Flower on top. You can play around with federated learning setups without destroying your actual pre-existing code.

The concept is easy to understand. We have to set up a *server* and take the :code:`cifar.py` for the *clients* that are connected to the *server*. The *server* sends model parameter to the clients. The *clients* running the training and updating the paramters. The updated parameters are evaluated and send back to the *server* that averages all received paramters. This is one round of a federated learning process. 

Our example consists of one *server* and two *clients* all having the same model. 

Let us set up the :code:`server.py` first. The *server* needs first the flower package. Then, you define the IP adress and how many federated learning rounds you need. 

.. code-block:: python

    import flwr as fl

    if __name__ == "__main__":
        fl.server.start_server("[::]:8080", config={"num_rounds": 3})

You can already start the *server* with:

.. code-block:: python

    python3 server.py

Finally, we will setup the *clients* with :code:`client.py` and use the previously defined centralized training in :code:`cifar.py`.  