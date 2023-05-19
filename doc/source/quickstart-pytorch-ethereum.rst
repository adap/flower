.. _quickstart-pytorch:


Quickstart PyTorch Ethereum
===========================

In this tutorial we will learn how to train a Convolutional Neural Network on CIFAR10 using Flower and PyTorch with Ethereum.

First of all, it is recommended to create a virtual environment and run everything within a `virtualenv <https://flower.dev/docs/recommended-env-setup.html>`_. 

Our example consists of one *server* and two *clients* all having the same model. 

*Clients* are responsible for generating individual weight-updates for the model based on their local datasets. 
These updates store the trained model in *IPFS (InterPlanetary File System)*, store the model ID of the current round in *ganache (Ethereum local network)*, deliver the learning process and model ID to *server*, , which are sent and aggregated to create better models.
clear.
Finally, the *Server* receives the information of the trained model from *ganache* and *IPFS*, and uses it to upload the improved version of the model to *ganache* and *IPFS*.
A complete cycle of weight updates is called a *round*.

Now that we have a rough idea of what is going on, let's get started. We first need to install Flower. You can do this by running :

.. code-block:: shell

  $ pip install flwr

Since we want to use PyTorch to solve a computer vision task, let's go ahead and install PyTorch and the **torchvision** library: 

.. code-block:: shell

  $ pip install torch torchvision

Lastly, let's install the necessary components to upload and download the model information based on local Ethereum and IPFS. We need **Truffle** for smart contract deployment, **Ganache** for a local Ethereum network environment, and finally **IPFS** which is a distributed file system.

.. code-block:: shell

  $ npm i -g ganache-cli
  $ npm i -g truffle
  $ wget https://dist.ipfs.tech/kubo/v0.15.0/kubo_v0.12.2_linux-amd64.tar.gz
  $ tar -xvzf kubo_v0.12.2_linux-amd64.tar.gz
  $ cd kubo
  $ sudo bash install.sh

Smart Contract
______________
Now begin deploying the smart contract for running the Flower example.

First, run *IPFS* daemon to store the Model.

.. code-block:: shell

  $ ipfs daemon

Second, Run *Ganache*

.. code-block:: shell

  $ ganache-cli --port 7545 --networkId 5777 -a 31 -d

Last, Deploy smart contract

.. code-block:: shell

  $ cd ~/flowr/py/flwr/client/eth_client
  $ npm i # first time only
  $ truffle migrate --network development --reset

After deployment, you can get the result like below.

.. code-block:: shell

  2_deploy_contracts.js
  =====================
     Replacing 'Crowdsource'
     -----------------------
     > transaction hash:    0x5e05aae436f07591464a11a0ba301220575038e8ff77cb6705401d521940aa5c
     > Blocks: 0            Seconds: 0
     > contract address:    0xCfEB869F69431e42cdB54A4F4f105C19C080A601
     > block number:        3
     > block timestamp:     1682418499
     > account:             0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1
     > balance:             99.95304242
     > gas used:            2103520 (0x2018e0)
     > gas price:           20 gwei
     > value sent:          0 ETH
     > total cost:          0.0420704 ETH

If you modify ``CONTRACT_ADDRESS`` of ``~/flowr/src/py/flwr/client/eth_client/eth_client.py`` to the contract address, everything is ready.


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

Flower provides a convenience class called :code:`EthClient` which makes it
easier to implement the :code:`Client` interface when your workload uses PyTorch with Ethereum.
Implementing :code:`EthClient` usually means defining the following methods
(:code:`set_parameters` is optional though):

#. :code:`__init__`
    * set the client id(Cid)
    * set the initial model
    * upload initial model's architecture and initial model's weight.
#. :code:`get_parameters`
    * return the model weight as a list of NumPy ndarrays
#. :code:`set_parameters` (optional)
    * update the local model weights with the parameters received from the server
#. :code:`fit`
    * get the Global model's cid(contents id) from IPFS. And then, get Global model's weight from Ganache using cid
    * set the local model weights
    * train the local model
    * upload the updated local model to IPFS and get cid(contents id). And then, save local model's cid in Ganache.
    * receive the updated local model cid(contents id)
#. :code:`evaluate`
    * test the local model

which can be implemented in the following way:

.. code-block:: python

  class FlowerClient(fl.client.EthClient):
      def __init__(self,
                   cid: str,
                   ):
          super(FlowerClient, self).__init__(cid)

          self.net = net
          self.IPFSClient.set_model(net)
          self.initial_setting()


      def get_parameters(self, config):
          return [val.cpu().numpy() for _, val in net.state_dict().items()]

      def set_parameters(self, parameters):
          params_dict = zip(net.state_dict().keys(), parameters)
          state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
          net.load_state_dict(state_dict, strict=True)


      def fit(self, config):
          print("Client FIT @ eth_client")
          training_round = self.EthBase.currentRound()
          print("training_round", training_round)
          if training_round == 1:
              g_model_cid = self.EthBase.getGenesis()
          else:
              g_model_cid = self.EthBase.getGlobalmodel(training_round)
          print("g_model_cid", g_model_cid)
          net = self.IPFSClient.get_model(g_model_cid)
          # self.set_parameters(parameters)
          train(net, trainloader, epochs=1)
          print('after model train')
          uploaded_cid = self.IPFSClient.add_model(self.net)
          print('IPFS upload done',uploaded_cid)
          tx = self.EthBase.addModelUpdate(uploaded_cid, training_round)
          self.EthBase.wait_for_tx(tx)
          print('Add Model done')
          return [uploaded_cid], len(trainloader.dataset), {}

      def evaluate(self, parameters, config):
          self.set_parameters(parameters)
          loss, accuracy = test(net, testloader)
          return loss, len(testloader.dataset), {"accuracy": accuracy}

We can now create an instance of our class :code:`FlowerClient` and add one line
to actually run this client.
Each client must have a different CID, and at least one client's CID must be 0.
The reason why we are using port 8081 is that IPFS uses port 8080 as the default port. Therefore, to avoid conflicts, we are using a different port (8081) for our example.:

.. code-block:: python

     fl.client.start_eth_client(server_address="127.0.0.1:8081",client=FlowerClient(cid=0))

That's it for the client. We only have to implement :code:`EthClient` and call :code:`fl.client.start_eth_client()`. The string :code:`"[::]:8081"` tells the client which server to connect to. In our case we can run the server and the client on the same machine, therefore we use
:code:`"[::]:8081"`. If we run a truly federated workload with the server and
clients running on different machines, all that needs to change is the
:code:`server_address` we point the client at.

Flower Server
-------------

For simple workloads we can start a Flower server and leave all the
configuration possibilities at their default values. In a file named
:code:`server.py`, import Flower and start the server:

.. code-block:: python

    import flwr as fl
    from flwr.server.client_manager import SimpleClientManager
    from flwr.server.server import EthServer


    client_manager = SimpleClientManager()
    eth_server = EthServer(client_manager = client_manager, strategy = strategy)


    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8081", # server port is 8081, cause by ipfs address
        server = eth_server,
        config = fl.server.ServerConfig(num_rounds=11),
        strategy=strategy,
    )

Train the model, federated!
---------------------------

With both client and server ready, we can now run everything and see federated
learning in action. FL systems usually have a server and multiple clients. We
therefore have to start the server first:

.. code-block:: shell

    $ python server.py

Once the server is running we can start the clients in different terminals.
Open a new terminal and start the first client with cid 0:

.. code-block:: shell

    $ python client.py

Open another terminal and start the second client with cid 1:

.. code-block:: shell

    $ python client.py

Each client will have its own dataset.
You should now see how the training does in the very first terminal (the one that started the server):

.. code-block:: shell

    WARNING flwr 2023-05-02 11:59:26,413 | app.py:203 | Both server and strategy were provided, ignoring strategy
    INFO flwr 2023-05-02 11:59:26,413 | app.py:151 | Starting Flower server, config: ServerConfig(num_rounds=11, round_timeout=None)
    INFO flwr 2023-05-02 11:59:26,430 | app.py:172 | Flower ECE: gRPC server running (11 rounds), SSL is disabled
    INFO flwr 2023-05-02 11:59:26,431 | server.py:290 | FL starting
    DEBUG flwr 2023-05-02 11:59:34,965 | server.py:370 | fit_round 1: strategy sampled 2 clients (out of 2)
    DEBUG flwr 2023-05-02 11:59:49,740 | server.py:384 | fit_round 1 received 2 results and 0 failures
    warnings.warn(exceptions.VersionMismatch(version, minimum, maximum))
    download from IPFS
    ipfs connection done
    check cid QmP927FRvfkTzwYGSzPaDz4GBnnbhhJ1HbowPGoBcduuRX
    upload done
    download from IPFS
    ipfs connection done
    check cid QmQcdfz6V8gMeyc4UQXaCdoL7qe3KHp8jVTJR9UMcWgp3A
    upload done
    WARNING flwr 2023-05-02 11:59:50,923 | fedavg.py:243 | No fit_metrics_aggregation_fn provided
    DEBUG flwr 2023-05-02 11:59:51,426 | server.py:172 | evaluate_round 1: strategy sampled 2 clients (out of 2)
    20574.6796875 {'accuracy': 0.2558}
    DEBUG flwr 2023-05-02 12:00:02,920 | server.py:185 | evaluate_round 1 received 2 results and 0 failures
    DEBUG flwr 2023-05-02 12:00:02,921 | server.py:370 | fit_round 2: strategy sampled 2 clients (out of 2)
    DEBUG flwr 2023-05-02 12:00:14,984 | server.py:384 | fit_round 2 received 2 results and 0 failures
    download from IPFS
    ipfs connection done
    check cid QmdTi88rJfBnodkmXE9bRNmsPjzsdrCQq7CQiybcQjBUwC
    upload done
    download from IPFS
    ipfs connection done
    check cid QmRbkjBrJwB7teXkM7iBfzTLaZ9x3BR1rF1rASbVhn8TcC
    upload done
    DEBUG flwr 2023-05-02 12:00:15,532 | server.py:172 | evaluate_round 2: strategy sampled 2 clients (out of 2)
    DEBUG flwr 2023-05-02 12:00:26,954 | server.py:185 | evaluate_round 2 received 2 results and 0 failures
    DEBUG flwr 2023-05-02 12:00:26,955 | server.py:370 | fit_round 3: strategy sampled 2 clients (out of 2)
    16749.130859375 {'accuracy': 0.3897}
    ...
    ...
    ...
    download from IPFS
    ipfs connection done
    check cid QmPVxGxF3o1PoxBPePnVK5gVv9i84Ap6zR5UYYcmbhSYjQ
    upload done
    download from IPFS
    ipfs connection done
    check cid QmUaRCQ75qHimf1fYbLKqfArdiWLX4EesuTzFXwEoCE5eS
    upload done
    DEBUG flwr 2023-05-02 12:03:52,972 | server.py:172 | evaluate_round 11: strategy sampled 2 clients (out of 2)
    11215.646484375 {'accuracy': 0.6027}
    DEBUG flwr 2023-05-02 12:04:04,397 | server.py:185 | evaluate_round 11 received 2 results and 0 failures
    INFO flwr 2023-05-02 12:04:04,398 | server.py:345 | FL finished in 277.96658609691076
    INFO flwr 2023-05-02 12:04:04,398 | app.py:218 | app_fit: losses_distributed [(1, 20574.6796875), (2, 16749.130859375), (3, 15086.33203125), (4, 14205.916015625), (5, 13493.5), (6, 12918.087890625), (7, 12538.3408203125), (8, 11985.8984375), (9, 11636.724609375), (10, 11441.1484375), (11, 11215.646484375)]
    INFO flwr 2023-05-02 12:04:04,398 | app.py:219 | app_fit: metrics_distributed_fit {}
    INFO flwr 2023-05-02 12:04:04,398 | app.py:220 | app_fit: metrics_distributed {'accuracy': [(1, 0.2558), (2, 0.3897), (3, 0.4479), (4, 0.4836), (5, 0.5146), (6, 0.5334), (7, 0.549), (8, 0.5715), (9, 0.5867), (10, 0.5969), (11, 0.6027)]}
    INFO flwr 2023-05-02 12:04:04,399 | app.py:221 | app_fit: losses_centralized []
    INFO flwr 2023-05-02 12:04:04,399 | app.py:222 | app_fit: metrics_centralized {}

Congratulations!
You've successfully built and run your first Blockchain based federated learning system.
The full `source code <https://github.com/adap/flower/blob/main/examples/quickstart_pytorch_ethereum/client.py>`_ for this example can be found in :code:`examples/quickstart_pytorch_ethereum`.
