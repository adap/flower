Example: FedBN in PyTorch - From Centralized To Federated
=========================================================

This tutorial will show you how to use Flower to build a federated version of an existing machine learning workload with `FedBN <https://github.com/med-air/FedBN>`_, a federated training strategy designed for non-iid data.
We are using PyTorch to train a Convolutional Neural Network(with Batch Normalization layers) on the CIFAR-10 dataset.
When applying FedBN, only few changes needed compared to :doc:`Example: PyTorch - From Centralized To Federated <example-pytorch-from-centralized-to-federated>`.

Centralized Training
--------------------
All files are revised based on :doc:`Example: PyTorch - From Centralized To Federated <example-pytorch-from-centralized-to-federated>`.
The only thing to do is modifying the file called :code:`cifar.py`, revised part is shown below:

The model architecture defined in class Net() is added with Batch Normalization layers accordingly.

.. code-block:: python

    class Net(nn.Module):

        def __init__(self) -> None:
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.bn1 = nn.BatchNorm2d(6)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.bn2 = nn.BatchNorm2d(16)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.bn3 = nn.BatchNorm1d(120)
            self.fc2 = nn.Linear(120, 84)
            self.bn4 = nn.BatchNorm1d(84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x: Tensor) -> Tensor:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.bn3(self.fc1(x)))
            x = F.relu(self.bn4(self.fc2(x)))
            x = self.fc3(x)
            return x

You can now run your machine learning workload:

.. code-block:: python

    python3 cifar.py

So far this should all look fairly familiar if you've used PyTorch before.
Let's take the next step and use what we've built to create a federated learning system within FedBN, the system consists of one server and two clients.

Federated Training
------------------

If you have read :doc:`Example: PyTorch - From Centralized To Federated <example-pytorch-from-centralized-to-federated>`, the following parts are easy to follow, only :code:`get_parameters` and :code:`set_parameters` function in :code:`client.py` needed to revise.
If not, please read the :doc:`Example: PyTorch - From Centralized To Federated <example-pytorch-from-centralized-to-federated>`. first.

Our example consists of one *server* and two *clients*. In FedBN, :code:`server.py` keeps unchanged, we can start the server directly.

.. code-block:: python

    python3 server.py

Finally, we will revise our *client* logic by changing :code:`get_parameters` and :code:`set_parameters` in :code:`client.py`, we will exclude batch normalization parameters from model parameter list when sending to or receiving from the server.

.. code-block:: python

    class CifarClient(fl.client.NumPyClient):
        """Flower client implementing CIFAR-10 image classification using
        PyTorch."""

        ...

        def get_parameters(self, config) -> List[np.ndarray]:
            # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
            return [val.cpu().numpy() for name, val in self.model.state_dict().items() if 'bn' not in name]

        def set_parameters(self, parameters: List[np.ndarray]) -> None:
            # Set model parameters from a list of NumPy ndarrays
            keys = [k for k in self.model.state_dict().keys() if 'bn' not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)

        ...

Now, you can now open two additional terminal windows and run

.. code-block:: python

    python3 client.py

in each window (make sure that the server is still running before you do so) and see your (previously centralized) PyTorch project run federated learning with FedBN strategy across two clients. Congratulations!

Next Steps
----------

The full source code for this example can be found `here <https://github.com/adap/flower/blob/main/examples/pytorch-from-centralized-to-federated>`_.
Our example is of course somewhat over-simplified because both clients load the exact same dataset, which isn't realistic.
You're now prepared to explore this topic further. How about using different subsets of CIFAR-10 on each client? How about adding more clients?
