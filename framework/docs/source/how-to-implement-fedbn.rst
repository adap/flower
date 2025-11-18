:og:description: Learn to build a federated ML workload using Flower with FedBN for non-IID data. Train a CNN with PyTorch on CIFAR-10 with minimal changes from the Quickstart.
.. meta::
    :description: Learn to build a federated ML workload using Flower with FedBN for non-IID data. Train a CNN with PyTorch on CIFAR-10 with minimal changes from the Quickstart.

#################
 Implement FedBN
#################

This tutorial will show you how to use Flower to build a federated version of an
existing machine learning workload with `FedBN <https://github.com/med-air/FedBN>`_, a
federated training method designed for non-IID data. We are using PyTorch to train a
Convolutional Neural Network (with Batch Normalization layers) on the CIFAR-10 dataset.
When applying FedBN, only minor changes are needed compared to :doc:`Quickstart PyTorch
<tutorial-quickstart-pytorch>`.

*******
 Model
*******

A full introduction to federated learning with PyTorch and Flower can be found in
:doc:`Quickstart PyTorch <tutorial-quickstart-pytorch>`. This how-to guide changes only
a few details in ``task.py``. FedBN requires a model architecture (defined in class
``Net()``) that uses Batch Normalization layers:

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

Try editing the model architecture, then run the project to ensure everything still
works:

.. code-block:: bash

    flwr run .

So far this should all look fairly familiar if you've used Flower with PyTorch before.

*******
 FedBN
*******

To adopt FedBN, we revise the ``train`` method in ``ClientApp``. The batch normalization
parameters are excluded from model state dict when sending to or receiving from the
``ServerApp``:

.. code-block:: python

    app = ClientApp()


    @app.train()
    def train(msg: Message, context: Context):
        """Train the model on local data."""

        # Load the model and initialize it with the received weights
        model = Net()
        state_dict = msg.content["arrays"].to_torch_state_dict()

        # Exclude batch normalization parameters
        state_dict_wo_bn = OrderedDict(
            (k, v) for k, v in state_dict.items() if "bn" not in k
        )
        model.load_state_dict(state_dict_wo_bn, strict=True)

        # ... [Perform training]

        # Construct and return reply Message
        state_dict_wo_bn = OrderedDict(
            (k, v) for k, v in model.state_dict().items() if "bn" not in k
        )
        model_record = ArrayRecord(state_dict_wo_bn)

        ...

To test the new approach, run the project again:

.. code-block:: bash

    flwr run .

Your PyTorch project now runs federated learning with FedBN. Congratulations!

************
 Next Steps
************

The example is certainly over-simplified since all ``ClientApp``\s load the exact same
dataset. This isn't realistic. You now have the tools to explore this topic further. How
about using different subsets of CIFAR-10 on each client? How about adding more clients?
