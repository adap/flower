:og:description: Learn how to train a Convolutional Neural Network on CIFAR-10 using federated learning with Flower and PyTorch in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a Convolutional Neural Network on CIFAR-10 using federated learning with Flower and PyTorch in this step-by-step tutorial.

.. _quickstart-pytorch:

Quickstart PyTorch
==================

In this federated learning tutorial we will learn how to train a Convolutional Neural
Network on CIFAR-10 using Flower and PyTorch. It is recommended to create a virtual
environment and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

Let's use `flwr new` to create a complete Flower+PyTorch project. It will generate all
the files needed to run, by default with the Flower Simulation Engine, a federation of
10 nodes using `FedAvg
<https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvg.html#flwr.server.strategy.FedAvg>`_.
The dataset will be partitioned using Flower Dataset's `IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_.

Now that we have a rough idea of what this example is about, let's get started. First,
install Flower in your new environment:

.. code-block:: shell

    # In a new Python environment
    $ pip install flwr

Then, run the command below. You will be prompted to select one of the available
templates (choose ``PyTorch``), give a name to your project, and type in your developer
name:

.. code-block:: shell

    $ flwr new

After running it you'll notice a new directory with your project name has been created.
It should have the following structure:

.. code-block:: shell

    <your-project-name>
    ├── <your-project-name>
    │   ├── __init__.py
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

If you haven't yet installed the project and its dependencies, you can do so by:

.. code-block:: shell

    # From the directory where your pyproject.toml is
    $ pip install -e .

To run the project, do:

.. code-block:: shell

    # Run with default arguments
    $ flwr run .

With default arguments you will see an output like this one:

.. code-block:: shell

    Loading project configuration...
    Success
    WARNING :   FAB ID is not provided; the default ClientApp will be loaded.
    INFO :      Starting Flower ServerApp, config: num_rounds=3, no round_timeout
    INFO :
    INFO :      [INIT]
    INFO :      Using initial global parameters provided by strategy
    INFO :      Evaluating initial global parameters
    INFO :
    INFO :      [ROUND 1]
    INFO :      configure_fit: strategy sampled 5 clients (out of 10)
    INFO :      aggregate_fit: received 5 results and 0 failures
    WARNING :   No fit_metrics_aggregation_fn provided
    INFO :      configure_evaluate: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_evaluate: received 10 results and 0 failures
    WARNING :   No evaluate_metrics_aggregation_fn provided
    INFO :
    INFO :      [ROUND 2]
    INFO :      configure_fit: strategy sampled 5 clients (out of 10)
    INFO :      aggregate_fit: received 5 results and 0 failures
    INFO :      configure_evaluate: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_evaluate: received 10 results and 0 failures
    INFO :
    INFO :      [ROUND 3]
    INFO :      configure_fit: strategy sampled 5 clients (out of 10)
    INFO :      aggregate_fit: received 5 results and 0 failures
    INFO :      configure_evaluate: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_evaluate: received 10 results and 0 failures
    INFO :
    INFO :      [SUMMARY]
    INFO :      Run finished 3 round(s) in 21.35s
    INFO :          History (loss, distributed):
    INFO :                  round 1: 2.2978184528648855
    INFO :                  round 2: 2.173852103948593
    INFO :                  round 3: 2.039920600131154
    INFO :

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config "num-server-rounds=5 local-epochs=3"

What follows is an explanation of each component in the project you just created:
dataset partition, the model, defining the ``ClientApp`` and defining the ``ServerApp``.

The Data
--------

This tutorial uses `Flower Datasets <https://flower.ai/docs/datasets/>`_ to easily
download and partition the `CIFAR-10` dataset. In this example you'll make use of the
`IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_
to generate `num_partitions` partitions. You can choose `other partitioners
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html>`_ available in
Flower Datasets. Each ``ClientApp`` will call this function to create dataloaders with
the data that correspond to their data partition.

.. code-block:: python

    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch


    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)

The Model
---------

We defined a simple Convolutional Neural Network (CNN), but feel free to replace it with
a more sophisticated model if you'd like:

.. code-block:: python

    class Net(nn.Module):
        """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

In addition to defining the model architecture, we also include two utility functions to
perform both training (i.e. ``train()``) and evaluation (i.e. ``test()``) using the
above model. These functions should look fairly familiar if you have some prior
experience with PyTorch. Note these functions do not have anything specific to Flower.
That being said, the training function will normally be called, as we'll see later, from
a Flower client passing its own data. In summary, your clients can use standard
training/testing functions to perform local training or evaluation:

.. code-block:: python

    def train(net, trainloader, epochs, device):
        """Train the model on the training set."""
        net.to(device)  # move model to GPU if available
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
        net.train()
        running_loss = 0.0
        for _ in range(epochs):
            for batch in trainloader:
                images = batch["img"]
                labels = batch["label"]
                optimizer.zero_grad()
                loss = criterion(net(images.to(device)), labels.to(device))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        avg_trainloss = running_loss / len(trainloader)
        return avg_trainloss


    def test(net, testloader, device):
        """Validate the model on the test set."""
        net.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for batch in testloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        return loss, accuracy

The ClientApp
-------------

The main changes we have to make to use `PyTorch` with `Flower` will be found in the
``get_weights()`` and ``set_weights()`` functions. In ``get_weights()`` PyTorch model
parameters are extracted and represented as a list of NumPy arrays. The
``set_weights()`` function that's the oposite: given a list of NumPy arrays it applies
them to an existing PyTorch model. Doing this in fairly easy in PyTorch.

.. note::

    The specific implementation of ``get_weights()`` and ``set_weights()`` depends on
    the type of models you use. The ones shown below work for a wide range of PyTorch
    models but you might need to adjust them if you have more exotic model
    architectures.

.. code-block:: python

    def get_weights(net):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]


    def set_weights(net, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

The rest of the functionality is directly inspired by the centralized case. The
``fit()`` method in the client trains the model using the local dataset. Similarly, the
``evaluate()`` method is used to evaluate the model received on a held-out validation
set that the client might have:

.. code-block:: python

    class FlowerClient(NumPyClient):
        def __init__(self, net, trainloader, valloader, local_epochs):
            self.net = net
            self.trainloader = trainloader
            self.valloader = valloader
            self.local_epochs = local_epochs
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.to(device)

        def fit(self, parameters, config):
            set_weights(self.net, parameters)
            results = train(
                self.net,
                self.trainloader,
                self.valloader,
                self.local_epochs,
                self.device,
            )
            return get_weights(self.net), len(self.trainloader.dataset), results

        def evaluate(self, parameters, config):
            set_weights(self.net, parameters)
            loss, accuracy = test(self.net, self.valloader, self.device)
            return loss, len(self.valloader.dataset), {"accuracy": accuracy}

Finally, we can construct a ``ClientApp`` using the ``FlowerClient`` defined above by
means of a ``client_fn()`` callback. Note that the `context` enables you to get access
to hyperparemeters defined in your ``pyproject.toml`` to configure the run. In this
tutorial we access the `local-epochs` setting to control the number of epochs a
``ClientApp`` will perform when running the ``fit()`` method. You could define
additioinal hyperparameters in ``pyproject.toml`` and access them here.

.. code-block:: python

    def client_fn(context: Context):
        # Load model and data
        net = Net()
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        trainloader, valloader = load_data(partition_id, num_partitions)
        local_epochs = context.run_config["local-epochs"]

        # Return Client instance
        return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


    # Flower ClientApp
    app = ClientApp(client_fn)

The ServerApp
-------------

To construct a ``ServerApp`` we define a ``server_fn()`` callback with an identical
signature to that of ``client_fn()`` but the return type is `ServerAppComponents
<https://flower.ai/docs/framework/ref-api/flwr.server.ServerAppComponents.html#serverappcomponents>`_
as opposed to a `Client
<https://flower.ai/docs/framework/ref-api/flwr.client.Client.html#client>`_. In this
example we use the `FedAvg`. To it we pass a randomly initialized model that will server
as the global model to federated. Note that the value of ``fraction_fit`` is read from
the run config. You can find the default value defined in the ``pyproject.toml``.

.. code-block:: python

    def server_fn(context: Context):
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]
        fraction_fit = context.run_config["fraction-fit"]

        # Initialize model parameters
        ndarrays = get_weights(Net())
        parameters = ndarrays_to_parameters(ndarrays)

        # Define strategy
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

Congratulations! You've successfully built and run your first federated learning system.

.. note::

    Check the `source code
    <https://github.com/adap/flower/blob/main/examples/quickstart-pytorch>`_ of the
    extended version of this tutorial in ``examples/quickstart-pytorch`` in the Flower
    GitHub repository.
