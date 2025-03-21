Get started with Flower
=======================

Welcome to the Flower federated learning tutorial!


In this tutorial, we’ll build a federated learning system using the
Flower framework, Flower Datasets and PyTorch. In part 1, we use PyTorch
for the model training pipeline and data loading. In part 2, we federate
the PyTorch project using Flower.

   `Star Flower on GitHub <https://github.com/adap/flower>`__ ⭐️ and
   join the Flower community on Flower Discuss and the Flower Slack to
   connect, ask questions, and get help: - `Join Flower
   Discuss <https://discuss.flower.ai/>`__ We’d love to hear from you in
   the ``Introduction`` topic! If anything is unclear, post in
   ``Flower Help - Beginners``. - `Join Flower
   Slack <https://flower.ai/join-slack>`__ We’d love to hear from you in
   the ``#introductions`` channel! If anything is unclear, head over to
   the ``#questions`` channel.

Let’s get started! 🌼

Step 0: Preparation
-------------------

Before we begin with any actual code, let’s make sure that we have
everything we need.

Install dependencies
~~~~~~~~~~~~~~~~~~~~

First, we install the Flower package ``flwr``:

.. code-block:: shell

    # In a new Python environment
    $ pip install -U flwr

Then, we create a new Flower app called ``flower-tutorial`` using the PyTorch template.
We also specify a username (``flwrlabs``) for the project:

.. code-block:: shell

    $ flwr new flower-tutorial --framework pytorch --username flwrlabs

After running the command, a new directory called ``flower-tutorial`` will be created.
It should have the following structure:

.. code-block:: shell

    ├── README.md
    ├── flower_tutorial
    │   ├── __init__.py
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

Next, we install the project and its dependencies, which are specified in the
``pyproject.toml`` file. For this tutorial, we'll also need ``matplotlib``, so we'll 
also install it:

.. code-block:: shell
  
    $ cd flower-tutorial
    $ pip install -e . matplotlib

Before we dive into federated learning, we'll take a look at the dataset that we'll be
using for this tutorial, which is the
`CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset, and run a simple
centralized training pipeline using PyTorch.


The ``CIFAR-10`` dataset
~~~~~~~~~~~~~~~~~~~~~~~~

Federated learning can be applied to many different types of tasks
across different domains. In this tutorial, we introduce federated
learning by training a simple convolutional neural network (CNN) on the
popular CIFAR-10 dataset. CIFAR-10 can be used to train image
classifiers that distinguish between images from ten different classes:
‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’,
‘ship’, and ‘truck’.

We simulate having multiple datasets from multiple organizations (also
called the “cross-silo” setting in federated learning) by splitting the
original CIFAR-10 dataset into multiple partitions. Each partition will
represent the data from a single organization. We’re doing this purely
for experimentation purposes, in the real world there’s no need for data
splitting because each organization already has their own data (the data
is naturally partitioned).

Each organization will act as a client in the federated learning system.
Having ten organizations participate in a federation means having ten
clients connected to the federated learning server.

We use the Flower Datasets library (``flwr-datasets``) to partition
CIFAR-10 into ten partitions using ``FederatedDataset``. We will create
a small training and test set for each of the ten organizations and wrap
each of these into a PyTorch ``DataLoader``:

.. code:: 

    NUM_CLIENTS = 10
    BATCH_SIZE = 32
    
    
    def load_datasets(partition_id: int):
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
        partition = fds.load_partition(partition_id)
        # Divide data on each node: 80% train, 20% test
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        pytorch_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    
        def apply_transforms(batch):
            # Instead of passing transforms to CIFAR10(..., transform=transform)
            # we will use this function to dataset.with_transform(apply_transforms)
            # The transforms object is exactly the same
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch
    
        # Create train/val for each partition and wrap it into DataLoader
        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
        )
        valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
        testset = fds.load_split("test").with_transform(apply_transforms)
        testloader = DataLoader(testset, batch_size=BATCH_SIZE)
        return trainloader, valloader, testloader

We now have a function that can return a training set and validation set
(``trainloader`` and ``valloader``) representing one dataset from one of
ten different organizations. Each ``trainloader``/``valloader`` pair
contains 4000 training examples and 1000 validation examples. There’s
also a single ``testloader`` (we did not split the test set). Again,
this is only necessary for building research or educational systems,
actual federated learning systems have their data naturally distributed
across multiple partitions.

Let’s take a look at the first batch of images and labels in the first
training set (i.e., ``trainloader`` from ``partition_id=0``) before we
move on:

.. code:: 

    trainloader, _, _ = load_datasets(partition_id=0)
    batch = next(iter(trainloader))
    images, labels = batch["img"], batch["label"]
    
    # Reshape and convert images to a NumPy array
    # matplotlib requires images with the shape (height, width, 3)
    images = images.permute(0, 2, 3, 1).numpy()
    
    # Denormalize
    images = images / 2 + 0.5
    
    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(4, 8, figsize=(12, 6))
    
    # Loop over the images and plot them
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.set_title(trainloader.dataset.features["label"].int2str([labels[i]])[0])
        ax.axis("off")
    
    # Show the plot
    fig.tight_layout()
    plt.show()

The output above shows a random batch of images from the ``trainloader``
from the first of ten partitions. It also prints the labels associated
with each image (i.e., one of the ten possible labels we’ve seen above).
If you run the cell again, you should see another batch of images.

Step 1: Centralized Training with PyTorch
-----------------------------------------

Next, we’re going to use PyTorch to define a simple convolutional neural
network. This introduction assumes basic familiarity with PyTorch, so it
doesn’t cover the PyTorch-related aspects in full detail. If you want to
dive deeper into PyTorch, we recommend `DEEP LEARNING WITH PYTORCH: A 60
MINUTE
BLITZ <https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>`__.

Define the model
~~~~~~~~~~~~~~~~

We use the simple CNN described in the `PyTorch
tutorial <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network>`__:

.. code:: 

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

Let’s continue with the usual training and test functions:

.. code:: 

    def train(net, trainloader, epochs: int, verbose=False):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters())
        net.train()
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in trainloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            epoch_loss /= len(trainloader.dataset)
            epoch_acc = correct / total
            if verbose:
                print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    
    
    def test(net, testloader):
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        net.eval()
        with torch.no_grad():
            for batch in testloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss /= len(testloader.dataset)
        accuracy = correct / total
        return loss, accuracy

Train the model
~~~~~~~~~~~~~~~

We now have all the basic building blocks we need: a dataset, a model, a
training function, and a test function. Let’s put them together to train
the model on the dataset of one of our organizations
(``partition_id=0``). This simulates the reality of most machine
learning projects today: each organization has their own data and trains
models only on this internal data:

.. code:: 

    trainloader, valloader, testloader = load_datasets(partition_id=0)
    net = Net().to(DEVICE)
    
    for epoch in range(5):
        train(net, trainloader, 1)
        loss, accuracy = test(net, valloader)
        print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")
    
    loss, accuracy = test(net, testloader)
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

Training the simple CNN on our CIFAR-10 split for 5 epochs should result
in a test set accuracy of about 41%, which is not good, but at the same
time, it doesn’t really matter for the purposes of this tutorial. The
intent was just to show a simple centralized training pipeline that sets
the stage for what comes next - federated learning!

Step 2: Federated Learning with Flower
--------------------------------------

Step 1 demonstrated a simple centralized training pipeline. All data was
in one place (i.e., a single ``trainloader`` and a single
``valloader``). Next, we’ll simulate a situation where we have multiple
datasets in multiple organizations and where we train a model over these
organizations using federated learning.

Update model parameters
~~~~~~~~~~~~~~~~~~~~~~~

In federated learning, the server sends global model parameters to the
client, and the client updates the local model with parameters received
from the server. It then trains the model on the local data (which
changes the model parameters locally) and sends the updated/changed
model parameters back to the server (or, alternatively, it sends just
the gradients back to the server, not the full model parameters).

We need two helper functions to update the local model with parameters
received from the server and to get the updated model parameters from
the local model: ``set_parameters`` and ``get_parameters``. The
following two functions do just that for the PyTorch model above.

The details of how this works are not really important here (feel free
to consult the PyTorch documentation if you want to learn more). In
essence, we use ``state_dict`` to access PyTorch model parameter
tensors. The parameter tensors are then converted to/from a list of
NumPy ndarray’s (which the Flower ``NumPyClient`` knows how to
serialize/deserialize):

.. code:: 

    def set_parameters(net, parameters: List[np.ndarray]):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
    
    
    def get_parameters(net) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

Define the Flower ClientApp
~~~~~~~~~~~~~~~~~~~~~~~~~~~

With that out of the way, let’s move on to the interesting part.
Federated learning systems consist of a server and multiple clients. In
Flower, we create a ``ServerApp`` and a ``ClientApp`` to run the
server-side and client-side code, respectively.

The first step toward creating a ``ClientApp`` is to implement a
subclasses of ``flwr.client.Client`` or ``flwr.client.NumPyClient``. We
use ``NumPyClient`` in this tutorial because it is easier to implement
and requires us to write less boilerplate. To implement ``NumPyClient``,
we create a subclass that implements the three methods
``get_parameters``, ``fit``, and ``evaluate``:

-  ``get_parameters``: Return the current local model parameters
-  ``fit``: Receive model parameters from the server, train the model on
   the local data, and return the updated model parameters to the server
-  ``evaluate``: Receive model parameters from the server, evaluate the
   model on the local data, and return the evaluation result to the
   server

We mentioned that our clients will use the previously defined PyTorch
components for model training and evaluation. Let’s see a simple Flower
client implementation that brings everything together:

.. code:: 

    class FlowerClient(NumPyClient):
        def __init__(self, net, trainloader, valloader):
            self.net = net
            self.trainloader = trainloader
            self.valloader = valloader
    
        def get_parameters(self, config):
            return get_parameters(self.net)
    
        def fit(self, parameters, config):
            set_parameters(self.net, parameters)
            train(self.net, self.trainloader, epochs=1)
            return get_parameters(self.net), len(self.trainloader), {}
    
        def evaluate(self, parameters, config):
            set_parameters(self.net, parameters)
            loss, accuracy = test(self.net, self.valloader)
            return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

Our class ``FlowerClient`` defines how local training/evaluation will be
performed and allows Flower to call the local training/evaluation
through ``fit`` and ``evaluate``. Each instance of ``FlowerClient``
represents a *single client* in our federated learning system. Federated
learning systems have multiple clients (otherwise, there’s not much to
federate), so each client will be represented by its own instance of
``FlowerClient``. If we have, for example, three clients in our
workload, then we’d have three instances of ``FlowerClient`` (one on
each of the machines we’d start the client on). Flower calls
``FlowerClient.fit`` on the respective instance when the server selects
a particular client for training (and ``FlowerClient.evaluate`` for
evaluation).

In this notebook, we want to simulate a federated learning system with
10 clients *on a single machine*. This means that the server and all 10
clients will live on a single machine and share resources such as CPU,
GPU, and memory. Having 10 clients would mean having 10 instances of
``FlowerClient`` in memory. Doing this on a single machine can quickly
exhaust the available memory resources, even if only a subset of these
clients participates in a single round of federated learning.

In addition to the regular capabilities where server and clients run on
multiple machines, Flower, therefore, provides special simulation
capabilities that create ``FlowerClient`` instances only when they are
actually necessary for training or evaluation. To enable the Flower
framework to create clients when necessary, we need to implement a
function that creates a ``FlowerClient`` instance on demand. We
typically call this function ``client_fn``. Flower calls ``client_fn``
whenever it needs an instance of one particular client to call ``fit``
or ``evaluate`` (those instances are usually discarded after use, so
they should not keep any local state). In federated learning experiments
using Flower, clients are identified by a partition ID, or
``partition-id``. This ``partition-id`` is used to load different local
data partitions for different clients, as can be seen below. The value
of ``partition-id`` is retrieved from the ``node_config`` dictionary in
the ``Context`` object, which holds the information that persists
throughout each training round.

With this, we have the class ``FlowerClient`` which defines client-side
training/evaluation and ``client_fn`` which allows Flower to create
``FlowerClient`` instances whenever it needs to call ``fit`` or
``evaluate`` on one particular client. Last, but definitely not least,
we create an instance of ``ClientApp`` and pass it the ``client_fn``.
``ClientApp`` is the entrypoint that a running Flower client uses to
call your code (as defined in, for example, ``FlowerClient.fit``).

.. code:: 

    def client_fn(context: Context) -> Client:
        """Create a Flower client representing a single organization."""
    
        # Load model
        net = Net().to(DEVICE)
    
        # Load data (CIFAR-10)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data partition
        # Read the node_config to fetch data partition associated to this node
        partition_id = context.node_config["partition-id"]
        trainloader, valloader, _ = load_datasets(partition_id=partition_id)
    
        # Create a single Flower client representing a single organization
        # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
        # to convert it to a subclass of `flwr.client.Client`
        return FlowerClient(net, trainloader, valloader).to_client()
    
    
    # Create the ClientApp
    client = ClientApp(client_fn=client_fn)

Define the Flower ServerApp
~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the server side, we need to configure a strategy which encapsulates
the federated learning approach/algorithm, for example, *Federated
Averaging* (FedAvg). Flower has a number of built-in strategies, but we
can also use our own strategy implementations to customize nearly all
aspects of the federated learning approach. For this example, we use the
built-in ``FedAvg`` implementation and customize it using a few basic
parameters:

.. code:: 

    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
        min_available_clients=10,  # Wait until all 10 clients are available
    )

Similar to ``ClientApp``, we create a ``ServerApp`` using a utility
function ``server_fn``. In ``server_fn``, we pass an instance of
``ServerConfig`` for defining the number of federated learning rounds
(``num_rounds``) and we also pass the previously created ``strategy``.
The ``server_fn`` returns a ``ServerAppComponents`` object containing
the settings that define the ``ServerApp`` behaviour. ``ServerApp`` is
the entrypoint that Flower uses to call all your server-side code (for
example, the strategy).

.. code:: 

    def server_fn(context: Context) -> ServerAppComponents:
        """Construct components that set the ServerApp behaviour.
    
        You can use the settings in `context.run_config` to parameterize the
        construction of all elements (e.g the strategy or the number of rounds)
        wrapped in the returned ServerAppComponents object.
        """
    
        # Configure the server for 5 rounds of training
        config = ServerConfig(num_rounds=5)
    
        return ServerAppComponents(strategy=strategy, config=config)
    
    
    # Create the ServerApp
    server = ServerApp(server_fn=server_fn)

Run the training
~~~~~~~~~~~~~~~~

In simulation, we often want to control the amount of resources each
client can use. In the next cell, we specify a ``backend_config``
dictionary with the ``client_resources`` key (required) for defining the
amount of CPU and GPU resources each client can access.

.. code:: 

    # Specify the resources each of your clients need
    # By default, each client will be allocated 1x CPU and 0x GPUs
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    
    # When running on GPU, assign an entire GPU for each client
    if DEVICE == "cuda":
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
        # Refer to our Flower framework documentation for more details about Flower simulations
        # and how to set up the `backend_config`

The last step is the actual call to ``run_simulation`` which - you
guessed it - runs the simulation. ``run_simulation`` accepts a number of
arguments: - ``server_app`` and ``client_app``: the previously created
``ServerApp`` and ``ClientApp`` objects, respectively -
``num_supernodes``: the number of ``SuperNodes`` to simulate which
equals the number of clients for Flower simulation - ``backend_config``:
the resource allocation used in this simulation

.. code:: 

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )

Behind the scenes
~~~~~~~~~~~~~~~~~

So how does this work? How does Flower execute this simulation?

When we call ``run_simulation``, we tell Flower that there are 10
clients (``num_supernodes=10``, where 1 ``SuperNode`` launches 1
``ClientApp``). Flower then goes ahead an asks the ``ServerApp`` to
issue an instructions to those nodes using the ``FedAvg`` strategy.
``FedAvg`` knows that it should select 100% of the available clients
(``fraction_fit=1.0``), so it goes ahead and selects 10 random clients
(i.e., 100% of 10).

Flower then asks the selected 10 clients to train the model. Each of the
10 ``ClientApp`` instances receives a message, which causes it to call
``client_fn`` to create an instance of ``FlowerClient``. It then calls
``.fit()`` on each the ``FlowerClient`` instances and returns the
resulting model parameter updates to the ``ServerApp``. When the
``ServerApp`` receives the model parameter updates from the clients, it
hands those updates over to the strategy (*FedAvg*) for aggregation. The
strategy aggregates those updates and returns the new global model,
which then gets used in the next round of federated learning.

Where’s the accuracy?
~~~~~~~~~~~~~~~~~~~~~

You may have noticed that all metrics except for ``losses_distributed``
are empty. Where did the ``{"accuracy": float(accuracy)}`` go?

Flower can automatically aggregate losses returned by individual
clients, but it cannot do the same for metrics in the generic metrics
dictionary (the one with the ``accuracy`` key). Metrics dictionaries can
contain very different kinds of metrics and even key/value pairs that
are not metrics at all, so the framework does not (and can not) know how
to handle these automatically.

As users, we need to tell the framework how to handle/aggregate these
custom metrics, and we do so by passing metric aggregation functions to
the strategy. The strategy will then call these functions whenever it
receives fit or evaluate metrics from clients. The two possible
functions are ``fit_metrics_aggregation_fn`` and
``evaluate_metrics_aggregation_fn``.

Let’s create a simple weighted averaging function to aggregate the
``accuracy`` metric we return from ``evaluate``:

.. code:: 

    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
    
        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}

.. code:: 

    def server_fn(context: Context) -> ServerAppComponents:
        """Construct components that set the ServerApp behaviour.
    
        You can use settings in `context.run_config` to parameterize the
        construction of all elements (e.g the strategy or the number of rounds)
        wrapped in the returned ServerAppComponents object.
        """
    
        # Create FedAvg strategy
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.5,
            min_fit_clients=10,
            min_evaluate_clients=5,
            min_available_clients=10,
            evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        )
    
        # Configure the server for 5 rounds of training
        config = ServerConfig(num_rounds=5)
    
        return ServerAppComponents(strategy=strategy, config=config)
    
    
    # Create a new server instance with the updated FedAvg strategy
    server = ServerApp(server_fn=server_fn)
    
    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )

We now have a full system that performs federated training and federated
evaluation. It uses the ``weighted_average`` function to aggregate
custom evaluation metrics and calculates a single ``accuracy`` metric
across all clients on the server side.

The other two categories of metrics (``losses_centralized`` and
``metrics_centralized``) are still empty because they only apply when
centralized evaluation is being used. Part two of the Flower tutorial
will cover centralized evaluation.

Final remarks
-------------

Congratulations, you just trained a convolutional neural network,
federated over 10 clients! With that, you understand the basics of
federated learning with Flower. The same approach you’ve seen can be
used with other machine learning frameworks (not just PyTorch) and tasks
(not just CIFAR-10 images classification), for example NLP with Hugging
Face Transformers or speech with SpeechBrain.

In the next notebook, we’re going to cover some more advanced concepts.
Want to customize your strategy? Initialize parameters on the server
side? Or evaluate the aggregated model on the server side? We’ll cover
all this and more in the next tutorial.

Next steps
----------

Before you continue, make sure to join the Flower community on Flower
Discuss (`Join Flower Discuss <https://discuss.flower.ai>`__) and on
Slack (`Join Slack <https://flower.ai/join-slack/>`__).

There’s a dedicated ``#questions`` channel if you need help, but we’d
also love to hear who you are in ``#introductions``!

The `Flower Federated Learning Tutorial - Part
2 <https://flower.ai/docs/framework/tutorial-use-a-federated-learning-strategy-pytorch.html>`__
goes into more depth about strategies and all the advanced things you
can build with them.
