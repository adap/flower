Get started with Flower
=======================

Welcome to the Flower federated learning tutorial!

In this tutorial, we’ll build a federated learning system using the Flower framework,
Flower Datasets and PyTorch.

    `Star Flower on GitHub <https://github.com/adap/flower>`__ ⭐️ and join the Flower
    community on Flower Discuss and the Flower Slack to connect, ask questions, and get
    help: - `Join Flower Discuss <https://discuss.flower.ai/>`__ We’d love to hear from
    you in the ``Introduction`` topic! If anything is unclear, post in ``Flower Help -
    Beginners``. - `Join Flower Slack <https://flower.ai/join-slack>`__ We’d love to
    hear from you in the ``#introductions`` channel! If anything is unclear, head over
    to the ``#questions`` channel.

Let’s get started! 🌼

Preparation
-----------

Before we begin with any actual code, let’s make sure that we have everything we need.

First, in a new Python environmentwe install the Flower package ``flwr``:

.. code-block:: shell

    # In a new Python environment
    $ pip install -U "flwr[simulation]"

Then, we create a new Flower app called ``flower-tutorial`` using the PyTorch template.
We also specify a username (``flwrlabs``) for the project or pass your username after
registering at `flower.ai <https://flower.ai/>`_.

.. code-block:: shell

    $ flwr new flower-tutorial --framework pytorch --username <your-username>

After running the command, a new directory called ``flower-tutorial`` will be created.
It should have the following structure:

.. code-block:: shell

    flower-tutorial
    ├── README.md
    ├── flower_tutorial
    │   ├── __init__.py
    │   ├── client_app.py   # Defines your ClientApp
    │   ├── server_app.py   # Defines your ServerApp
    │   └── task.py         # Defines your model, training and data loading
    ├── pyproject.toml      # Project metadata like dependencies and configs
    └── README.md

Next, we follow the instructions displayed and proceed to install the project and its
dependencies, which are specified in the ``pyproject.toml`` file.

.. code-block:: shell

    $ cd flower-tutorial
    $ pip install -e .

.. tip::

    The `flower-tutorial` project is a Flower app, which means that it can be run using
    the ``flwr run`` command. Flower apps created via ``flwr new`` always contain a
    ``README.md`` providing instructions on how to run and customize the app.

The Flower App we just created via ``flwr new`` shows how to federate a small
Convolutional Neural Network (CNN) for the `CIFAR-10
<https://huggingface.co/datasets/uoft-cs/cifar10>`_ dataset using PyTorch. By default,
it will run using Flower's `Simulation Runtime
<https://flower.ai/docs/runtimes/simulation/>`_ involving 10 clients (or SuperNodes),
each with it's own dataset partition, and will run for just three rounds.

Dataset partitioning
~~~~~~~~~~~~~~~~~~~~

We simulate having multiple datasets from multiple organizations (or clients) by
splitting the original CIFAR-10 dataset into multiple partitions. Each partition will
represent the data from a single organization.

Each organization will act as a client in the federated learning system. Having ten
organizations participate in a federation means having ten clients connected to the
federated learning server.

We use the `Flower Datasets <https://flower.ai/docs/datasets/>`_ library
(``flwr-datasets``) to partition CIFAR-10 into ten partitions using
``FederatedDataset``. Flower Dataset build on top of HuggingFace Dataset abstraction
allowing for seamless integration with existing datasets and data loaders. See for
instance the highlighted lines, the `dataset` name is the HF identifier for the CIFAR-10
dataset. In this case we are creating 10 IID partitions. Using the ``load_data()``
function defined in ``task.py``, we will create a small training and test set for each
of the ten organizations and wrap each of these into a PyTorch ``DataLoader``:

.. code-block:: python
    :emphasize-lines: 6-11

    def load_data(partition_id: int, num_partitions: int):
        """Load partition CIFAR10 data."""
        # Only initialize `FederatedDataset` once
        global fds
        if fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
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

.. note::

    This is a simulation of a federated learning system. In real-world federated
    learning systems, each organization has its own data and trains/evaluates models
    only on this internal data. In this tutorial, we simulate this by splitting the
    dataset into multiple partitions using Flower Datasets. There are over a dozen
    `partitioners
    <https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html>`_ available
    in Flower Datasets, that allow you to control the degree of heterogeneity in the
    data splits and how these are generated.

We now have a function that can return a training set and validation set
(``trainloader`` and ``valloader``) representing one dataset from one of ten different
organizations.

Define the Flower ClientApp
---------------------------

With that out of the way, let’s move on to the interesting part. Federated learning
systems consist of a server and multiple clients. In Flower, we create a ``ServerApp``
and a ``ClientApp`` to run the server-side and client-side code, respectively.

The first step toward creating a ``ClientApp`` is to implement a subclasses of
``flwr.client.Client`` or ``flwr.client.NumPyClient``. We use ``NumPyClient`` in this
tutorial because it is easier to implement and requires us to write less boilerplate. To
implement ``NumPyClient``, we create a subclass that implements the three methods
``get_weights``, ``fit``, and ``evaluate``:

- ``get_weights``: Return the current local model parameters
- ``fit``: Receive model parameters from the server, train the model on the local data,
  and return the updated model parameters to the server
- ``evaluate``: Receive model parameters from the server, evaluate the model on the
  local data, and return the evaluation result to the server

We mentioned that our clients will use the previously defined PyTorch components for
model training and evaluation. Let’s see a simple Flower client implementation that
brings everything together. Note that all of this boilerplate implementation has already
been done for us in our Flower project:

.. code-block:: python
    :emphasize-lines: 11, 29

    class FlowerClient(NumPyClient):
        def __init__(self, net, trainloader, valloader, local_epochs):
            self.net = net  # The model to train
            ...

        def fit(self, parameters, config):
            """Train the model on the local data."""
            # Set the model parameters to the received ones
            set_weights(self.net, parameters)
            # Train the model on the local data
            train_loss = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                self.device,
            )
            # Return the updated model parameters and metrics
            return (
                get_weights(self.net),
                len(self.trainloader.dataset),
                {"train_loss": train_loss},
            )

        def evaluate(self, parameters, config):
            """Evaluate the model on the local data."""
            # Set the model parameters to the received ones
            set_weights(self.net, parameters)
            # Evaluate the model on the local data
            loss, accuracy = test(self.net, self.valloader, self.device)
            # Return the evaluation result
            return loss, len(self.valloader.dataset), {"accuracy": accuracy}

Our class ``FlowerClient`` defines how local training/evaluation will be performed and
allows Flower to call the local training/evaluation through ``fit`` and ``evaluate``.
Each instance of ``FlowerClient`` represents a *single client* in our federated learning
system. Federated learning systems have multiple clients (otherwise, there’s not much to
federate), so each client will be represented by its own instance of ``FlowerClient``.
If we have, for example, three clients in our workload, then we’d have three instances
of ``FlowerClient`` (one on each of the machines we’d start the client on). Flower calls
``FlowerClient.fit`` on the respective instance when the server selects a particular
client for training (and ``FlowerClient.evaluate`` for evaluation).

In this project, we want to simulate a federated learning system with 10 clients *on a
single machine*. This means that the server and all 10 clients will live on a single
machine and share resources such as CPU, GPU, and memory. Having 10 clients would mean
having 10 instances of ``FlowerClient`` in memory. Doing this on a single machine can
quickly exhaust the available memory resources, even if only a subset of these clients
participates in a single round of federated learning.

In addition to the regular capabilities where server and clients run on multiple
machines, Flower, therefore, provides special simulation capabilities that create
``FlowerClient`` instances only when they are actually necessary for training or
evaluation. To enable the Flower framework to create clients when necessary, we need to
implement a function that creates a ``FlowerClient`` instance on demand. We typically
call this function ``client_fn``. Flower calls ``client_fn`` whenever it needs an
instance of one particular client to call ``fit`` or ``evaluate`` (those instances are
usually discarded after use, so they should not keep any local state). In federated
learning experiments using Flower, clients are identified by a partition ID, or
``partition_id``. This ``partition_id`` is used to load different local data partitions
for different clients, as can be seen below. The value of ``partition_id`` is retrieved
from the ``node_config`` dictionary in the ``Context`` object, which holds the
information that persists throughout each training round.

With this, we have the class ``FlowerClient`` which defines client-side
training/evaluation and ``client_fn`` which allows Flower to create ``FlowerClient``
instances whenever it needs to call ``fit`` or ``evaluate`` on one particular client.
Last, but definitely not least, we create an instance of ``ClientApp`` and pass it the
``client_fn``. ``ClientApp`` is the entrypoint that a running Flower client uses to call
your code (as defined in, for example, ``FlowerClient.fit``). The following code is
reproduced from ``client_app.py`` with additional comments:

.. code-block:: python

    def client_fn(context: Context):
        # Load model and data
        net = Net()
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        # Load data (CIFAR-10)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data partition
        # Read the node_config to fetch data partition associated to this node
        trainloader, valloader = load_data(partition_id, num_partitions)
        local_epochs = context.run_config["local-epochs"]

        # Create a single Flower client representing a single organization
        # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
        # to convert it to a subclass of `flwr.client.Client`
        return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


    # Create the Flower ClientApp
    app = ClientApp(client_fn=client_fn)

Define the Flower ServerApp
~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the server side, we need to configure a strategy which encapsulates the federated
learning approach/algorithm, for example, *Federated Averaging* (FedAvg). Flower has a
number of built-in strategies, but we can also use our own strategy implementations to
customize nearly all aspects of the federated learning approach. For this example, we
use the built-in ``FedAvg`` implementation and customize it using a few basic
parameters:

.. code-block:: python

    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,  # Sample this value of available client for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_available_clients=2,  # Wait until 2 clients are available
        initial_parameters=parameters,  # Use these initial model parameters
    )

Similar to ``ClientApp``, we create a ``ServerApp`` using a utility function
``server_fn``. This function is predefined for us in ``server_app.py``. In
``server_fn``, we pass an instance of ``ServerConfig`` for defining the number of
federated learning rounds (``num_rounds``) and we also pass the previously created
``strategy``. The ``server_fn`` returns a ``ServerAppComponents`` object containing the
settings that define the ``ServerApp`` behaviour. ``ServerApp`` is the entrypoint that
Flower uses to call all your server-side code (for example, the strategy).

.. code-block:: python

    def server_fn(context: Context):
        """Construct components that set the ServerApp behaviour.

        You can use the settings in `context.run_config` to parameterize the
        construction of all elements (e.g the strategy or the number of rounds)
        wrapped in the returned ServerAppComponents object.
        """
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

Run the training
~~~~~~~~~~~~~~~~

With all of these components in place, we can now run the federated learning simulation
with Flower! The last step is to run our simulation in command line, as follows:

.. code-block:: shell

    $ flwr run .

This will execute the federated learning simulation with 10 clients, or SuperNodes,
defined in the ``[tool.flwr.federations.local-simulation]`` section in the
``pyproject.toml``. You can also override the parameters defined in the
``[tool.flwr.app.config]`` section in ``pyproject.toml`` like this:

.. code-block:: shell

    # Run the simulation with 5 server rounds and 3 local epochs
    $ flwr run . --run-config "num-server-rounds=5 local-epochs=3"

Behind the scenes
~~~~~~~~~~~~~~~~~

So how does this work? How does Flower execute this simulation?

When we execute ``flwr run``, we tell Flower that there are 10 clients
(``options.num-supernodes = 10``, where 1 ``SuperNode`` launches 1 ``ClientApp``).
Flower then goes ahead an asks the ``ServerApp`` to issue an instructions to those nodes
using the ``FedAvg`` strategy. ``FedAvg`` knows that it should select 50% of the
available clients (``fraction-fit=0.5``), so it goes ahead and selects 5 random clients
(i.e., 50% of 10).

Flower then asks the selected 5 clients to train the model. Each of the 5 ``ClientApp``
instances receives a message, which causes it to call ``client_fn`` to create an
instance of ``FlowerClient``. It then calls ``.fit()`` on each the ``FlowerClient``
instances and returns the resulting model parameter updates to the ``ServerApp``. When
the ``ServerApp`` receives the model parameter updates from the clients, it hands those
updates over to the strategy (*FedAvg*) for aggregation. The strategy aggregates those
updates and returns the new global model, which then gets used in the next round of
federated learning.

Where’s the accuracy?
~~~~~~~~~~~~~~~~~~~~~

You may have noticed that all metrics except for ``losses_distributed`` are empty. Where
did the ``{"accuracy": float(accuracy)}`` go?

Flower can automatically aggregate losses returned by individual clients, but it cannot
do the same for metrics in the generic metrics dictionary (the one with the ``accuracy``
key). Metrics dictionaries can contain very different kinds of metrics and even
key/value pairs that are not metrics at all, so the framework does not (and can not)
know how to handle these automatically.

As users, we need to tell the framework how to handle/aggregate these custom metrics,
and we do so by passing metric aggregation functions to the strategy. The strategy will
then call these functions whenever it receives fit or evaluate metrics from clients. The
two possible functions are ``fit_metrics_aggregation_fn`` and
``evaluate_metrics_aggregation_fn``.

Let’s create a simple weighted averaging function to aggregate the ``accuracy`` metric
we return from ``evaluate``. Copy the following ``weighted_average()`` function to
``task.py``:

.. code-block:: python

    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}

Now, in ``server_app.py``, we import the function and pass it to the ``FedAvg``
strategy:

.. code-block:: python

    from flower_tutorial.task import weighted_average


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
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

We now have a full system that performs federated training and federated evaluation. It
uses the ``weighted_average`` function to aggregate custom evaluation metrics and
calculates a single ``accuracy`` metric across all clients on the server side.

The other two categories of metrics (``losses_centralized`` and ``metrics_centralized``)
are still empty because they only apply when centralized evaluation is being used. Part
two of the Flower tutorial will cover centralized evaluation.

Final remarks
-------------

Congratulations, you just trained a convolutional neural network, federated over 10
clients! With that, you understand the basics of federated learning with Flower. The
same approach you’ve seen can be used with other machine learning frameworks (not just
PyTorch) and tasks (not just CIFAR-10 images classification), for example NLP with
Hugging Face Transformers or speech with SpeechBrain.

In the next tutorial, we’re going to cover some more advanced concepts. Want to
customize your strategy? Initialize parameters on the server side? Or evaluate the
aggregated model on the server side? We’ll cover all this and more in the next tutorial.

Next steps
----------

Before you continue, make sure to join the Flower community on Flower Discuss (`Join
Flower Discuss <https://discuss.flower.ai>`__) and on Slack (`Join Slack
<https://flower.ai/join-slack/>`__).

There’s a dedicated ``#questions`` channel if you need help, but we’d also love to hear
who you are in ``#introductions``!

The :doc:`Flower Federated Learning Tutorial - Part 2
<tutorial-series-use-a-federated-learning-strategy-pytorch>` goes into more depth about
strategies and all the advanced things you can build with them.
