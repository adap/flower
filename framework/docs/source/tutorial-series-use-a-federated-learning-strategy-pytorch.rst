Use a federated learning strategy
=================================

Welcome to the next part of the federated learning tutorial. In previous parts of this
tutorial, we introduced federated learning with PyTorch and Flower (:doc:`part 1
<tutorial-series-get-started-with-flower-pytorch>`).

In part 2, we'll begin to customize the federated learning system we built in part 1
using the Flower framework, Flower Datasets, and PyTorch.

    `Star Flower on GitHub <https://github.com/adap/flower>`__ ⭐️ and join the Flower
    community on Flower Discuss and the Flower Slack to connect, ask questions, and get
    help:

    - `Join Flower Discuss <https://discuss.flower.ai/>`__ We'd love to hear from you in
      the ``Introduction`` topic! If anything is unclear, post in ``Flower Help -
      Beginners``.
    - `Join Flower Slack <https://flower.ai/join-slack>`__ We'd love to hear from you in
      the ``#introductions`` channel! If anything is unclear, head over to the
      ``#questions`` channel.

Let's move beyond FedAvg with Flower strategies! 🌼

Preparation
-----------

Before we begin with the actual code, let's make sure that we have everything we need.

Installing dependencies
~~~~~~~~~~~~~~~~~~~~~~~

.. note::

    If you've completed part 1 of the tutorial, you can skip this step.

First, we install the Flower package ``flwr``:

.. code-block:: shell

    # In a new Python environment
    $ pip install -U "flwr[simulation]"

Then, we create a new Flower app called ``flower-tutorial`` using the PyTorch template.
We also specify a username (``flwrlabs``) for the project:

.. code-block:: shell

    $ flwr new flower-tutorial --framework pytorch --username flwrlabs

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

Next, we install the project and its dependencies, which are specified in the
``pyproject.toml`` file:

.. code-block:: shell

    $ cd flower-tutorial
    $ pip install -e .

Strategy customization
----------------------

So far, everything should look familiar if you've worked through the introductory
tutorial. With that, we're ready to introduce a number of new features.

Starting with a customized strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In part 1, we created a ``ServerApp`` (in ``server_app.py``) using the ``server_fn``. In
it, we defined the strategy and number of training rounds.

The strategy encapsulates the federated learning approach/algorithm, for example,
``FedAvg`` or ``FedAdagrad``. Let's try to use a different strategy this time. Add this
line to the top of your ``server_app.py``: ``from flwr.server.strategy import
FedAdagrad`` and replace the ``server_fn()`` with the following code:

.. code-block:: python

    def server_fn(context: Context):
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]
        fraction_fit = context.run_config["fraction-fit"]

        # Initialize model parameters
        ndarrays = get_weights(Net())
        parameters = ndarrays_to_parameters(ndarrays)

        # Define strategy
        strategy = FedAdagrad(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)

Next, run the training with the following command:

.. code-block:: shell

    $ flwr run .

Server-side parameter **evaluation**
------------------------------------

Flower can evaluate the aggregated model on the server-side or on the client-side.
Client-side and server-side evaluation are similar in some ways, but different in
others.

**Centralized Evaluation** (or *server-side evaluation*) is conceptually simple: it
works the same way that evaluation in centralized machine learning does. If there is a
server-side dataset that can be used for evaluation purposes, then that's great. We can
evaluate the newly aggregated model after each round of training without having to send
the model to clients. We're also fortunate in the sense that our entire evaluation
dataset is available at all times.

**Federated Evaluation** (or *client-side evaluation*) is more complex, but also more
powerful: it doesn't require a centralized dataset and allows us to evaluate models over
a larger set of data, which often yields more realistic evaluation results. In fact,
many scenarios require us to use **Federated Evaluation** if we want to get
representative evaluation results at all. But this power comes at a cost: once we start
to evaluate on the client side, we should be aware that our evaluation dataset can
change over consecutive rounds of learning if those clients are not always available.
Moreover, the dataset held by each client can also change over consecutive rounds. This
can lead to evaluation results that are not stable, so even if we would not change the
model, we'd see our evaluation results fluctuate over consecutive rounds.

We've seen how federated evaluation works on the client side (i.e., by implementing the
``evaluate`` method in ``FlowerClient``). Now let's see how we can evaluate aggregated
model parameters on the server-side. First we define a new function ``evaluate`` in
``task.py``:

.. code-block:: python

    from datasets import load_dataset


    def evaluate(
        server_round: int,
        parameters,
        config,
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = Net().to(device)

        # Load the entire CIFAR10 test dataset
        # It's a huggingface dataset, so we can load it directly and apply transforms
        cifar10_test = load_dataset("cifar10", split="test")
        pytorch_transforms = Compose(
            [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        def apply_transforms(batch):
            batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
            return batch

        testset = cifar10_test.with_transform(apply_transforms)
        testloader = DataLoader(testset, batch_size=64)

        set_weights(net, parameters)  # Update model with the latest parameters
        loss, accuracy = test(net, testloader, device)
        return loss, {"accuracy": accuracy}

Next, in ``server_app.py``, we pass the ``evaluate`` function to the ``evaluate_fn``
parameter of the ``FedAvg`` strategy:

.. code-block:: python

    def server_fn(context: Context) -> ServerAppComponents:
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]
        fraction_fit = context.run_config["fraction-fit"]

        # Initialize model parameters
        ndarrays = get_weights(Net())
        parameters = ndarrays_to_parameters(ndarrays)

        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_fn=evaluate,
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

Finally, we run the simulation.

.. code-block:: shell

    $ flwr run .

Sending configurations to clients from strategies
-------------------------------------------------

In some situations, we want to configure client-side execution (training, evaluation)
from the server-side. One example for that is the server asking the clients to train for
a certain number of local epochs. Flower provides a way to send configuration values
from the server to the clients using a dictionary. Let's look at an example where the
clients receive values from the server through the ``config`` parameter in ``fit``
(``config`` is also available in ``evaluate``). The ``fit`` method receives the
configuration dictionary through the ``config`` parameter and can then read values from
this dictionary. In this example, it reads ``server_round`` and ``local_epochs`` and
uses those values to improve the logging and configure the number of local training
epochs. In our ``client_app.py``, replace the ``FlowerClient()`` class and
``client_fn()`` with the following code:

.. code-block:: python

    class FlowerClient(NumPyClient):
        def __init__(self, pid, net, trainloader, valloader):
            self.pid = pid  # partition ID of a client
            self.net = net
            self.trainloader = trainloader
            self.valloader = valloader
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.to(self.device)

        def get_weights(self, config):
            print(f"[Client {self.pid}] get_weights")
            return get_weights(self.net)

        def fit(self, parameters, config):
            # Read values from config
            server_round = config["server_round"]
            local_epochs = config["local_epochs"]

            # Use values provided by the config
            print(f"[Client {self.pid}, round {server_round}] fit, config: {config}")
            set_weights(self.net, parameters)
            train(self.net, self.trainloader, epochs=local_epochs, device=self.device)
            return get_weights(self.net), len(self.trainloader), {}

        def evaluate(self, parameters, config):
            print(f"[Client {self.pid}] evaluate, config: {config}")
            set_weights(self.net, parameters)
            loss, accuracy = test(self.net, self.valloader, device=self.device)
            return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


    def client_fn(context: Context):
        net = Net()
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        trainloader, valloader = load_data(partition_id, num_partitions)

        return FlowerClient(partition_id, net, trainloader, valloader).to_client()

So how can we send this config dictionary from server to clients? The built-in Flower
Strategies provide way to do this, and it works similarly to the way server-side
evaluation works. We provide a callback to the strategy, and the strategy calls this
callback for every round of federated learning. Add the following to your
``server_app.py``:

.. code-block:: python

    def fit_config(server_round: int):
        """Return training configuration dict for each round.

        Perform two rounds of training with one local epoch, increase to two local
        epochs afterwards.
        """
        config = {
            "server_round": server_round,  # The current round of federated learning
            "local_epochs": 1 if server_round < 2 else 2,
        }
        return config

Next, we'll pass this function to the FedAvg strategy before starting the simulation.
Change the ``server_fn()`` function in ``server_app.py`` to the following:

.. code-block:: python

    def server_fn(context: Context):
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]
        fraction_fit = context.run_config["fraction-fit"]

        # Initialize model parameters
        ndarrays = get_weights(Net())
        parameters = ndarrays_to_parameters(ndarrays)

        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_fn=evaluate,
            on_fit_config_fn=fit_config,
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)

Finally, run the training with the following command:

.. code-block:: shell

    $ flwr run .

As we can see, the client logs now include the current round of federated learning
(which they read from the ``config`` dictionary). We can also configure local training
to run for one epoch during the first and second round of federated learning, and then
for two epochs during the third round.

Clients can also return arbitrary values to the server. To do so, they return a
dictionary from ``fit`` and/or ``evaluate``. We have seen and used this concept
throughout this tutorial without mentioning it explicitly: our ``FlowerClient`` returns
a dictionary containing a custom key/value pair as the third return value in
``evaluate``.

Scaling federated learning
--------------------------

As a last step in this tutorial, let's see how we can use Flower to experiment with a
large number of clients. In the ``pyproject.toml``, increase the number of SuperNodes to
1000:

.. code-block:: toml

    [tool.flwr.federations.local-simulation]
    options.num-supernodes = 1000

Note that we can reuse the ``ClientApp`` for different ``num-supernodes`` since the
Context is defined by the ``num-partitions`` argument in the ``client_fn()`` and for
simulations with Flower, the number of partitions is equal to the number of SuperNodes.

We now have 1000 partitions, each holding 45 training and 5 validation examples. Given
that the number of training examples on each client is quite small, we should probably
train the model a bit longer, so we configure the clients to perform 3 local training
epochs. We should also adjust the fraction of clients selected for training during each
round (we don't want all 1000 clients participating in every round), so we adjust
``fraction_fit`` to ``0.025``, which means that only 2.5% of available clients (so 25
clients) will be selected for training each round. We update the ``fraction-fit`` value
in the ``pyproject.toml``:

.. code-block:: toml

    [tool.flwr.app.config]
    fraction-fit = 0.025

Then, we update the ``fit_config`` and ``server_fn`` functions in ``server_app.py`` to
the following:

.. code-block:: python

    def fit_config(server_round: int):
        config = {
            "server_round": server_round,
            "local_epochs": 3,
        }
        return config


    def server_fn(context: Context):
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]
        fraction_fit = context.run_config["fraction-fit"]

        # Initialize model parameters
        ndarrays = get_weights(Net())
        parameters = ndarrays_to_parameters(ndarrays)

        # Create FedAvg strategy
        strategy = FedAvg(
            fraction_fit=fraction_fit,  # Train on 25 clients (each round)
            fraction_evaluate=0.05,  # Evaluate on 50 clients (each round)
            min_fit_clients=20,
            min_evaluate_clients=40,  # Optional config
            min_available_clients=1000,  # Optional config
            initial_parameters=parameters,
            on_fit_config_fn=fit_config,
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)


    # Create the ServerApp
    server = ServerApp(server_fn=server_fn)

Finally, run the simulation with the following command:

.. code-block:: shell

    $ flwr run .

Recap
-----

In this tutorial, we've seen how we can gradually enhance our system by customizing the
strategy, initializing parameters on the server side, choosing a different strategy, and
evaluating models on the server-side. That's quite a bit of flexibility with so little
code, right?

In the later sections, we've seen how we can communicate arbitrary values between server
and clients to fully customize client-side execution. With that capability, we built a
large-scale Federated Learning simulation using the Flower Virtual Client Engine and ran
an experiment involving 1000 clients in the same workload - all in the same Flower
project!

Next steps
----------

Before you continue, make sure to join the Flower community on Flower Discuss (`Join
Flower Discuss <https://discuss.flower.ai>`__) and on Slack (`Join Slack
<https://flower.ai/join-slack/>`__).

There's a dedicated ``#questions`` Slack channel if you need help, but we'd also love to
hear who you are in ``#introductions``!

The :doc:`Flower Federated Learning Tutorial - Part 3
<tutorial-series-build-a-strategy-from-scratch-pytorch>` shows how to build a fully
custom ``Strategy`` from scratch.
