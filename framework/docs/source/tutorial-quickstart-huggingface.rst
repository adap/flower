:og:description: Learn how to train a large language model on the IMDB dataset using federated learning with Flower and 🤗 Hugging Face in this step-by-step tutorial.
.. meta::
    :description: Learn how to train a large language model on the IMDB dataset using federated learning with Flower and 🤗 Hugging Face in this step-by-step tutorial.

.. _quickstart-huggingface:

Quickstart 🤗 Transformers
==========================

In this federated learning tutorial we will learn how to train a large language model
(LLM) on the `IMDB <https://huggingface.co/datasets/stanfordnlp/imdb>`_ dataset using
Flower and the 🤗 Hugging Face Transformers library. It is recommended to create a
virtual environment and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

Let's use ``flwr new`` to create a complete Flower+🤗 Hugging Face project. It will
generate all the files needed to run, by default with the Flower Simulation Engine, a
federation of 10 nodes using |fedavg|_ The dataset will be partitioned using
|flowerdatasets|_'s |iidpartitioner|_.

Now that we have a rough idea of what this example is about, let's get started. First,
install Flower in your new environment:

.. code-block:: shell

    # In a new Python environment
    $ pip install flwr

Then, run the command below. You will be prompted to select one of the available
templates (choose ``HuggingFace``), give a name to your project, and type in your
developer name:

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
    INFO :      Starting Flower ServerApp, config: num_rounds=3, no round_timeout
    INFO :
    INFO :      [INIT]
    INFO :      Using initial global parameters provided by strategy
    INFO :      Starting evaluation of initial global parameters
    INFO :      Evaluation returned no results (`None`)
    INFO :
    INFO :      [ROUND 1]
    INFO :      configure_fit: strategy sampled 2 clients (out of 10)
    INFO :      aggregate_fit: received 2 results and 0 failures
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
    INFO :      Run finished 3 round(s) in 249.11s
    INFO :          History (loss, distributed):
    INFO :                  round 1: 0.02111011856794357
    INFO :                  round 2: 0.019722302150726317
    INFO :                  round 3: 0.018227258533239362
    INFO :

You can also run the project with GPU as follows:

.. code-block:: shell

    # Run with default arguments
    $ flwr run . localhost-gpu

This will use the default arguments where each ``ClientApp`` will use 2 CPUs and at most
4 ``ClientApp``\s will run in a given GPU.

You can also override the parameters defined in the ``[tool.flwr.app.config]`` section
in ``pyproject.toml`` like this:

.. code-block:: shell

    # Override some arguments
    $ flwr run . --run-config "num-server-rounds=5 fraction-fit=0.2"

What follows is an explanation of each component in the project you just created:
dataset partition, the model, defining the ``ClientApp`` and defining the ``ServerApp``.

The Data
--------

This tutorial uses |flowerdatasets|_ to easily download and partition the `IMDB
<https://huggingface.co/datasets/stanfordnlp/imdb>`_ dataset. In this example you'll
make use of the |iidpartitioner|_ to generate ``num_partitions`` partitions. You can
choose |otherpartitioners|_ available in Flower Datasets. To tokenize the text, we will
also load the tokenizer from the pre-trained Transformer model that we'll use during
training - more on that in the next section. Each ``ClientApp`` will call this function
to create dataloaders with the data that correspond to their data partition.

.. code-block:: python

    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="stanfordnlp/imdb",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    # Divide data: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, add_special_tokens=True, max_length=512
        )


    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns("text")
    partition_train_test = partition_train_test.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        partition_train_test["test"], batch_size=32, collate_fn=data_collator
    )

The Model
---------

We will leverage 🤗 Hugging Face to federate the training of language models over
multiple clients using Flower. More specifically, we will fine-tune a pre-trained
Transformer model (|berttiny|_) for sequence classification over the dataset of IMDB
ratings. The end goal is to detect if a movie rating is positive or negative. If you
have access to larger GPUs, feel free to use larger models!

.. code-block:: python

    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

Note that here, ``model_name`` is a string that will be loaded from the ``Context`` in
the ClientApp and ServerApp.

In addition to loading the pretrained model weights and architecture, we also include
two utility functions to perform both training (i.e. ``train()``) and evaluation (i.e.
``test()``) using the above model. These functions should look fairly familiar if you
have some prior experience with PyTorch. Note these functions do not have anything
specific to Flower. That being said, the training function will normally be called, as
we'll see later, from a Flower client passing its own data. In summary, your clients can
use standard training/testing functions to perform local training or evaluation:

.. code-block:: python

    def train(net, trainloader, epochs, device):
        optimizer = AdamW(net.parameters(), lr=5e-5)
        net.train()
        for _ in range(epochs):
            for batch in trainloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = net(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


    def test(net, testloader, device):
        metric = load_metric("accuracy")
        loss = 0
        net.eval()
        for batch in testloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = net(**batch)
            logits = outputs.logits
            loss += outputs.loss.item()
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        loss /= len(testloader.dataset)
        accuracy = metric.compute()["accuracy"]
        return loss, accuracy

The ClientApp
-------------

The main changes we have to make to use 🤗 Hugging Face with Flower will be found in the
``get_weights()`` and ``set_weights()`` functions. Under the hood, the ``transformers``
library uses PyTorch, which means we can reuse the ``get_weights()`` and
``set_weights()`` code that we defined in the :doc:`Quickstart PyTorch
<tutorial-quickstart-pytorch>` tutorial. As a reminder, in ``get_weights()``, PyTorch
model parameters are extracted and represented as a list of NumPy arrays. The
``set_weights()`` function that's the opposite: given a list of NumPy arrays it applies
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
        def __init__(self, net, trainloader, testloader, local_epochs):
            self.net = net
            self.trainloader = trainloader
            self.testloader = testloader
            self.local_epochs = local_epochs
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net.to(self.device)

        def fit(self, parameters, config):
            set_weights(self.net, parameters)
            train(self.net, self.trainloader, epochs=self.local_epochs, device=self.device)
            return get_weights(self.net), len(self.trainloader), {}

        def evaluate(self, parameters, config):
            set_weights(self.net, parameters)
            loss, accuracy = test(self.net, self.testloader, self.device)
            return float(loss), len(self.testloader), {"accuracy": accuracy}

Finally, we can construct a ``ClientApp`` using the ``FlowerClient`` defined above by
means of a ``client_fn()`` callback. Note that the `context` enables you to get access
to hyperparemeters defined in your ``pyproject.toml`` to configure the run. In this
tutorial we access the ``local-epochs`` setting to control the number of epochs a
``ClientApp`` will perform when running the ``fit()`` method. You could define
additional hyperparameters in ``pyproject.toml`` and access them here.

.. code-block:: python

    def client_fn(context: Context):

        # Get this client's dataset partition
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        model_name = context.run_config["model-name"]
        trainloader, valloader = load_data(partition_id, num_partitions, model_name)

        # Load model
        num_labels = context.run_config["num-labels"]
        net = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        local_epochs = context.run_config["local-epochs"]

        # Return Client instance
        return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


    # Flower ClientApp
    app = ClientApp(client_fn)

The ServerApp
-------------

To construct a ``ServerApp`` we define a ``server_fn()`` callback with an identical
signature to that of ``client_fn()`` but the return type is |serverappcomponents|_ as
opposed to a |client|_ In this example we use the `FedAvg` strategy. To it we pass a
randomly initialized model that will server as the global model to federated. Note that
the value of ``fraction_fit`` is read from the run config. You can find the default
value defined in the ``pyproject.toml``.

.. code-block:: python

    def server_fn(context: Context):
        # Read from config
        num_rounds = context.run_config["num-server-rounds"]
        fraction_fit = context.run_config["fraction-fit"]

        # Initialize global model
        model_name = context.run_config["model-name"]
        num_labels = context.run_config["num-labels"]
        net = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        weights = get_weights(net)
        initial_parameters = ndarrays_to_parameters(weights)

        # Define strategy
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            initial_parameters=initial_parameters,
        )
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)


    # Create ServerApp
    app = ServerApp(server_fn=server_fn)

Congratulations! You've successfully built and run your first federated learning system
for an LLM.

.. note::

    Check the source code of the extended version of this tutorial in
    |quickstart_hf_link|_ in the Flower GitHub repository. For a comprehensive example
    of a federated fine-tuning of an LLM with Flower, refer to the |flowertune|_ example
    in the Flower GitHub repository.

.. |quickstart_hf_link| replace:: ``examples/quickstart-huggingface``

.. |fedavg| replace:: ``FedAvg``

.. |iidpartitioner| replace:: ``IidPartitioner``

.. |otherpartitioners| replace:: other partitioners

.. |berttiny| replace:: ``bert-tiny``

.. |serverappcomponents| replace:: ``ServerAppComponents``

.. |client| replace:: ``Client``

.. |flowerdatasets| replace:: Flower Datasets

.. |flowertune| replace:: FlowerTune LLM

.. _berttiny: https://huggingface.co/prajjwal1/bert-tiny

.. _client: ref-api/flwr.client.Client.html#client

.. _fedavg: ref-api/flwr.server.strategy.FedAvg.html#flwr.server.strategy.FedAvg

.. _flowerdatasets: https://flower.ai/docs/datasets/

.. _flowertune: https://github.com/adap/flower/tree/main/examples/flowertune-llm

.. _iidpartitioner: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner

.. _otherpartitioners: https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html

.. _quickstart_hf_link: https://github.com/adap/flower/tree/main/examples/quickstart-huggingface

.. _serverappcomponents: ref-api/flwr.server.ServerAppComponents.html#serverappcomponents
