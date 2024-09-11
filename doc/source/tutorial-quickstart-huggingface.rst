.. _quickstart-huggingface:

###########################
 Quickstart 🤗 Transformers
###########################

In this federated learning tutorial we will learn how to train a
language model on the IMBD dataset using Flower and the 🤗 Hugging Face
Transformers library. It is recommended to create a virtual environment
and run everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

Let's use `flwr new` to create a complete Flower+🤗 Hugging Face project.
It will generate all the files needed to run, by default with the Flower
Simulation Engine, a federation of 10 nodes using `FedAvg
<https://flower.ai/docs/framework/ref-api/flwr.server.strategy.FedAvg.html#flwr.server.strategy.FedAvg>`_.
The dataset will be partitioned using Flower Dataset's `IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_.

Now that we have a rough idea of what this example is about, let's get
started. First, install Flower in your new environment:

.. code:: shell

   # In a new Python environment
   $ pip install flwr

Then, run the command below. You will be prompted to select one of the
available templates (choose ``HuggingFace``), give a name to your
project, and type in your developer name:

.. code:: shell

   $ flwr new

After running it you'll notice a new directory with your project name
has been created. It should have the following structure:

.. code:: shell

   <your-project-name>
   ├── <your-project-name>
   │   ├── __init__.py
   │   ├── client_app.py   # Defines your ClientApp
   │   ├── server_app.py   # Defines your ServerApp
   │   └── task.py         # Defines your model, training and data loading
   ├── pyproject.toml      # Project metadata like dependencies and configs
   └── README.md

If you haven't yet installed the project and its dependencies, you can
do so by:

.. code:: shell

   # From the directory where your pyproject.toml is
   $ pip install -e .

To run the project, do:

.. code:: shell

   # Run with default arguments
   $ flwr run .

With default arguments you will see an output like this one:

.. code:: shell

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

.. code:: shell

   # Run with default arguments
   $ flwr run . localhost-gpu

This will use the default arguments where each ClientApp will use 2 CPUs
and at most 4 ClientApps will run in a given GPU.

You can also override the parameters defined in the
``[tool.flwr.app.config]`` section in ``pyproject.toml`` like this:

.. code:: shell

   # Override some arguments
   $ flwr run . --run-config "num-server-rounds=5 fraction-fit=0.2"

What follows is an explanation of each component in the project you just
created: dataset partition, the model, defining the ``ClientApp`` and
defining the ``ServerApp``.

**********
 The Data
**********

This tutorial uses `Flower Datasets <https://flower.ai/docs/datasets/>`_
to easily download and partition the `IMDB
<https://huggingface.co/datasets/stanfordnlp/imdb>`_ dataset. In this
example you'll make use of the `IidPartitioner
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.IidPartitioner.html#flwr_datasets.partitioner.IidPartitioner>`_
to generate `num_partitions` partitions. You can choose `other
partitioners
<https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html>`_
available in Flower Datasets. To tokenize the text, we will also load
the tokenizer from the pre-trained Transformer model that we'll use
during training - more on that in the next section. Each ``ClientApp``
will call this function to create dataloaders with the data that
correspond to their data partition.

.. code:: python

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

***********
 The Model
***********

We will leverage Hugging Face to federate the training of language
models over multiple clients using Flower. More specifically, we will
fine-tune a pre-trained Transformer model (`bert-tiny
<https://huggingface.co/prajjwal1/bert-tiny>`_) for sequence
classification over the dataset of IMDB ratings. The end goal is to
detect if a movie rating is positive or negative. If you have access to
larger GPUs, feel free to use larger models!

.. code:: python

   net = AutoModelForSequenceClassification.from_pretrained(
       model_name, num_labels=num_labels
   )

Note that here, ``model_name`` is a string that will be loaded from the
``Context`` in the ClientApp and ServerApp.

In addition to loading the pretrained model weights and architecture, we
also include two utility functions to perform both training (i.e.
``train()``) and evaluation (i.e. ``test()``) using the above model.
These functions should look fairly familiar if you have some prior
experience with PyTorch. Note these functions do not have anything
specific to Flower. That being said, the training function will normally
be called, as we'll see later, from a Flower client passing its own
data. In summary, your clients can use standard training/testing
functions to perform local training or evaluation:

.. code:: python

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

***************
 The ClientApp
***************

The main changes we have to make to use Hugging Face with `Flower` will
be found in the ``get_weights()`` and ``set_weights()`` functions. Under
the hood, the ``transformers`` library uses `PyTorch`, which means we
can reuse the ``get_weights()`` and ``set_weights()`` code that we
defined in the :doc:`Quickstart PyTorch tutorial
<tutorial-quickstart-pytorch>`. As a reminder, in ``get_weights()``,
PyTorch model parameters are extracted and represented as a list of
NumPy arrays. The ``set_weights()`` function that's the oposite: given a
list of NumPy arrays it applies them to an existing PyTorch model. Doing
this in fairly easy in PyTorch.

.. note::

   The specific implementation of ``get_weights()`` and
   ``set_weights()`` depends on the type of models you use. The ones
   shown below work for a wide range of PyTorch models but you might
   need to adjust them if you have more exotic model architectures.

.. code:: python

   def get_weights(net):
       return [val.cpu().numpy() for _, val in net.state_dict().items()]


   def set_weights(net, parameters):
       params_dict = zip(net.state_dict().keys(), parameters)
       state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
       net.load_state_dict(state_dict, strict=True)

The rest of the functionality is directly inspired by the centralized
case. The ``fit()`` method in the client trains the model using the
local dataset. Similarly, the ``evaluate()`` method is used to evaluate
the model received on a held-out validation set that the client might
have:

.. code:: python

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

Finally, we can construct a ``ClientApp`` using the ``FlowerClient``
defined above by means of a ``client_fn()`` callback. Note that the
`context` enables you to get access to hyperparemeters defined in your
``pyproject.toml`` to configure the run. In this tutorial we access the
`local-epochs` setting to control the number of epochs a ``ClientApp``
will perform when running the ``fit()`` method. You could define
additional hyperparameters in ``pyproject.toml`` and access them here.

.. code:: python

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

.. meta::
   :description: Check out this Federating Learning quickstart tutorial for using Flower with HuggingFace Transformers in order to fine-tune an LLM.

Let's build a federated learning system using Hugging Face Transformers
and the Flower framework!

Dependencies
============

First of all, it is recommended to create a virtual environment and run
everything within a :doc:`virtualenv
<contributor-how-to-set-up-a-virtual-env>`.

To follow along this tutorial you will need to install the following
packages: ``evaluate``, ``flwr``, ``flwr-datasets``, ``torch``, and
``transformers``. This can be done using ``pip``:

.. code:: shell

   $ pip install evaluate flwr flwr-datasets torch transformers

Flower Client
=============

Now that we have all our dependencies installed, let's run a simple
distributed training with two clients and one server. In a file called
``client.py``, import Flower and PyTorch related packages:

.. code:: python

   from collections import OrderedDict

   import flwr as fl
   import torch
   from evaluate import load as load_metric
   from flwr.client import Client, ClientApp
   from flwr_datasets import FederatedDataset
   from torch.optim import AdamW
   from torch.utils.data import DataLoader
   from transformers import (
       AutoModelForSequenceClassification,
       AutoTokenizer,
       DataCollatorWithPadding,
   )

In addition, we define the device allocation in PyTorch with:

.. code:: python

   DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

We also specify the model checkpoint:

.. code:: python

   CHECKPOINT = "distilbert-base-uncased"

Handling the data
-----------------

To fetch the IMDB dataset, we will use `Flower Datasets
<https://flower.ai/docs/datasets/>`_. The ``FederatedDataset()`` module
downloads and partitions the dataset. We then need to tokenize the data
and create ``PyTorch`` dataloaders, this is all done in the
``load_data`` function:

.. code:: python

   def load_data(partition_id):
       """Load IMDB data (training and eval)"""
       fds = FederatedDataset(dataset="imdb", partitioners={"train": 1_000})
       partition = fds.load_partition(partition_id)
       # Divide data: 80% train, 20% test
       partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

       tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, model_max_length=512)

       def tokenize_function(examples):
           return tokenizer(examples["text"], truncation=True)

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

       return trainloader, testloader

Training and testing the model
------------------------------

Once we have a way of creating our trainloader and testloader, we can
take care of the training and testing. This is very similar to any
``PyTorch`` training or testing loop:

.. code:: python

   def train(net, trainloader, epochs):
       optimizer = AdamW(net.parameters(), lr=5e-5)
       net.train()
       for _ in range(epochs):
           for batch in trainloader:
               batch = {k: v.to(DEVICE) for k, v in batch.items()}
               outputs = net(**batch)
               loss = outputs.loss
               loss.backward()
               optimizer.step()
               optimizer.zero_grad()


   def test(net, testloader):
       metric = load_metric("accuracy")
       loss = 0
       net.eval()
       for batch in testloader:
           batch = {k: v.to(DEVICE) for k, v in batch.items()}
           with torch.no_grad():
               outputs = net(**batch)
           logits = outputs.logits
           loss += outputs.loss.item()
           predictions = torch.argmax(logits, dim=-1)
           metric.add_batch(predictions=predictions, references=batch["labels"])
       loss /= len(testloader.dataset)
       accuracy = metric.compute()["accuracy"]
       return loss, accuracy

Creating the model
------------------

To create the model itself, we will just load the pre-trained
distillBERT model using Hugging Face’s
``AutoModelForSequenceClassification`` :

.. code:: python

   net = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2).to(
       DEVICE
   )

Creating the IMDBClient
-----------------------

To federate our example to multiple clients, we first need to write our
Flower client class (inheriting from ``flwr.client.NumPyClient``). This
is very easy, as our model is a standard ``PyTorch`` model:

.. code:: python

   class IMDBClient(fl.client.NumPyClient):
       def get_parameters(self, config):
           return [val.cpu().numpy() for _, val in net.state_dict().items()]

       def set_parameters(self, parameters):
           params_dict = zip(net.state_dict().keys(), parameters)
           state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
           net.load_state_dict(state_dict, strict=True)

       def fit(self, parameters, config):
           self.set_parameters(parameters)
           print("Training Started...")
           train(net, trainloader, epochs=1)
           print("Training Finished.")
           return self.get_parameters(config={}), len(trainloader), {}

       def evaluate(self, parameters, config):
           self.set_parameters(parameters)
           loss, accuracy = test(net, testloader)
           return float(loss), len(testloader), {"accuracy": float(accuracy)}

The ``get_parameters`` function lets the server get the client's
parameters. Inversely, the ``set_parameters`` function allows the server
to send its parameters to the client. Finally, the ``fit`` function
trains the model locally for the client, and the ``evaluate`` function
tests the model locally and returns the relevant metrics.

Next, we create a client function that returns instances of
``IMDBClient`` on-demand when called:

.. code:: python

   def client_fn(cid: str) -> Client:
       return IMBDClient().to_client()

Finally, we create a ``ClientApp()`` object that uses this client
function:

.. code:: python

   app = ClientApp(client_fn=client_fn)

That's it for the client. We only have to implement ``Client`` or
``NumPyClient``, create a ``ClientApp``, and pass the client function to
it. If we implement a client of type ``NumPyClient`` we'll need to first
call its ``to_client()`` method.

Flower Server
=============

Now that we have a way to instantiate clients, we need to create our
server in order to aggregate the results. Using Flower, this can be done
very easily by first choosing a strategy (here, we are using ``FedAvg``,
which will define the global weights as the average of all the clients'
weights at each round). In a file named ``server.py``, import Flower and
define the strategy as follows:

.. code:: python

   import flwr as fl
   from flwr.server import ServerApp, ServerConfig


   def weighted_average(metrics):
       accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
       losses = [num_examples * m["loss"] for num_examples, m in metrics]
       examples = [num_examples for num_examples, _ in metrics]
       return {
           "accuracy": sum(accuracies) / sum(examples),
           "loss": sum(losses) / sum(examples),
       }


   # Define strategy
   strategy = fl.server.strategy.FedAvg(
       fraction_fit=1.0,
       fraction_evaluate=1.0,
       evaluate_metrics_aggregation_fn=weighted_average,
   )

The ``weighted_average`` function is there to provide a way to aggregate
the metrics distributed amongst the clients (basically this allows us to
display a nice average accuracy and loss for every round). Next, we set
the number of federated learning rounds in `ServerConfig` using the
parameter ``num_rounds``:

.. code:: python

   config = ServerConfig(num_rounds=3)

Last but not least, we create a ``ServerApp`` and pass both `strategy`
and `config`:

.. code:: python

   app = ServerApp(
       config=config,
       strategy=strategy,
   )

Train the model, federated!
===========================

With both ``ClientApps`` and ``ServerApp`` ready, we can now run
everything and see federated learning in action. First, we run the
``flower-superlink`` command in one terminal to start the
infrastructure. This step only needs to be run once.

.. admonition:: Note
   :class: note

   In this example, the ``--insecure`` command line argument starts
   Flower without HTTPS and is only used for prototyping. To run with
   HTTPS, we instead use the arguments ``--ssl-ca-certfile``,
   ``--ssl-certfile``, and ``--ssl-keyfile`` and pass the paths to the
   certificates. Please refer to `Flower CLI reference
   <ref-api-cli.html#flower-superlink>`_ for implementation details.

.. code:: shell

   $ flower-superlink --insecure

FL systems usually have a server and multiple clients. We therefore need
to start multiple `SuperNodes`, one for each client, respectively.
First, we open a new terminal and start the first `SuperNode` using the
``flower-client-app`` command.

.. code:: shell

   $ flower-client-app client:app --insecure

In the above, we launch the ``app`` object in the ``client.py`` module.
Open another terminal and start the second `SuperNode`:

.. code:: shell

   $ flower-client-app client:app --insecure

Finally, in another terminal window, we run the `ServerApp`. This starts
the actual training run:

.. code:: shell

   $ flower-server-app server:app --insecure

We should now see how the training does in the last terminal (the one
that started the ``ServerApp``):

.. code:: shell

   WARNING :   Option `--insecure` was set. Starting insecure HTTP client connected to 0.0.0.0:9091.
   INFO :      Starting Flower ServerApp, config: num_rounds=3, no round_timeout
   INFO :
   INFO :      [INIT]
   INFO :      Requesting initial parameters from one random client
   INFO :      Received initial parameters from one random client
   INFO :      Evaluating initial global parameters
   INFO :
   INFO :      [ROUND 1]
   INFO :      configure_fit: strategy sampled 2 clients (out of 2)
   INFO :      aggregate_fit: received 2 results and 0 failures
   WARNING :   No fit_metrics_aggregation_fn provided
   INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
   INFO :      aggregate_evaluate: received 2 results and 0 failures
   WARNING :   No evaluate_metrics_aggregation_fn provided
   INFO :
   INFO :      [ROUND 2]
   INFO :      configure_fit: strategy sampled 2 clients (out of 2)
   INFO :      aggregate_fit: received 2 results and 0 failures
   INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
   INFO :      aggregate_evaluate: received 2 results and 0 failures
   INFO :
   INFO :      [ROUND 3]
   INFO :      configure_fit: strategy sampled 2 clients (out of 2)
   INFO :      aggregate_fit: received 2 results and 0 failures
   INFO :      configure_evaluate: strategy sampled 2 clients (out of 2)
   INFO :      aggregate_evaluate: received 2 results and 0 failures
   INFO :
   INFO :      [SUMMARY]
   INFO :      Run finished 3 rounds in 56.57s
   INFO :      History (loss, distributed):
   INFO :          ('\tround 1: 0.13953592777252197\n'
   INFO :           '\tround 2: 0.134615957736969\n'
   INFO :           '\tround 3: 0.1451723337173462\n')

Congratulations! You've successfully built and run your first federated
learning system for an LLM. The full source code for this can be found
in |quickstart_hf_link|_.

.. |quickstart_hf_link| replace::

   :code:`examples/quickstart-huggingface`

.. _quickstart_hf_link: https://github.com/adap/flower/tree/main/examples/quickstart-huggingface

Of course, this is a very basic example, and a lot can be added or
modified, it was just to showcase how simply we could federate a Hugging
Face workflow using Flower.
