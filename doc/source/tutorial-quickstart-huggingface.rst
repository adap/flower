.. _quickstart-huggingface:


Quickstart 🤗 Transformers
==========================

.. meta::
   :description: Check out this Federating Learning quickstart tutorial for using Flower with HuggingFace Transformers in order to fine-tune an LLM.

Let's build a federated learning system using Hugging Face Transformers and the Flower framework!

We will leverage Hugging Face to federate the training of language models over multiple clients using Flower.
More specifically, we will fine-tune a pre-trained Transformer model (distilBERT)
for sequence classification over a dataset of IMDB ratings.
The end goal is to detect if a movie rating is positive or negative.

Dependencies
------------

First of all, it is recommended to create a virtual environment and run everything within a :doc:`virtualenv <contributor-how-to-set-up-a-virtual-env>`.

To follow along this tutorial you will need to install the following packages:
:code:`evaluate`, :code:`flwr`, :code:`flwr-datasets`, :code:`torch`, and :code:`transformers`.
This can be done using :code:`pip`:

.. code-block:: shell

    $ pip install evaluate flwr flwr-datasets torch transformers


Flower Client
-------------
Now that we have all our dependencies installed, let's run a simple distributed training with two clients and one server.
In a file called :code:`client.py`, import Flower and PyTorch related packages:

.. code-block:: python

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

.. code-block:: python
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

We also specify the model checkpoint:

.. code-block:: python
    CHECKPOINT = "distilbert-base-uncased"

Handling the data
^^^^^^^^^^^^^^^^^

To fetch the IMDB dataset, we will use `Flower Datasets <https://flower.ai/docs/datasets/>`_.
The :code:`FederatedDataset()` module downloads and partitions the dataset.
We then need to tokenize the data and create :code:`PyTorch` dataloaders,
this is all done in the :code:`load_data` function:

.. code-block:: python

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once we have a way of creating our trainloader and testloader,
we can take care of the training and testing.
This is very similar to any :code:`PyTorch` training or testing loop:

.. code-block:: python

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
^^^^^^^^^^^^^^^^^^

To create the model itself,
we will just load the pre-trained distillBERT model using Hugging Face’s :code:`AutoModelForSequenceClassification` :

.. code-block:: python

    net = AutoModelForSequenceClassification.from_pretrained(
            CHECKPOINT, num_labels=2
        ).to(DEVICE)

Creating the IMDBClient
^^^^^^^^^^^^^^^^^^^^^^^

To federate our example to multiple clients,
we first need to write our Flower client class (inheriting from :code:`flwr.client.NumPyClient`).
This is very easy, as our model is a standard :code:`PyTorch` model:

.. code-block:: python

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


The :code:`get_parameters` function lets the server get the client's parameters.
Inversely, the :code:`set_parameters` function allows the server to send its parameters to the client.
Finally, the :code:`fit` function trains the model locally for the client,
and the :code:`evaluate` function tests the model locally and returns the relevant metrics.

Next, we create a client function that returns instances of :code:`IMDBClient` on-demand when called:

.. code-block:: python

    def client_fn(cid: str) -> Client:
        return IMBDClient().to_client()

Finally, we create a :code:`ClientApp()` object that uses this client function:

.. code-block:: python

    app = ClientApp(client_fn=client_fn)

That's it for the client. We only have to implement :code:`Client` or :code:`NumPyClient`, create a :code:`ClientApp`, and pass the client function to it. If we implement a client of type :code:`NumPyClient` we'll need to first call its :code:`to_client()` method.


Flower Server
-------------

Now that we have a way to instantiate clients, we need to create our server in order to aggregate the results.
Using Flower, this can be done very easily by first choosing a strategy (here, we are using :code:`FedAvg`,
which will define the global weights as the average of all the clients' weights at each round).
In a file named :code:`server.py`, import Flower and define the strategy as follows:

.. code-block:: python

    import flwr as fl
    from flwr.server import ServerApp, ServerConfig

    def weighted_average(metrics):
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

The :code:`weighted_average` function is there to provide a way to aggregate the metrics distributed amongst
the clients (basically this allows us to display a nice average accuracy and loss for every round).
Next, we set the number of federated learning rounds in `ServerConfig` using the parameter :code:`num_rounds`:

.. code-block:: python

    config = ServerConfig(num_rounds=3)

Last but not least, we create a :code:`ServerApp` and pass both `strategy` and `config`:

.. code-block:: python

    app = ServerApp(
        config=config,
        strategy=strategy,
    )


Train the model, federated!
---------------------------

With both :code:`ClientApps` and :code:`ServerApp` ready, we can now run everything and see federated
learning in action. First, we run the :code:`flower-superlink` command in one terminal to start the infrastructure. This step only needs to be run once.

.. admonition:: Note
    :class: note

    In this example, the :code:`--insecure` command line argument starts Flower without HTTPS and is only used for prototyping. To run with HTTPS, we instead use the argument :code:`--certificates` and pass the paths to the certificates. Please refer to `Flower CLI reference <ref-api-cli.html>`_ for implementation details.

.. code-block:: shell

    $ flower-superlink --insecure

FL systems usually have a server and multiple clients. We therefore need to start multiple `SuperNode`s, one for each client, respectively. First, we open a new terminal and start the first `SuperNode` using the :code:`flower-client-app` command.

.. code-block:: shell

    $ flower-client-app client:app --insecure

In the above, we launch the :code:`app` object in the :code:`client.py` module.
Open another terminal and start the second `SuperNode`:

.. code-block:: shell

    $ flower-client-app client:app --insecure

Finally, in another terminal window, we run the `ServerApp`. This starts the actual training run:

.. code-block:: shell

    $ flower-server-app server:app --insecure

We should now see how the training does in the last terminal (the one that started the :code:`ServerApp`):

.. code-block:: shell

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
learning system for an LLM. The full source code for this can be found in
|quickstart_hf_link|_.

.. |quickstart_hf_link| replace:: :code:`examples/quickstart-huggingface`
.. _quickstart_hf_link: https://github.com/adap/flower/tree/main/examples/quickstart-huggingface

Of course, this is a very basic example, and a lot can be added or modified,
it was just to showcase how simply we could federate a Hugging Face workflow using Flower.
