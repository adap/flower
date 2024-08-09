Use Secure Aggregation
======================

What is Secure Aggregation?
---------------------------

Secure Aggregation is a class of Secure Multi-Party Computation algorithm that provides a way for an aggregated value (such as the sum of model parameters) from multiple parties to be computed without revealing their individual inputs to one another.

To use Secure Aggregration, we need two components: the :code:`secaggplus_mod` and :code:`SecAggPlusWorkflow`.

:code:`secaggplus_mod`
~~~~~~~~~~~~~~~~~~~~~~

This mod is provided in the Flower framework to handle incoming messages and to return results according to the `SecAgg+ <https://doi.org/10.1145/3372297.3417885>`_ protocol, which makes it easier for you to start using the :code:`SecAggPlusWorkflow`. Let’s walk through how to use this mod with a simple example; We create a federated task whereby each client returns an array of :code:`[1.0, 1.0, 1.0]` to the server for aggregation. For reference on how to use mods in Flower, you can refer to :doc:`this how-to guide <how-to-use-built-in-mods>`.

We’ll start by importing :code:`flwr` and defining a :code:`FlowerClient`:

.. code-block:: python

    # file: client.py
    from flwr.client import ClientApp, NumPyClient
    from flwr.client.mod import secaggplus_mod
    import numpy as np


    class FlowerClient(NumPyClient):
        def fit(self, parameters, config):
            # Instead of training and returning model parameters,
            # the client directly returns [1.0, 1.0, 1.0] for demonstration purposes.
            return_vector = [np.ones(3)]
            return return_vector, 1, {} 

Next, we implement the :code:`client_fn`, create the :code:`ClientApp`, and pass the imported :code:`secaggplus_mod` as a list to the :code:`mods` parameter in the :code:`ClientApp` as follows:

.. code-block:: python

    # File: client.py
    def client_fn(cid: str):
        """Create and return an instance of Flower `Client`."""
        return FlowerClient().to_client()


    # Create ClientApp
    app = ClientApp(
        client_fn=client_fn,
        mods=[
            secaggplus_mod,
        ],
    )

Now that the :code:`ClientApp` is ready, we need to prepare the :code:`ServerApp`. 

:code:`SecAggPlusWorkFlow`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`SecAggPlusWorkFlow()` is a convenient class provided in Flower that includes all the necessary steps to run federated learning with secure aggregation, such as handling the cryptographic setup, and model encryption/decryption. Let’s walk through an example of a :code:`ServerApp` that uses this workflow. 

First, we import the necessary libraries, define :code:`FedAvg()` as our aggregation strategy for this example, and create the :code:`ServerApp`:

.. code-block:: python

    # file: server.py
    from flwr.common import Context
    from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.server.workflow import SecAggPlusWorkflow


    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,        # Select all available clients
        fraction_evaluate=0.0,   # Disable evaluation
        min_available_clients=5,
    )

    # Create ServerApp
    app = ServerApp()

Next, we will register the :code:`@app.main()` function as described in the :doc:`how-to use the Driver API guide <how-to-use-driver-api>`, and construct a :code:`Context` to be used by the workflow. In this context, we specify 3 rounds of federated learning based on the :code:`FedAvg` strategy:

.. code-block:: python

    # file: server.py
    @app.main()
    def main(driver: Driver, context: Context) -> None:
        context = LegacyContext(
            state=context.state,
            config=ServerConfig(num_rounds=3),
            strategy=strategy,
        )

Now, we create a workflow by wrapping Flower’s :code:`DefaultWorkflow()` class around the :code:`SecAggPlusWorkflow()` and set the 2 required parameters for it:

* :code:`num_shares` - This is the number of shares into which each client's private key is split under the SecAgg+ protocol.
* :code:`reconstruction_threshold` - This is the minimum number of shares required to reconstruct a client's private key, or, if specified as a float, it represents the proportion of the total number of shares needed for reconstruction.

In this example, we will set :code:`num_shares = 3` and :code:`reconstruction_threshold = 2` :

.. code-block:: python

    # file: server.py

    # Create the workflow
    workflow = DefaultWorkflow(
         fit_workflow=SecAggPlusWorkflow(
             num_shares=3,
             reconstruction_threshold=2,
         )
    )
    
    # Execute workflow
    workflow(driver, context)

The workflow accepts two arguments, which are the :code:`driver`, which handles the node selections and message relays, and :code:`context`, which holds the local information of the :code:`ServerApp` that is used to execute a task.

Running Secure Aggregation
--------------------------

With the :code:`client.py` and :code:`server.py` modules, we can run the example as follows. First start the Flower Superlink in one terminal window:

.. code-block:: shell

    $ flower-superlink --insecure

Next, start 5 Flower :code:`ClientApps` in 5 separate terminal windows:

.. code-block:: shell

    $ flower-client-app client:app --insecure

Finally, start the Flower :code:`ServerApp`:

.. code-block:: shell

    $ flower-server-app server:app --insecure --verbose

.. admonition:: Note
    :class: note

    In this example, the :code:`--insecure` command line argument starts Flower without HTTPS and is only used for prototyping. To run with HTTPS, we instead use the argument :code:`--certificates` and pass the paths to the certificates. Please refer to `Flower CLI reference <ref-api-cli.html>`_ for implementation details.

Conclusion
----------

Congratulations! You have successfully executed secure aggregation using the Flower framework. This guide is based on the `Secure aggregation with Flower example <https://flower.ai/docs/examples/app-secure-aggregation.html>`_ on Flower’s GitHub repository.

.. admonition:: Important
    :class: important

    As we continuously enhance Flower at a rapid pace, we'll periodically update the functionality and this how-to document. Please feel free to share any feedback with us!
