:og:description: Upgrade seamlessly to Flower 1.21 with this guide for transitioning your setup to the latest features and enhancements powered by Flower's Message API.
.. meta::
    :description: Upgrade seamlessly to Flower 1.21 with this guide for transitioning your setup to the latest features and enhancements powered by Flower's Message API.

.. |numpyclient_link| replace:: ``NumPyClient``

.. _numpyclient_link: ref-api/flwr.client.NumPyClient.html

.. |clientapp_link| replace:: ``ClientApp``

.. _clientapp_link: ref-api/flwr.client.ClientApp.html

.. |serverapp_link| replace:: ``ServerApp``

.. _serverapp_link: ref-api/flwr.server.ServerApp.html

.. |fedavg_link| replace:: ``FedAvg``

.. _fedavg_link: ref-api/flwr.serverapp.FedAvg.html

.. |message_link| replace:: ``Message``

.. _message_link: ref-api/flwr.common.Message.html

.. |arrayrecord_link| replace:: ``ArrayRecord``

.. _arrayrecord_link: ref-api/flwr.common.ArrayRecord.html

.. |metricrecord_link| replace:: ``MetricRecord``

.. _metricrecord_link: ref-api/flwr.common.MetricRecord.html

.. |configrecord_link| replace:: ``ConfigRecord``

.. _configrecord_link: ref-api/flwr.common.ConfigRecord.html

.. |recorddict_link| replace:: ``RecordDict``

.. _recorddict_link: ref-api/flwr.common.RecordDict.html

Upgrade to the Message API
==========================

Welcome to the migration guide for updating your FlowerApps to use Flower's Message API!
This guide will walk you through the necessary steps to transition from Flower Apps
based on Strategies and NumPyClient to their equivalent using the new Message API. This
guide is relevant when updating pre-``1.21`` Flower apps to the latest stable version.

Let's dive in!

Summary of changes
------------------

Thousands of Flower Apps have been created using the Strategies and |numpyclient_link|_
abstractions. With the introduction of the Message API, these apps can now take
advantage of a more powerful and flexible communication layer with the |message_link|_
abstraction being its cornerstone. Messages replace the previous `FitIns` and `FitRes`
data structures (and their equivalent for the other operations) into a single, unified
and more versatile datastructure.

To fully take advantage of the new Message API, you will need to update your app's code
to use the new message-based communication patterns. This guide will show you how to:

1. Update your |serverapp_link|_ to make use of the new `Message`-based strategies. You
   won't need to use the `server_fn` anymore. The new strategies make it easier to
   customize how the different FL rounds are executed, to retrieve results from your
   run, and more.
2. Update your |clientapp_link|_ so it operates directly on `Message` objects received
   from the |serverapp_link|_. You will be able to keep most of the code from your
   |numpyclient_link|_ implementation but you won't need to create a new class anymore
   or use the helper `client_fn` function.

.. tip::

    The main payload Message objects carry are of type |recorddict_link|_. You can think
    of it as a dictionary that can hold other types of records, namely
    |arrayrecord_link|_, |metricrecord_link|_, and |configrecord_link|_. Please refer to
    the documentation for each record for all the details on how they can be constructed
    and adapted to your usecase. In this guide we won't delve into the specifics of each
    record type, but rather focus on the overall migration process.

Install update
--------------

The first step is to update the Flower version defined in the `pyproject.toml` in your
app:

.. code-block:: toml
    :caption: pyproject.toml
    :emphasize-lines: 2

    dependencies = [
        "flwr[simulation]>=1.21.0", # update Flower package
        # ...
    ]

Then, run the following command to install the updated dependencies:

.. code-block:: bash

    # Install the app with updated dependencies
    $ pip install -e .

Update your ServerApp
---------------------

Starting with Flower 1.21, the `ServerApp` no longer requires a `server_fn` function to
make use of strategies. This is because a new collection of strategies has been created
to operate directly on `Message` objects, allowing for a more streamlined and flexible
approach to federated learning rounds.

.. note::

    The new `Message`-based strategies are located in the `flwr.serverapp` module unlike
    the previous strategies which were located in the `flwr.server` module. Over time
    more strategies will be added to the `flwr.serverapp` module. Users are encouraged
    to use these new strategies.

Since Flower 1.10, the recommended `ServerApp` implementation would look something like
the code snippet below. Naturally, more customization can be applied to the Strategy by,
for example, reading the config from the `Context`. But to keep things focused, we will
use a simple example and assume we are federating a PyTorch model.

.. code-block:: python

    from flwr.common import Context
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig, start_server
    from flwr.server.strategy import FedAvg


    def server_fn(context: Context):

        # Instantiate strategy with initial parameters
        model = MyModel()
        parameters = ndarrays_to_parameters(
            [v.cpu().numpy() for v in model.state_dict().values()]
        )
        strategy = FedAvg(fraction_fit=0.5, initial_parameters=parameters)
        # Set number of rounds and return
        config = ServerConfig(num_rounds=3)
        return ServerAppComponents(config=config, strategy=strategy)


    # Create ServerApp with helper function
    app = ServerApp(server_fn=server_fn)

With Flower 1.21 and later, the equivalent `ServerApp` using the new Message API would
look as shown below. Note how we no longer need the `server_fn` function. The `Context`
is still accessible, allowing you to customize how the `ServerApp` behaves at runtime.
With the new strategies, a new `start` method is available. It defines a for loop which
sets the steps involved in a round of FL. By default it behaves as the original
strategies do. Note how the `start` method returns results. These are of type `Result`
and by default contain the final `global model` as well as aggregated
|metricrecord_link|_ from federated stages and, optionally, metrics from evaluation
stages done at the `ServerApp`.

.. code-block:: python
    :emphasize-lines: 3,9,10,14,17,20

    from flwr.common import ArrayRecord, ConfigRecord, Context, MetricRecord
    from flwr.server import Grid, ServerApp
    from flwr.serverapp import FedAvg

    # Create ServerApp
    app = ServerApp()


    @app.main()
    def main(grid: Grid, context: Context) -> None:

        # Defined model to federate and extract parameters
        model = MyModel()
        arrays = ArrayRecord(global_model.state_dict())

        # Instantiate strategy
        strategy = FedAvg(fraction_train=0.5)

        # Start the strategy
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=3,
        )

Update your ClientApp
---------------------
