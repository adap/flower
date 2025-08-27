:og:description: Upgrade seamlessly to Flower 1.21 with this guide for transitioning your setup to the latest features and enhancements powered by Flower's Message API.
.. meta::
    :description: Upgrade seamlessly to Flower 1.21 with this guide for transitioning your setup to the latest features and enhancements powered by Flower's Message API.

Upgrade to Flower with Message API
==================================

Welcome to the migration guide for updating your FlowerApps to use Flower's Message API!
This guide will walk you through the necessary steps to transition from Flower Apps
based on Strategies and NumPyClient to their equivalent using the new Message API. This
guide is relevant when updating pre-``1.21`` Flower apps to the latest stable version.

Let's dive in!

Summary of changes
------------------

Thousands of Flower Apps have been created using the Strategies and `NumPyClient
<ref-api/flwr.client.NumPyClient.html>`_ abstractions. With the introduction of the
Message API, these apps can now take advantage of a more powerful and flexible
communication layer with the `Message <ref-api/flwr.common.Message.html>`_ abstraction
being its cornerstone. Messages replace the previous `FitIns` and `FitRes` data
structures (and their equivalent for the other operations) into a single, unified and
more versatile datastructure.

To fully take advantage of the new Message API, you will need to update your app's code
to use the new message-based communication patterns. This guide will show you how to:

1. Update your `ServerApp` to make use of the new `Message`-based strategies. You won't
   need to use the `server_fn` anymore. The new strategies make it easier to customize
   how the different FL rounds are executed, to retrieve results from your run more easily, and more.
2. Update your `ClientApp` so it operates directly on `Message` objects received from
   the `ServerApp`. You will be able to keep most of the code from your `NumPyClient`
   implementation but you won't need to create a new class anymore or use the helper
   `client_fn` function.

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
