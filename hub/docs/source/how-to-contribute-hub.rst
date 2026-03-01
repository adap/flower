Contribute Hub
==============

Have you built an exciting federated app? Now it's time to share it with the world! Contributing your federated learning app to Flower Hub allows you to share your work with the community, receive feedback, and enable others to build on top of your contributions.

There are no restrictions on the type of federated apps you can publish. Whether you've developed:

- A new federated learning algorithm  
- A novel real-world application  
- A federated agent  
- An implementation accompanying a research paper  
- A benchmark  
- A tool that enhances the federated learning ecosystem  

We welcome your contribution!


Build Your App
--------------

If you're unsure what to contribute, start by exploring the `available apps on Flower Hub <https://flower.ai/apps>`_, and see if you can add a new app that is not yet present or if you can improve an existing one.

Simulation vs. Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~

Your app should run in both **Simulation** and **Deployment** runtimes **without requiring code changes**.

To achieve this, implement separate data-loading logic for each mode while keeping the training logic identical.

One way to distinguish between Simulation and Deployment using :code:`context.node_config` inside your ClientApp. If the keys :code:`"partition-id"` and :code:`"num-partitions"` are present, the app is running in Simulation mode.

Example:

.. code-block:: python

    @app.train()
    def train(msg: Message, context: Context):
        """Train the model on local data."""

        batch_size = context.run_config["batch-size"]

        if (
            "partition-id" in context.node_config
            and "num-partitions" in context.node_config
        ):
            # Simulation Engine: partition data on the fly
            partition_id = context.node_config["partition-id"]
            num_partitions = context.node_config["num-partitions"]
            trainloader, _ = load_sim_data(
                partition_id, num_partitions, batch_size
            )
        else:
            # Deployment Engine: load demo or real user data
            data_path = context.node_config["data-path"]
            trainloader, _ = load_local_data(data_path, batch_size)

        # Training logic continues identically for both modes


.. tip::
   Recommendations:
   
   - For Simulation, use `Flower Datasets <https://flower.ai/docs/datasets/index.html>`_ to partition data on the fly.
   - For Deployment, generate demo data using the CLI command :code:`flwr-datasets create` (see the `deployment data guide <https://flower.ai/docs/datasets/how-to-generate-demo-data-for-deployment.html>`_ for details).


Update App Metadata
~~~~~~~~~~~~~~~~~~~

Before publishing your app, ensure that its metadata is complete and accurate. This information is defined in :code:`pyproject.toml` and is used by Flower Hub to display and identify your application.

Under :code:`[project]`, provide:

- :code:`name` — the app's unique name
- :code:`version` — the current release version
- :code:`description` — a short summary of the app
- :code:`license` — the applicable software license

Under :code:`[tool.flwr.app]`, specify:

- :code:`publisher` — your Flower account username

Example:

.. code-block:: toml

    [project]
    name = "my-federated-app"
    version = "0.1.0"
    description = "Federated training for medical image classification."
    license = "Apache-2.0"

    [tool.flwr.app]
    publisher = "your-username"  # Must match your Flower account username


.. warning::
   The :code:`name` and :code:`description` are publicly visible on Flower Hub.
   Choose them carefully to ensure your app is clear, descriptive, and easy to discover.
   The :code:`name` **cannot be changed** after the first publication, so make sure it is final before releasing your app.


Create a Flower Account
-----------------------

If you don't already have one, create a Flower account at: `https://flower.ai/ <https://flower.ai/>`_.

Click **Sign Up** in the top-right corner and follow the instructions. Make sure the username is same as the publisher name defined in your app's :code:`pyproject.toml`.

Publishing on behalf of an organization?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since organization accounts are not yet officially supported, please:

- Create a standard user account  
- Use your organization's name as the username (e.g., ``flwrlabs``)  
- Update your profile with the appropriate logo and description  

.. note::
   Organization accounts will be migrated once official organization support becomes available.


Publish Your App on Flower Hub
------------------------------

Apps are published using the Flower CLI.

First, ensure :code:`flwr` is installed:

.. code-block:: bash

    pip install flwr

Next, log in:

.. code-block:: bash

    flwr login

This will open a browser window where you can authenticate using your Flower account.

After logging in, publish your app:

.. code-block:: bash

    flwr app publish <your-app-path>

🎉 That's it! Your app is now live on Flower Hub.

You can view it at:

.. code-block:: text

    https://flower.ai/apps/<account_name>/<app_name>/


Publish a New Version of Your App
---------------------------------

Published apps cannot be removed from Flower Hub. However, you can release updated versions of your app to introduce improvements, new features, or bug fixes.

To publish a new version:

1. Make the necessary changes to your app's source code.
2. Update the :code:`version` field under :code:`[project]` in :code:`pyproject.toml`.

For example, if the current version is ``0.1.0``, you might update it to:

- ``0.1.1`` for a bug fix (patch release)
- ``0.2.0`` for new features (minor release)
- ``1.0.0`` for major changes

After updating the version number, publish the new release using the same command:

.. code-block:: bash

    flwr app publish <your-app-path>

The new version will be published alongside previous versions, allowing users to reference and run specific releases as needed.

We encourage you to share your app with colleagues and on social media to help grow the Flower Hub ecosystem and make federated AI more accessible to everyone.
