Use Hub
=============

Applications on Flower Hub leverage Flower's infrastructure so you can focus on application logic instead of managing distributed systems complexity.

Fetch a Hub App
---------------

First, ensure that :code:`flwr` is installed on your machine:

.. code-block:: bash

    pip install flwr

Next, fetch the app you want to run:

.. code-block:: bash

    flwr new @<account_name>/<app_name>

This command downloads the application from Flower Hub into your local environment.


Run a Hub App
-------------

After fetching the app, you can run it in either **Simulation** or **Deployment** mode without modifying the source code.

If you are new to Flower, we recommend starting with **Simulation mode**, as it requires fewer components to be launched manually. By default, :code:`flwr run` uses the Simulation Engine.


Run with the Simulation Runtime
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tip::
   See the `Simulation Engine documentation <https://flower.ai/docs/framework/how-to-run-simulations.html>`_ to learn more about running simulations, scaling the number of virtual SuperNodes, and configuring CPU/GPU resources for your ClientApp.

First, install the dependencies defined in `pyproject.toml`:

.. code-block:: bash

    cd <app_name>
    pip install -e .

Then, run the app with the default settings:

.. code-block:: bash

    flwr run .

This starts the simulation locally using Flower's built-in Simulation Engine.


Run with the Deployment Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the app using Flower's Deployment Engine, we recommend first generating demo data using `Flower Datasets <https://flower.ai/docs/datasets/how-to-generate-demo-data-for-deployment.html>`_.

Next, assign one data partition to each SuperNode:

.. code-block:: bash

    flower-supernode \
        --insecure \
        --superlink <SUPERLINK-FLEET-API> \
        --node-config="data-path=/path/to/demo_data/partition_0"

Make sure the environment of each SuperNode has all required dependencies installed.

Finally, launch the run using :code:`flwr run`, pointing to the appropriate SuperLink connection:

.. code-block:: bash

    flwr run . <SUPERLINK-CONNECTION> --stream

.. tip::
   Follow this `step-by-step guide <https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html>`_ to run the same app with the Deployment Engine.  
   Afterward, consider enabling `secure TLS connections <https://flower.ai/docs/framework/how-to-enable-tls-connections.html>`_ and configuring `SuperNode authentication <https://flower.ai/docs/framework/how-to-authenticate-supernodes.html>`_ for production deployments.


Run a Hub App Without Fetching Locally
---------------------------------------

In Deployment mode, you can run an app directly from Flower Hub without downloading it locally (after setting up SuperLink and SuperNodes as described above):

.. code-block:: bash

    flwr run @<account_name>/<app_name> --stream

In this case, the SuperLink downloads the app from Flower Hub and distributes the :code:`FAB` file to each SuperNode for execution.