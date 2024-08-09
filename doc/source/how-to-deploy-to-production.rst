Deploy to production
====================

When developing a Flower federated learning system, a common starting point is for a data scientist/ML engineer to first test and iterate on the system in a simulated environment. Apart from using it for research, running the system in a simulation has many benefits since it is useful to be able to rigorously test and iterate on the system without going through a potentially complex setup process to spin up and connect to physical devices. After simulation, we then need to apply the system to a real-world setting, e.g. deploying it to a cohort of devices.

With the Flower framework, your Flower projects are *deployment ready*; The same modules that you've written and tested in a simulation environment can directly be deployed to production *without code changes*. 

Let's see this in action! 🌼

Setup
-----

To illustrate deployment readiness of a Flower project, we will start a Flower simulation project with one server and 10 clients. Then, once we are happy with the project, we will deploy the project *without any code changes*. 

Firstly, we define a :code:`ClientApp` that contains the components for training a PyTorch model and uses a Flower :code:`NumpyClient` :

.. code-block:: python

    from flwr.client import NumPyClient, ClientApp
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, Normalize, ToTensor

    # #############################################################################
    # 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
    # #############################################################################

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class Net(nn.Module):
        """NN Model"""
    		# [...]

    def train(net, trainloader, epochs):
        """Train the model on the training set."""
        # [...]

    def test(net, testloader):
        """Validate the model on the test set."""
        # [...]

    def load_data(partition_id):
        """Load partitioned data."""
        # [...]

    # #############################################################################
    # 2. Federation of the pipeline with Flower
    # #############################################################################

    # Load model and data
    net = Net().to(DEVICE)
    trainloader, testloader = load_data(partition_id=partition_id)

    # Define Flower client
    class FlowerClient(NumPyClient):
        # [...]

    def client_fn(cid: str):
        """Create and return an instance of Flower `Client`."""
        return FlowerClient().to_client()

    # Flower ClientApp
    app = ClientApp(client_fn=client_fn)

Since we are focused on the differences in command line execution, the detailed implementation has been intentionally omitted for simplicity. You can however, refer to the :doc:`PyTorch quickstart tutorial <tutorial-quickstart-pytorch>` for full details. What is important to note here is that we have a :code:`ClientApp` that can create multiple instances of the :code:`FlowerClient` for a federated learning workload. 

Now, for the :code:`ServerApp`, we keep the simplest structure for it, which is as follows:

.. code-block:: python

    from flwr.server import ServerApp

    app = ServerApp()

And that’s it for the :code:`ServerApp` ! For this simple example, we use the default configurations of :code:`ServerApp` .


Running the Simulation Engine
-----------------------------

With the modules above, we can start experimenting with our settings by using Flower framework’s Virtual Client Engine or VCE. This VCE allows us to develop and test a federated learning system on a single machine with shared resources before deploying the system to a real-world setting. To simulate a federated learning system with 10 clients, in a terminal, we run:

.. code-block:: shell

    flower-simulation --server-app server:app --client-app client:app --num-supernodes 10

And shortly thereafter, we will see the following logs:

.. code-block:: shell

    [...]
    INFO :      Flower ECE: Starting Driver API (gRPC-rere) on 0.0.0.0:9091
    INFO :      Registered 10 nodes
    INFO :      Supported backends: ['ray']
    INFO :      Starting Flower ServerApp, config: num_rounds=1, no round_timeout
    INFO :
    INFO :      [INIT]
    INFO :      Requesting initial parameters from one random client
    INFO :      DriverServicer.CreateRun
    INFO :      Initialising: RayBackend
    INFO :      Backend config: {'client_resources': {'num_cpus': 2, 'num_gpus': 0.0}, 'tensorflow': 0}
    2024-05-03 13:50:33,647 INFO worker.py:1749 -- Started a local Ray instance.
    INFO :      Constructed ActorPool with: 5 actors
    INFO :      Received initial parameters from one random client
    INFO :      Evaluating initial global parameters
    INFO :
    INFO :      [ROUND 1]
    INFO :      configure_fit: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_fit: received 10 results and 0 failures
    WARNING :   No fit_metrics_aggregation_fn provided
    INFO :      configure_evaluate: strategy sampled 10 clients (out of 10)
    INFO :      aggregate_evaluate: received 10 results and 0 failures
    WARNING :   No evaluate_metrics_aggregation_fn provided
    INFO :
    INFO :      [SUMMARY]
    INFO :      Run finished 1 rounds in 16.12s
    INFO :      History (loss, distributed):
    INFO :          '\tround 1: 241.09168696403503\n'
    INFO :
    WARNING :   Triggered stop event for Simulation Engine.
    INFO :      Terminated RayBackend
    INFO :      Stopping Simulation Engine now.

We can then continue to iterate on our federated learning system to achieve the performance that we desire, e.g. modifying the aggregation strategies, model architecture, hyperparameters, client sampling, checking privacy budgets, adding mods, and much more. Once we are happy with the modifications, we move to the next stage, which is deploying our Flower project to a real-world setting.


Deploying to production
-----------------------

There are a variety of architectures on which a federated learning system can be deployed to, such as to servers in organizations, hospitals, or even devices like smartphones or Raspberry Pis. Let’s say we would like to deploy the federated learning project above to hospitals, where each hospital will host a :code:`ClientApp` that trains the model on their local data. To do so, we can use the same modules that have been written above and run a set of command line instructions to deploy the :code:`ServerApp` and :code:`ClientApps`. 

First, we launch the :code:`SuperLink` that forms part of the long-running infrastructure of a Flower federated learning system:

.. code-block:: shell

    flower-superlink  --certificates \
        <your-ca-cert-filepath> \
        <your-server-cert-filepath> \
        <your-privatekey-filepath>

Then, on each respective machines/devices that the federated learning task will be executed on, start a long-running :code:`SuperNode` as follows:

.. code-block:: shell

    flower-client-app client:app \
        --root-certificates <your-ca-cert-filepath> \
        --server <server-ip>:<server-port>

(For example, in a deployment to 10 hospitals, 10 long-running :code:`SuperNodes` will be deployed in each hospital’s server/host machine).

Last but not least, launch the :code:`ServerApp` that is connected to the :code:`SuperLink` to start federated learning on the supernodes:

.. code-block:: shell

    flower-server-app server:app \
        --root-certificates <your-ca-cert-filepath> \
        --server <server-ip>:<server-port>

Congratulations! You have deployed a Flower federated learning system after testing it using the Virtual Client Engine. As you can see, deploying a Flower project from a simulation environment is extremely seamless. Notice also that for deployment, we completely reused the modules in this example without changing a single line of code. 

Of course, in real-world settings, there are other considerations to take into account such as how to ensure consistent environments for the :code:`ClientApps` , how to ensure clients are authenticated for participating in a federated learning routine, loading from local datasets, and others. To help answer some of these questions, below are a list of documentation to get you started: 

*  :doc:`How to run simulations <how-to-run-simulations>`
*  :doc:`How to run Flower using Docker <how-to-run-flower-using-docker>`
*  `How to use Local Data <https://flower.ai/docs/datasets/how-to-use-with-local-data.html>`_

..
    TODO: Add how to enable client authentication