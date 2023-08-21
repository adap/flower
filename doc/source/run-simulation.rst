Run Simulation
==============

Simulating Federated Learning workloads is useful for a multitude of use-cases: you might want to run your workload on a large cohort of clients but without having to source, configure and mange large number of physical devices; you might want to run your FL workloads as fast as possible on the compute systems you have access to without having to go through a complex setup process; you might want to validate your algorithm on different scenarios at varying levels of data and system heterogeneity, client availability, privacy budgets, etc. These are among some of the use-cases where simulating FL workloads makes sense. Flower can accommodate these scenarios by means of its `VirtualClientEngine <architecture.html#virtual-client-engine>`_ or VCE.

The :code:`VirtualClientEngine` schedules, launches and manages `virtual` clients. These clients are identical to `non-virtual` clients (i.e. the ones you launch via the command `flwr.client.start_numpy_client <apiref-flwr.html#start-numpy-client>`_) in the sense that they can be configure by creating a class inheriting, for example, from `flwr.client.NumPyClient <apiref-flwr.html#flwr.client.NumPyClient>`_ and therefore behave in an identical way. In addition to that, clients managed by the :code:`VirtualClientEngine` are:

* resource-aware: this means that each client gets assigned a portion of the compute and memory on your system. You as a user can control this at the beginning of the simulation and allows you to control the degree of parallelism of your Flower FL simulation. The fewer the resources per client, the more clients can run concurrently.
* self-managed: this means that you as a user do not need to launch clients manually, instead this gets delegated to :code:`VirtualClientEngine`'s :code:`ClientProxy`.
* ephemeral: this means that a client is only materialized when it is required in the FL process (e.g. to do :code:`fit()`). The object is destroyed afterwards, releasing the resources it was assigned and allowing in this way other clients to participate.

The :code:`VirtualClientEngine` implements `virtual` clients using `Ray <https://www.ray.io/>`_, an open-source framework for scalable Python workloads. In particular, Flower's :code:`VirtualClientEngine` makes use of `Actors <https://docs.ray.io/en/latest/ray-core/actors.html>`_ to spawn `virtual` clients and run their workload. 

Launch your Flower FL Simulation
--------------------------------

Running Flower simulations still require you to define your client class, a strategy, and utility functions to download and load (and potentially partition) your dataset. With that out of the way, launching your simulation is done with `start_simulation <apiref-flwr.html#flwr.simulation.start_simulation>`_ and a minimal example looks as follows:


.. code-block:: python

    import flwr as fl
    from flwr.server.strategy import FedAvg
    
    def client_fn(cid: str):
        # Return a standard Flower client
        return MyFlowerClient()

    # Launch the simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn, # A function to run a _virtual_ client when required
        num_clients=50, # Total number of clients available
        config=fl.server.ServerConfig(num_rounds=3), # Specify number of FL rounds
        strategy=FedAvg() # A Flower strategy
    )


VirtualClientEngine resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By default the VCE has access to all system resources (i.e. all CPUs, all GPUs, etc) since that is also the default behavior when starting Ray. However, in some settings you might want to limit how many of your system resources are used for simulation. You can do this via the :code:`ray_init_args` input argument to :code:`start_simulation` which the VCE internally passes to Ray's :code:`ray.init` command. For a complete list of settings you can configure check the `ray.init <https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html#ray-init>`_ documentation. Do not set :code:`ray_init_args` if you want the VCE to use all your system's CPUs and GPUs.

.. code-block:: python

    import flwr as fl

    # Launch the simulation by limiting resources visible to Flower's VCE
    hist = fl.simulation.start_simulation(
        ...
        # Out of all CPUs and GPUs available in your system
        # only 8xCPUs and 1xGPUs would be used for simulation.
        ray_init_args = {'num_cpus': 8, 'num_gpus': 1}
    )



Assigning Client Resources
~~~~~~~~~~~~~~~~~~~~~~~~~~
By default :code:`VirtualClientEngine` assigns a single CPU core (and nothing else) to each virtual client. This means that if your system has 10 cores, that many virtual clients can be concurrently running.

More often than not, you would probably like to adjust the resources your clients get assigned based on the complexity (i.e. compute and memory footprint) of your FL workload. You can do so when starting your simulation by passing the argument `client_resources`. Two keys are internally used by `Ray` to schedule and spawn workloads (in our case FL clients): :code:`num_cpus` indicates the number of CPU cores a client would get; and :code:`num_gpus` indicates the **ratio** of GPU memory a client gets assigned.

.. code-block:: python

    import flwr as fl

    # each client gets 1xCPU (this is the default if no resources are specified)
    my_client_resources = {'num_cpus': 1, 'num_gpus': 0.0}
    # each client gets 2xCPUs and half a GPU. (with a single GPU, 2 clients run concurrently)
    my_client_resources = {'num_cpus': 2, 'num_gpus': 0.5}
    # 10 client can run concurrently on a single GPU, but only if you have 20 CPU threads.
    my_client_resources = {'num_cpus': 2, 'num_gpus': 0.1}

    # Launch the simulation
    hist = fl.simulation.start_simulation(
        ...
        client_resources = my_client_resources # A Python dict specifying CPU/GPU resources
    )

While the :code:`client_resources` can be used to control the degree of concurrency in your FL simulation, this does not stop you from running dozens, hundreds or even thousands of clients in the same round and having orders of magnitude more `dormant` (i.e. not participating in a round). Let's say you want to have 100 clients per round but your system can only accommodate 8 clients concurrently. The :code:`VirtualClientEngine` will schedule 100 jobs to run (each simulating a client sampled by the strategy) and then will execute them in a resource-aware manner in batches of 8.

To understand all the intricate details on how resources are used to schedule FL clients and how to define custom resources, please take a look at the `Ray documentation <https://docs.ray.io/en/latest/ray-core/scheduling/resources.html>`_.

Simulation Examples
~~~~~~~~~~~~~~~~~~~

You can find read-to-run complete examples for Flower simulation in Tensorflow/Keras and PyTorch in our repository. You can run them on Google Colab too:

* `Tensorflow/Keras Simulation <https://github.com/adap/flower/tree/main/examples/simulation-tensorflow>`_: 100 clients collaboratively train a MLP model on MNIST.
* `PyTorch Simulation <https://github.com/adap/flower/tree/main/examples/simulation-pytorch>`_: 100 clients collaboratively train a CNN model on MNIST.



Multi-node Flower Simulations
-----------------------------

Flower's :code:`VirtualClientEngine` allows you to run FL simulations across multiple compute nodes. Before starting your multi-node simulation ensure that you:

#. Have the same Python environment in all nodes.
#. Have a copy of your code (e.g. your entire repo) in all nodes.
#. Have a copy of your dataset in all nodes (more about this in :ref:`simulation considerations <considerations-for-simulations>`) 
#. Pass :code:`ray_init_args={"address"="auto"}` to `start_simulation <apiref-flwr.html#flwr.simulation.start_simulation>`_ so the :code:`VirtualClientEngine` attaches to a running Ray instance.
#. Start Ray on you head node: on the terminal type :code:`ray start --head`. This command will print a few lines, one of which indicates how to attach other nodes to the head node.
#. Attach other nodes to the head node: copy the command shown after starting the head and execute it on terminal of a new node: for example :code:`ray start --address='192.168.1.132:6379'`

With all the above done, you can run your code from the head node as you would if the simulation was running on a single node.

Once your simulation is finished, if you'd like to dismantle your cluster you simple need to run the command :code:`ray stop` in each node's terminal (including the head node).

Multi-node Simulation good-to-know
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we list a few interesting functionality when running multi-node FL simulations:

User :code:`ray status` to check all nodes connected to your head node as well as the total resources available to the :code:`VirtualClientEngine`.

When attaching a new node to the head, all its resources (i.e. all CPUs, all GPUs) will be visible by the head node. This means that the :code:`VirtualClientEngine` can schedule as many _virtual_ clients as that node can possible run. In some settings you might want to exclude certain resources from the simulation. You can do this by appending `--num-cpus=<NUM_CPUS_FROM_NODE>` and/or `--num-gpus=<NUM_GPUS_FROM_NODE>` in any :code:`ray start` command (including when starting the head)

.. _considerations-for-simulations:

Considerations for Simulations
------------------------------


.. note::
  We are actively working on these fronts so to make it trivial to run any FL workload with Flower simulation.


Multi-node setups
~~~~~~~~~~~~~~~~~

* The VCE does not currently offer a way to control on which node a particular `virtual` client is executed. In other words, if more than a single node have the resources needed by a client to run, then any of those nodes could get the client workload scheduled onto. Later in the FL process (i.e. in a different round) the same client could be executed by a different node.

* By definition virtual clients are `stateless` due to their ephemeral nature. A client state can be implemented as part of the Flower client class but users need to ensure this saved to persistent storage (e.g. a database, disk) and that can be retrieve later by the same client regardless on which node it is running from.



* Fault tolerance
* Expected homogeneous nodes (as far as GPU memory is concerned) -- client-to-node pinning (TODO)
* GPU growth ?





