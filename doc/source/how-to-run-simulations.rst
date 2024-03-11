Run simulations
===============

..  youtube:: cRebUIGB5RU
   :url_parameters: ?list=PLNG4feLHqCWlnj8a_E1A_n5zr2-8pafTB
   :width: 100%

Simulating Federated Learning workloads is useful for a multitude of use-cases: you might want to run your workload on a large cohort of clients but without having to source, configure and mange a large number of physical devices; you might want to run your FL workloads as fast as possible on the compute systems you have access to without having to go through a complex setup process; you might want to validate your algorithm on different scenarios at varying levels of data and system heterogeneity, client availability, privacy budgets, etc. These are among some of the use-cases where simulating FL workloads makes sense. Flower can accommodate these scenarios by means of its `VirtualClientEngine <contributor-explanation-architecture.html#virtual-client-engine>`_ or VCE.

The :code:`VirtualClientEngine` schedules, launches and manages `virtual` clients. These clients are identical to `non-virtual` clients (i.e. the ones you launch via the command `flwr.client.start_client <ref-api-flwr.html#start-client>`_) in the sense that they can be configure by creating a class inheriting, for example, from `flwr.client.NumPyClient <ref-api-flwr.html#flwr.client.NumPyClient>`_ and therefore behave in an identical way. In addition to that, clients managed by the :code:`VirtualClientEngine` are:

* resource-aware: this means that each client gets assigned a portion of the compute and memory on your system. You as a user can control this at the beginning of the simulation and allows you to control the degree of parallelism of your Flower FL simulation. The fewer the resources per client, the more clients can run concurrently on the same hardware.
* self-managed: this means that you as a user do not need to launch clients manually, instead this gets delegated to :code:`VirtualClientEngine`'s internals.
* ephemeral: this means that a client is only materialized when it is required in the FL process (e.g. to do `fit() <ref-api-flwr.html#flwr.client.Client.fit>`_). The object is destroyed afterwards, releasing the resources it was assigned and allowing in this way other clients to participate.

The :code:`VirtualClientEngine` implements `virtual` clients using `Ray <https://www.ray.io/>`_, an open-source framework for scalable Python workloads. In particular, Flower's :code:`VirtualClientEngine` makes use of `Actors <https://docs.ray.io/en/latest/ray-core/actors.html>`_ to spawn `virtual` clients and run their workload. 


Launch your Flower simulation
-----------------------------

Running Flower simulations still require you to define your client class, a strategy, and utility functions to download and load (and potentially partition) your dataset. With that out of the way, launching your simulation is done with `start_simulation <ref-api-flwr.html#flwr.simulation.start_simulation>`_ and a minimal example looks as follows:


.. code-block:: python

    import flwr as fl
    from flwr.server.strategy import FedAvg
    
    def client_fn(cid: str):
        # Return a standard Flower client
        return MyFlowerClient().to_client()

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
        # Out of all CPUs and GPUs available in your system,
        # only 8xCPUs and 1xGPUs would be used for simulation.
        ray_init_args = {'num_cpus': 8, 'num_gpus': 1}
    )



Assigning client resources
~~~~~~~~~~~~~~~~~~~~~~~~~~
By default the :code:`VirtualClientEngine` assigns a single CPU core (and nothing else) to each virtual client. This means that if your system has 10 cores, that many virtual clients can be concurrently running.

More often than not, you would probably like to adjust the resources your clients get assigned based on the complexity (i.e. compute and memory footprint) of your FL workload. You can do so when starting your simulation by setting the argument `client_resources` to `start_simulation <ref-api-flwr.html#flwr.simulation.start_simulation>`_. Two keys are internally used by Ray to schedule and spawn workloads (in our case Flower clients): 

* :code:`num_cpus` indicates the number of CPU cores a client would get.
* :code:`num_gpus` indicates the **ratio** of GPU memory a client gets assigned.

Let's see a few examples:

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

While the :code:`client_resources` can be used to control the degree of concurrency in your FL simulation, this does not stop you from running dozens, hundreds or even thousands of clients in the same round and having orders of magnitude more `dormant` (i.e. not participating in a round) clients. Let's say you want to have 100 clients per round but your system can only accommodate 8 clients concurrently. The :code:`VirtualClientEngine` will schedule 100 jobs to run (each simulating a client sampled by the strategy) and then will execute them in a resource-aware manner in batches of 8.

To understand all the intricate details on how resources are used to schedule FL clients and how to define custom resources, please take a look at the `Ray documentation <https://docs.ray.io/en/latest/ray-core/scheduling/resources.html>`_.

Simulation examples
~~~~~~~~~~~~~~~~~~~

A few ready-to-run complete examples for Flower simulation in Tensorflow/Keras and PyTorch are provided in the `Flower repository <https://github.com/adap/flower>`_. You can run them on Google Colab too:

* `Tensorflow/Keras Simulation <https://github.com/adap/flower/tree/main/examples/simulation-tensorflow>`_: 100 clients collaboratively train a MLP model on MNIST.
* `PyTorch Simulation <https://github.com/adap/flower/tree/main/examples/simulation-pytorch>`_: 100 clients collaboratively train a CNN model on MNIST.



Multi-node Flower simulations
-----------------------------

Flower's :code:`VirtualClientEngine` allows you to run FL simulations across multiple compute nodes. Before starting your multi-node simulation ensure that you:

#. Have the same Python environment in all nodes.
#. Have a copy of your code (e.g. your entire repo) in all nodes.
#. Have a copy of your dataset in all nodes (more about this in :ref:`simulation considerations <considerations-for-simulations>`) 
#. Pass :code:`ray_init_args={"address"="auto"}` to `start_simulation <ref-api-flwr.html#flwr.simulation.start_simulation>`_ so the :code:`VirtualClientEngine` attaches to a running Ray instance.
#. Start Ray on you head node: on the terminal type :code:`ray start --head`. This command will print a few lines, one of which indicates how to attach other nodes to the head node.
#. Attach other nodes to the head node: copy the command shown after starting the head and execute it on terminal of a new node: for example :code:`ray start --address='192.168.1.132:6379'`

With all the above done, you can run your code from the head node as you would if the simulation was running on a single node.

Once your simulation is finished, if you'd like to dismantle your cluster you simply need to run the command :code:`ray stop` in each node's terminal (including the head node).

Multi-node simulation good-to-know
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we list a few interesting functionality when running multi-node FL simulations:

User :code:`ray status` to check all nodes connected to your head node as well as the total resources available to the :code:`VirtualClientEngine`.

When attaching a new node to the head, all its resources (i.e. all CPUs, all GPUs) will be visible by the head node. This means that the :code:`VirtualClientEngine` can schedule as many `virtual` clients as that node can possible run. In some settings you might want to exclude certain resources from the simulation. You can do this by appending `--num-cpus=<NUM_CPUS_FROM_NODE>` and/or `--num-gpus=<NUM_GPUS_FROM_NODE>` in any :code:`ray start` command (including when starting the head)

.. _considerations-for-simulations:


Considerations for simulations
------------------------------

.. note::
  We are actively working on these fronts so to make it trivial to run any FL workload with Flower simulation.


The current VCE allows you to run Federated Learning workloads in simulation mode whether you are prototyping simple scenarios on your personal laptop or you want to train a complex FL pipeline across multiple high-performance GPU nodes. While we add more capabilities to the VCE, the points below highlight some of the considerations to keep in mind when designing your FL pipeline with Flower. We also highlight a couple of current limitations in our implementation.

GPU resources
~~~~~~~~~~~~~

The VCE assigns a share of GPU memory to a client that specifies the key :code:`num_gpus` in :code:`client_resources`. This being said, Ray (used internally by the VCE) is by default:


*   not aware of the total VRAM available on the GPUs. This means that if you set :code:`num_gpus=0.5` and you have two GPUs in your system with different (e.g. 32GB and 8GB) VRAM amounts, they both would run 2 clients concurrently.
*   not aware of other unrelated (i.e. not created by the VCE) workloads are running on the GPU. Two takeaways from this are:

    *    Your Flower server might need a GPU to evaluate the `global model` after aggregation (by instance when making use of the `evaluate method <how-to-implement-strategies.html#the-evaluate-method>`_)
    *    If you want to run several independent Flower simulations on the same machine you need to mask-out your GPUs with :code:`CUDA_VISIBLE_DEVICES="<GPU_IDs>"` when launching your experiment. 


In addition, the GPU resource limits passed to :code:`client_resources` are not `enforced` (i.e. they can be exceeded) which can result in the situation of client using more VRAM than the ratio specified when starting the simulation. 

TensorFlow with GPUs
""""""""""""""""""""

When `using a GPU with TensorFlow <https://www.tensorflow.org/guide/gpu>`_ nearly your entire GPU memory of all your GPUs visible to the process will be mapped. This is done by TensorFlow for optimization purposes. However, in settings such as FL simulations where we want to split the GPU into multiple `virtual` clients, this is not a desirable mechanism. Luckily we can disable this default behavior by `enabling memory growth <https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth>`_. 

This would need to be done in the main process (which is where the server would run) and in each Actor created by the VCE. By means of :code:`actor_kwargs` we can pass the reserved key `"on_actor_init_fn"` in order to specify a function to be executed upon actor initialization. In this case, to enable GPU growth for TF workloads. It would look as follows:

.. code-block:: python

    import flwr as fl
    from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

    # Enable GPU growth in the main thread (the one used by the
    # server to quite likely run global evaluation using GPU)
    enable_tf_gpu_growth()

    # Start Flower simulation
    hist = fl.simulation.start_simulation(
        ...
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth # <-- To be executed upon actor init.
        },
    )

This is precisely the mechanism used in `Tensorflow/Keras Simulation <https://github.com/adap/flower/tree/main/examples/simulation-tensorflow>`_ example.


Multi-node setups
~~~~~~~~~~~~~~~~~

* The VCE does not currently offer a way to control on which node a particular `virtual` client is executed. In other words, if more than a single node have the resources needed by a client to run, then any of those nodes could get the client workload scheduled onto. Later in the FL process (i.e. in a different round) the same client could be executed by a different node. Depending on how your clients access their datasets, this might require either having a copy of all dataset partitions on all nodes or a dataset serving mechanism (e.g. using nfs, a database) to circumvent data duplication. 

* By definition virtual clients are `stateless` due to their ephemeral nature. A client state can be implemented as part of the Flower client class but users need to ensure this saved to persistent storage (e.g. a database, disk) and that can be retrieve later by the same client regardless on which node it is running from. This is related to the point above also since, in some way, the client's dataset could be seen as a type of `state`.

