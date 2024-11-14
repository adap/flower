Run simulations
===============

Simulating Federated Learning workloads is useful for a multitude of use-cases: you
might want to run your workload on a large cohort of clients but without having to
source, configure and mange a large number of physical devices; you might want to run
your FL workloads as fast as possible on the compute systems you have access to without
having to go through a complex setup process; you might want to validate your algorithm
on different scenarios at varying levels of data and system heterogeneity, client
availability, privacy budgets, etc. These are among some of the use-cases where
simulating FL workloads makes sense. Flower can accommodate these scenarios by means of
its `Simulation Engine <contributor-explanation-architecture.html#simulation-engine>`_ .

The ``SimulationEngine`` schedules, launches and manages ``ClientApp`` instances. It
does so by first stargin a ``Backend``, which contains several workers (i.e. Python
processes) that can execute a ``ClientApp`` by passing it a ``Context`` and a
``Message``. These ``ClientApp`` objects are identical to those used by Flower's
`Deployment Engine <contributor-explanation-architecture.html>`_, making alternating
between _simulation_ to _deployment_ an effortless process. The execution of
``ClientApp`` objects through Flower's ``Simulation Engine`` is:

- resource-aware: this means that each backend cowrking executing ``ClientApps`` gets
  assigned a portion of the compute and memory on your system. You can define these at
  the beginning of the simulation, allowing you to control the degree of parallelism of
  your simulation. The fewer the resources per backend worker, the more ``ClientApps``
  can run concurrently on the same hardware.
- batchable: When there are more ``ClientApps`` to execute than workers the backend has,
  ``ClientApps`` are queued and executed as soon as resources get freed. This means that
  ``ClientApps`` are typically executed in batches of N, where N is the number of
  backend workers.
- self-managed: this means that you as a user do not need to launch nodes or
  ``ClientApps`` manually, instead this gets delegated to ``Simulation Engine``'s
  internals.
- ephemeral: this means that a ``ClientApp`` is only materialized when it is required by
  the application (e.g. to do `fit() <ref-api-flwr.html#flwr.client.Client.fit>`_). The
  object is destroyed afterwards, releasing the resources it was assigned and allowing
  in this way other clients to participate.

.. note::

    You can preserver the state (e.g. internal variables, parts of a ML model,
    intermediate results) of a ``ClientApp`` by saving it to its ``Context``. Check the
    `Designing Stateful Clients <how-to-design-stateful-clients.rst>`_ guide.

The ``Simulation Engine`` delegates to a ``Backend`` the role of spawning and managing
``ClientApps``. The default backend is the ``RayBackend`` which uses `Ray
<https://www.ray.io/>`_, an open-source framework for scalable Python workloads. In
particular, each worker is an `Actor
<https://docs.ray.io/en/latest/ray-core/actors.html>`_ capable of spawning a
``ClientApp`` given its ``Context`` and a ``Message`` to process.

Launch your Flower simulation
-----------------------------

Running a simulation is straightforward, in fact it is the default mode of operation for
`flwr run <ref-api-cli.html#flwr-run>`_. Therfore, running Flower simulations primarily
require you to first define a ``ClientApp`` and a ``ServerApp``. A convenient way to
generate a minimal, but fully functional, Flower app is by means of the `flwr new
<ref-api-cli.html#flwr-new>`_ command. There are multiple templates to choose from. The
example below uses the ``PyTorch`` template. With that out of the way, launching your
simulation is done with `start_simulation
<ref-api-flwr.html#flwr.simulation.start_simulation>`_ and a minimal example is shown
below.

.. tip::

    If you haven't already, install flower via ``pip install -U flwr`` on a Python
    environement.

.. code-block:: shell

    # or simply execute `flwr run` for a fully interactive process
    flwr new my-app --framework="PyTorch" --username="alice"

Then follow the instructions shown after completing the ``flwr new`` command. When you
execute ``flwr run``, you'll be using the ``Simulation Egnine``.

If we take a look at the `pyproject.toml` that was generated from the ``flwr new``
command (and loaded upon ``flwr run`` execution), we see that a _default_ federation is
defined. It sets the number of supernodes to 10.

.. code-block:: toml

    [tool.flwr.federations]
    default = "local-simulation"

    [tool.flwr.federations.local-simulation]
    options.num-supernodes = 10

You can modify the size of your simulations by adjusting ``options.num-supernodes``.

Defining backend resources
~~~~~~~~~~~~~~~~~~~~~~~~~~

By default the ``Simulation Engine`` assigns a two CPU cores to each backend worker.
This means that if your system has 10 CPU cores, five backend workers can be running in
parallel, each executing a different ``ClientApp`` instance.

More often than not, you would probably like to adjust the resources your ``ClientApp``
get assigned based on the complexity (i.e. compute and memory footprint) of your
workload. You can do so by adjusting the backend resources for your federation.

.. caution::

    Note that the resources the backend assigns to each worker (and hence to each
    ``ClientApp`` being executed) is performed in a _soft_ manner. This means that the
    resources are primarily taken into account in order to control the degree of
    parallelism at which ``ClientApp`` instances should be executed. Resource
    assignation is **not strict**, meaning that if you specified your ``ClientApp`` to
    make use of 25% of the available VRAM but it ends up using 50%, it might make other
    ``ClientApp`` instances running to crash.

Customizing resources can be done directly on the `pyproject.toml` of your app.

.. code-block:: toml

    [tool.flwr.federations.local-simulation]
    options.num-supernodes = 10
    options.backend.client-resources.num-cpus = 1 # each ClientApp assumes to use 1CPUs (default is 2)
    options.backend.client-resources.num-gpus = 0.0 # no GPU access to the ClientApp (default is 0.0)

With the above backend settings, your simulation will run as many ``ClientApps`` in
parallel as CPUs you have in your system. GPU resources for your ``ClientApp`` can be
assigned by specifying the **ratio** of VRAM each should make use of.

.. code-block:: toml

    [tool.flwr.federations.local-simulation]
    options.num-supernodes = 10
    options.backend.client-resources.num-cpus = 1 # each ClientApp assumes to use 1CPUs (default is 2)
    options.backend.client-resources.num-gpus = 0.25 # each ClientApp uses 25% of VRAM (default is 0.0)

Let's see how the above configurate results in a different number of ``ClientApps``
running in parallel depending on the resources available in your system. If your system
has:

- 10x CPUs and 1x GPU: at most 4 ``ClientApps`` will run in parallel since each require
  25% of the available VRAM.
- 10x CPUs and 2x GPUs: at most 8 ``ClientApps`` will run in parallel.
- 6x CPUs and 2x GPUs: at most 6 ``ClientApps`` will run in parallel.
- 10x CPUs but 0x GPUs: you won't be able to run the simulation since not a single
  ``ClientApp`` will be able to run.

A generalization of this is given by the following equation. It gives the maximum number
of ``ClientApps`` that can be executed in parallel on available CPU cores (SYS_CPUS) and
VRAM (SYS_GPUS).

.. math::

    N = \min\left(\left\lfloor \frac{\text{SYS_CPUS}}{\text{num_cpus}} \right\rfloor, \left\lfloor \frac{\text{SYS_GPUS}}{\text{num_gpus}} \right\rfloor\right)

Both ``num_cpus`` (an integer higher than 1) and ``num_gpus`` (a non-negative real
number) should be set in a per ``ClientApp`` basis. If, for example you want only a
single ``ClientApp`` to run in each GPU, then set ``num_gpus=1.0``. If, for example a
``ClientApp`` requires access to two whole GPUs you'd set ``num_gpus=2``.

While the ``options.backend.client-resources`` can be used to control the degree of
concurrency in your simulations, this does not stop you from running hundreds or even
thousands of clients in the same round and having orders of magnitude more `dormant`
(i.e. not participating in a round) clients. Let's say you want to have 100 clients per
round but your system can only accommodate 8 clients concurrently. The
``SimulationEngine`` will schedule 100 ``ClientApps`` to run and then will execute them
in a resource-aware manner in batches of 8.

Simulation Engine resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default the ``SimulationEngine`` has **access to all system resources** (i.e. all
CPUs, all GPUs). However, in some settings you might want to limit how many of your
system resources are used for simulation. You can do this in the ``pyproject.toml`` of
your app.

.. code-block:: toml

    [tool.flwr.federations.local-simulation]
    options.num-supernodes = 10
    options.backend.client-resources.num-cpus = 1 # Each ClientApp will get assigned 1 CPU core
    options.backend.client-resources.num-gpus = 0.5 # Each ClientApp will get 50% of each available GPU
    options.backend.init_args.num_cpus = 1
    options.backend.init_args.num_gpus = 1

With the above setup, the Backend will be initialized with a single CPU and GPU.
Therefore, even if more CPUs and GPUs are avaialabel in your system, they will not be
used for the simulation. The example above results in a singe ``ClientApp`` running at
any give point.

For a complete list of settings you can configure check the `ray.init
<https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html#ray-init>`_ documentation.

For the highest performance, do not set ``options.backend.init_args``.

Simulation examples
~~~~~~~~~~~~~~~~~~~

In addition to the quickstart tutorials in the documentation (e.g `quickstart PyTorch
Tutorial <tutorial-quickstart-pytorch.html>`_, `quickstart JAX
<tutorial-quickstart-jax.html>`_), most examples in the Flower repository are
simulation-ready.

- `Quickstart Tensorflow/Keras
  <https://github.com/adap/flower/tree/main/examples/quickstart-tensorflow>`_.
- `Quickstart Pytorch
  <https://github.com/adap/flower/tree/main/examples/quickstart-pytorch>`_
- `Advanced PyTorch
  <https://github.com/adap/flower/tree/main/examples/advanced-pytorch>`_
- `Quickstart MLX <https://github.com/adap/flower/tree/main/examples/quickstart-mlx>`_
- `ViT finetuning <https://github.com/adap/flower/tree/main/examples/flowertune-vit>`_

The complete list of examples can be found in `the Flower GitHub
<https://github.com/adap/flower/tree/main/examples>`_.

Simulation with Colab/Jupyter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-node Flower simulations
-----------------------------

Flower's ``VirtualClientEngine`` allows you to run FL simulations across multiple
compute nodes. Before starting your multi-node simulation ensure that you:

1. Have the same Python environment in all nodes.
2. Have a copy of your code (e.g. your entire repo) in all nodes.
3. Have a copy of your dataset in all nodes (more about this in :ref:`simulation
   considerations <considerations-for-simulations>`)
4. Pass ``ray_init_args={"address"="auto"}`` to `start_simulation
   <ref-api-flwr.html#flwr.simulation.start_simulation>`_ so the ``VirtualClientEngine``
   attaches to a running Ray instance.
5. Start Ray on you head node: on the terminal type ``ray start --head``. This command
   will print a few lines, one of which indicates how to attach other nodes to the head
   node.
6. Attach other nodes to the head node: copy the command shown after starting the head
   and execute it on terminal of a new node: for example ``ray start
   --address='192.168.1.132:6379'``

With all the above done, you can run your code from the head node as you would if the
simulation was running on a single node.

Once your simulation is finished, if you'd like to dismantle your cluster you simply
need to run the command ``ray stop`` in each node's terminal (including the head node).

Multi-node simulation good-to-know
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we list a few interesting functionality when running multi-node FL simulations:

User ``ray status`` to check all nodes connected to your head node as well as the total
resources available to the ``VirtualClientEngine``.

When attaching a new node to the head, all its resources (i.e. all CPUs, all GPUs) will
be visible by the head node. This means that the ``VirtualClientEngine`` can schedule as
many `virtual` clients as that node can possible run. In some settings you might want to
exclude certain resources from the simulation. You can do this by appending
`--num-cpus=<NUM_CPUS_FROM_NODE>` and/or `--num-gpus=<NUM_GPUS_FROM_NODE>` in any ``ray
start`` command (including when starting the head)

.. _considerations-for-simulations:

Considerations for simulations
------------------------------

.. note::

    We are actively working on these fronts so to make it trivial to run any FL workload
    with Flower simulation.

The current VCE allows you to run Federated Learning workloads in simulation mode
whether you are prototyping simple scenarios on your personal laptop or you want to
train a complex FL pipeline across multiple high-performance GPU nodes. While we add
more capabilities to the VCE, the points below highlight some of the considerations to
keep in mind when designing your FL pipeline with Flower. We also highlight a couple of
current limitations in our implementation.

GPU resources
~~~~~~~~~~~~~

The VCE assigns a share of GPU memory to a client that specifies the key ``num_gpus`` in
``client_resources``. This being said, Ray (used internally by the VCE) is by default:

- not aware of the total VRAM available on the GPUs. This means that if you set
  ``num_gpus=0.5`` and you have two GPUs in your system with different (e.g. 32GB and
  8GB) VRAM amounts, they both would run 2 clients concurrently.
- not aware of other unrelated (i.e. not created by the VCE) workloads are running on
  the GPU. Two takeaways from this are:

  - Your Flower server might need a GPU to evaluate the `global model` after aggregation
    (by instance when making use of the `evaluate method
    <how-to-implement-strategies.html#the-evaluate-method>`_)
  - If you want to run several independent Flower simulations on the same machine you
    need to mask-out your GPUs with ``CUDA_VISIBLE_DEVICES="<GPU_IDs>"`` when launching
    your experiment.

In addition, the GPU resource limits passed to ``client_resources`` are not `enforced`
(i.e. they can be exceeded) which can result in the situation of client using more VRAM
than the ratio specified when starting the simulation.

TensorFlow with GPUs
++++++++++++++++++++

When `using a GPU with TensorFlow <https://www.tensorflow.org/guide/gpu>`_ nearly your
entire GPU memory of all your GPUs visible to the process will be mapped. This is done
by TensorFlow for optimization purposes. However, in settings such as FL simulations
where we want to split the GPU into multiple `virtual` clients, this is not a desirable
mechanism. Luckily we can disable this default behavior by `enabling memory growth
<https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth>`_.

This would need to be done in the main process (which is where the server would run) and
in each Actor created by the VCE. By means of ``actor_kwargs`` we can pass the reserved
key `"on_actor_init_fn"` in order to specify a function to be executed upon actor
initialization. In this case, to enable GPU growth for TF workloads. It would look as
follows:

.. code-block:: python

    import flwr as fl
    from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth

    # Enable GPU growth in the main thread (the one used by the
    # server to quite likely run global evaluation using GPU)
    enable_tf_gpu_growth()

    # Start Flower simulation
    hist = fl.simulation.start_simulation(
        # ...
        actor_kwargs={
            "on_actor_init_fn": enable_tf_gpu_growth  # <-- To be executed upon actor init.
        },
    )

This is precisely the mechanism used in `Tensorflow/Keras Simulation
<https://github.com/adap/flower/tree/main/examples/simulation-tensorflow>`_ example.

Multi-node setups
~~~~~~~~~~~~~~~~~

- The VCE does not currently offer a way to control on which node a particular `virtual`
  client is executed. In other words, if more than a single node have the resources
  needed by a client to run, then any of those nodes could get the client workload
  scheduled onto. Later in the FL process (i.e. in a different round) the same client
  could be executed by a different node. Depending on how your clients access their
  datasets, this might require either having a copy of all dataset partitions on all
  nodes or a dataset serving mechanism (e.g. using nfs, a database) to circumvent data
  duplication.
- By definition virtual clients are `stateless` due to their ephemeral nature. A client
  state can be implemented as part of the Flower client class but users need to ensure
  this saved to persistent storage (e.g. a database, disk) and that can be retrieve
  later by the same client regardless on which node it is running from. This is related
  to the point above also since, in some way, the client's dataset could be seen as a
  type of `state`.
