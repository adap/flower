.. |clientapp_link| replace:: ``ClientApp``

.. |message_link| replace:: ``Message``

.. |context_link| replace:: ``Context``

.. |flwr_run_link| replace:: ``flwr run``

.. |flwr_new_link| replace:: ``flwr new``

.. _clientapp_link: ref-api/flwr.client.ClientApp.html

.. _context_link: ref-api/flwr.common.Context.html

.. _flwr_new_link: ref-api-cli.html#flwr-new

.. _flwr_run_link: ref-api-cli.html#flwr-run

.. _message_link: ref-api/flwr.common.Message.html

Run simulations
===============

Simulating Federated Learning workloads is useful for a multitude of use cases: you
might want to run your workload on a large cohort of clients without having to source,
configure, and manage a large number of physical devices; you might want to run your FL
workloads as fast as possible on the compute systems you have access to without going
through a complex setup process; you might want to validate your algorithm in different
scenarios at varying levels of data and system heterogeneity, client availability,
privacy budgets, etc. These are among some of the use cases where simulating FL
workloads makes sense.

Flower's ``Simulation Engine`` schedules, launches, and manages |clientapp_link|_
instances. It does so through a ``Backend``, which contains several workers (i.e.,
Python processes) that can execute a ``ClientApp`` by passing it a |context_link|_ and a
|message_link|_. These ``ClientApp`` objects are identical to those used by Flower's
`Deployment Engine <contributor-explanation-architecture.html>`_, making alternating
between *simulation* and *deployment* an effortless process. The execution of
``ClientApp`` objects through Flower's ``Simulation Engine`` is:

- **Resource-aware**: Each backend worker executing ``ClientApp``\s gets assigned a
  portion of the compute and memory on your system. You can define these at the
  beginning of the simulation, allowing you to control the degree of parallelism of your
  simulation. For a fixed total pool of resources, the fewer the resources per backend
  worker, the more ``ClientApps`` can run concurrently on the same hardware.
- **Batchable**: When there are more ``ClientApps`` to execute than backend workers,
  ``ClientApps`` are queued and executed as soon as resources are freed. This means that
  ``ClientApps`` are typically executed in batches of N, where N is the number of
  backend workers.
- **Self-managed**: This means that you, as a user, do not need to launch ``ClientApps``
  manually; instead, the ``Simulation Engine``'s internals orchestrates the execution of
  all ``ClientApp``\s.
- **Ephemeral**: This means that a ``ClientApp`` is only materialized when it is
  required by the application (e.g., to do `fit()
  <ref-api-flwr.html#flwr.client.Client.fit>`_). The object is destroyed afterward,
  releasing the resources it was assigned and allowing other clients to participate.

.. note::

    You can preserve the state (e.g., internal variables, parts of an ML model,
    intermediate results) of a ``ClientApp`` by saving it to its ``Context``. Check the
    `Designing Stateful Clients <how-to-design-stateful-clients.rst>`_ guide for a
    complete walkthrough.

The ``Simulation Engine`` delegates to a ``Backend`` the role of spawning and managing
``ClientApps``. The default backend is the ``RayBackend``, which uses `Ray
<https://www.ray.io/>`_, an open-source framework for scalable Python workloads. In
particular, each worker is an `Actor
<https://docs.ray.io/en/latest/ray-core/actors.html>`_ capable of spawning a
``ClientApp`` given its ``Context`` and a ``Message`` to process.

Launch your Flower simulation
-----------------------------

Running a simulation is straightforward; in fact, it is the default mode of operation
for |flwr_run_link|_. Therefore, running Flower simulations primarily requires you to
first define a ``ClientApp`` and a ``ServerApp``. A convenient way to generate a minimal
but fully functional Flower app is by means of the |flwr_new_link|_ command. There are
multiple templates to choose from. The example below uses the ``PyTorch`` template.

.. tip::

    If you haven't already, install Flower via ``pip install -U flwr`` in a Python
    environment.

.. code-block:: shell

    # or simply execute `flwr run` for a fully interactive process
    flwr new my-app --framework="PyTorch" --username="alice"

Then, follow the instructions shown after completing the |flwr_new_link|_ command. When
you execute |flwr_run_link|_, you'll be using the ``Simulation Engine``.

If we take a look at the ``pyproject.toml`` that was generated from the |flwr_new_link|_
command (and loaded upon |flwr_run_link|_ execution), we see that a *default* federation
is defined. It sets the number of supernodes to 10.

.. code-block:: toml

    [tool.flwr.federations]
    default = "local-simulation"

    [tool.flwr.federations.local-simulation]
    options.num-supernodes = 10

You can modify the size of your simulations by adjusting ``options.num-supernodes``.

Simulation examples
~~~~~~~~~~~~~~~~~~~

In addition to the quickstart tutorials in the documentation (e.g., `quickstart PyTorch
Tutorial <tutorial-quickstart-pytorch.html>`_, `quickstart JAX Tutorial
<tutorial-quickstart-jax.html>`_), most examples in the Flower repository are
simulation-ready.

- `Quickstart TensorFlow/Keras
  <https://github.com/adap/flower/tree/main/examples/quickstart-tensorflow>`_.
- `Quickstart PyTorch
  <https://github.com/adap/flower/tree/main/examples/quickstart-pytorch>`_
- `Advanced PyTorch
  <https://github.com/adap/flower/tree/main/examples/advanced-pytorch>`_
- `Quickstart MLX <https://github.com/adap/flower/tree/main/examples/quickstart-mlx>`_
- `ViT fine-tuning <https://github.com/adap/flower/tree/main/examples/flowertune-vit>`_

The complete list of examples can be found in `the Flower GitHub
<https://github.com/adap/flower/tree/main/examples>`_.

.. _clientappresources:

Defining ``ClientApp`` resources
--------------------------------

By default, the ``Simulation Engine`` assigns two CPU cores to each backend worker. This
means that if your system has 10 CPU cores, five backend workers can be running in
parallel, each executing a different ``ClientApp`` instance.

More often than not, you would probably like to adjust the resources your ``ClientApp``
gets assigned based on the complexity (i.e., compute and memory footprint) of your
workload. You can do so by adjusting the backend resources for your federation.

.. caution::

    Note that the resources the backend assigns to each worker (and hence to each
    ``ClientApp`` being executed) are assigned in a *soft* manner. This means that the
    resources are primarily taken into account in order to control the degree of
    parallelism at which ``ClientApp`` instances should be executed. Resource assignment
    is **not strict**, meaning that if you specified your ``ClientApp`` is assumed to
    make use of 25% of the available VRAM but it ends up using 50%, it might cause other
    ``ClientApp`` instances to crash throwing an out-of-memory (OOM) error.

Customizing resources can be done directly in the ``pyproject.toml`` of your app.

.. code-block:: toml

    [tool.flwr.federations.local-simulation]
    options.num-supernodes = 10
    options.backend.client-resources.num-cpus = 1 # each ClientApp assumes to use 1 CPU (default is 2)
    options.backend.client-resources.num-gpus = 0.0 # no GPU access to the ClientApp (default is 0.0)

With the above backend settings, your simulation will run as many ``ClientApps`` in
parallel as CPUs you have in your system. GPU resources for your ``ClientApp`` can be
assigned by specifying the **ratio** of VRAM each should make use of.

.. code-block:: toml

    [tool.flwr.federations.local-simulation]
    options.num-supernodes = 10
    options.backend.client-resources.num-cpus = 1 # each ClientApp assumes to use 1 CPU (default is 2)
    options.backend.client-resources.num-gpus = 0.25 # each ClientApp uses 25% of VRAM (default is 0.0)

.. note::

    If you are using TensorFlow, you need to `enable memory growth
    <https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth>`_ so multiple
    ``ClientApp`` instances can share a GPU. This needs to be done before launching the
    simulation. To do so, set the environment variable
    ``TF_FORCE_GPU_ALLOW_GROWTH="1"``.

Let's see how the above configuration results in a different number of ``ClientApps``
running in parallel depending on the resources available in your system. If your system
has:

- 10x CPUs and 1x GPU: at most 4 ``ClientApps`` will run in parallel since each requires
  25% of the available VRAM.
- 10x CPUs and 2x GPUs: at most 8 ``ClientApps`` will run in parallel (VRAM-limited).
- 6x CPUs and 4x GPUs: at most 6 ``ClientApps`` will run in parallel (CPU-limited).
- 10x CPUs but 0x GPUs: you won't be able to run the simulation since not even the
  resources for a single ``ClientApp`` can be met.

A generalization of this is given by the following equation. It gives the maximum number
of ``ClientApps`` that can be executed in parallel on available CPU cores (SYS_CPUS) and
VRAM (SYS_GPUS).

.. math::

    N = \min\left(\left\lfloor \frac{\text{SYS_CPUS}}{\text{num_cpus}} \right\rfloor, \left\lfloor \frac{\text{SYS_GPUS}}{\text{num_gpus}} \right\rfloor\right)

Both ``num_cpus`` (an integer higher than 1) and ``num_gpus`` (a non-negative real
number) should be set on a per ``ClientApp`` basis. If, for example, you want only a
single ``ClientApp`` to run on each GPU, then set ``num_gpus=1.0``. If, for example, a
``ClientApp`` requires access to two whole GPUs, you'd set ``num_gpus=2``.

While the ``options.backend.client-resources`` can be used to control the degree of
concurrency in your simulations, this does not stop you from running hundreds or even
thousands of clients in the same round and having orders of magnitude more *dormant*
(i.e., not participating in a round) clients. Let's say you want to have 100 clients per
round but your system can only accommodate 8 clients concurrently. The ``Simulation
Engine`` will schedule 100 ``ClientApps`` to run and then will execute them in a
resource-aware manner in batches of 8.

Simulation Engine resources
---------------------------

By default, the ``Simulation Engine`` has **access to all system resources** (i.e., all
CPUs, all GPUs). However, in some settings, you might want to limit how many of your
system resources are used for simulation. You can do this in the ``pyproject.toml`` of
your app by setting the ``options.backend.init_args`` variable.

.. code-block:: toml

    [tool.flwr.federations.local-simulation]
    options.num-supernodes = 10
    options.backend.client-resources.num-cpus = 1 # Each ClientApp will get assigned 1 CPU core
    options.backend.client-resources.num-gpus = 0.5 # Each ClientApp will get 50% of each available GPU
    options.backend.init_args.num_cpus = 1 # Only expose 1 CPU to the simulation
    options.backend.init_args.num_gpus = 1 # Expose a single GPU to the simulation

With the above setup, the Backend will be initialized with a single CPU and GPU.
Therefore, even if more CPUs and GPUs are available in your system, they will not be
used for the simulation. The example above results in a single ``ClientApp`` running at
any given point.

For a complete list of settings you can configure, check the `ray.init
<https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html#ray-init>`_ documentation.

For the highest performance, do not set ``options.backend.init_args``.

Simulation in Colab/Jupyter
---------------------------

The preferred way of running simulations should always be |flwr_run_link|_. However, the
core functionality of the ``Simulation Engine`` can be used from within a Google Colab
or Jupyter environment by means of `run_simulation
<ref-api-flwr.html#flwr.simulation.run_simulation>`_.

.. code-block:: python

    from flwr.simulation import run_simulation

    # Construct the ClientApp passing the client generation function
    client_app = ClientApp(client_fn=client_fn)

    # Create your ServerApp passing the server generation function
    server_app = ServerApp(server_fn=server_fn)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=10,  # equivalent to setting `num-supernodes` in the pyproject.toml
    )

With ``run_simulation``, you can also control the amount of resources for your
``ClientApp`` instances. Do so by setting ``backend_config``. If unset, the default
resources are assigned (i.e., 2xCPUs per ``ClientApp`` and no GPU).

.. code-block:: python

    run_simulation(
        # ...
        backend_config={"client_resources": {"num_cpus": 2, "num_gpus": 0.25}}
    )

Refer to the `30 minutes Federated AI Tutorial
<https://colab.research.google.com/github/adap/flower/blob/main/examples/flower-in-30-minutes/tutorial.ipynb>`_
for a complete example on how to run Flower Simulations in Colab.

.. _multinodesimulations:

Multi-node Flower simulations
-----------------------------

Flower's ``Simulation Engine`` allows you to run FL simulations across multiple compute
nodes so that you're not restricted to running simulations on a _single_ machine. Before
starting your multi-node simulation, ensure that you:

1. Have the same Python environment on all nodes.
2. Have a copy of your code on all nodes.
3. Have a copy of your dataset on all nodes. If you are using partitions from `Flower
   Datasets <https://flower.ai/docs/datasets>`_, ensure the partitioning strategy its
   parameterization are the same. The expectation is that the i-th dataset partition is
   identical in all nodes.
4. Start Ray on your head node: on the terminal, type ``ray start --head``. This command
   will print a few lines, one of which indicates how to attach other nodes to the head
   node.
5. Attach other nodes to the head node: copy the command shown after starting the head
   and execute it on the terminal of a new node (before executing |flwr_run_link|_). For
   example: ``ray start --address='192.168.1.132:6379'``. Note that to be able to attach
   nodes to the head node they should be discoverable by each other.

With all the above done, you can run your code from the head node as you would if the
simulation were running on a single node. In other words:

.. code-block:: shell

    # From your head node, launch the simulation
    flwr run

Once your simulation is finished, if you'd like to dismantle your cluster, you simply
need to run the command ``ray stop`` in each node's terminal (including the head node).

.. note::

    When attaching a new node to the head, all its resources (i.e., all CPUs, all GPUs)
    will be visible by the head node. This means that the ``Simulation Engine`` can
    schedule as many ``ClientApp`` instances as that node can possibly run. In some
    settings, you might want to exclude certain resources from the simulation. You can
    do this by appending ``--num-cpus=<NUM_CPUS_FROM_NODE>`` and/or
    ``--num-gpus=<NUM_GPUS_FROM_NODE>`` in any ``ray start`` command (including when
    starting the head).

FAQ for Simulations
-------------------

.. dropdown:: Can I make my ``ClientApp`` instances stateful?

    Yes. Use the ``state`` attribute of the |context_link|_ object that is passed to the ``ClientApp`` to save variables, parameters, or results to it. Read the `Designing Stateful Clients <how-to-design-stateful-clients.rst>`_ guide for a complete walkthrough.

.. dropdown:: Can I run multiple simulations on the same machine?

    Yes, but bear in mind that each simulation isn't aware of the resource usage of the other. If your simulations make use of GPUs, consider setting the ``CUDA_VISIBLE_DEVICES`` environment variable to make each simulation use a different set of the available GPUs. Export such an environment variable before starting |flwr_run_link|_.

.. dropdown:: Do the CPU/GPU resources set for each ``ClientApp`` restrict how much compute/memory these make use of?

    No. These resources are exclusively used by the simulation backend to control how many workers can be created on startup. Let's say N backend workers are launched, then at most N ``ClientApp`` instances will be running in parallel. It is your responsibility to ensure ``ClientApp`` instances have enough resources to execute their workload (e.g., fine-tune a transformer model).

.. dropdown:: My ``ClientApp`` is triggering OOM on my GPU. What should I do?

    It is likely that your `num_gpus` setting, which controls the number of ``ClientApp`` instances that can share a GPU, is too low (meaning too many ``ClientApps`` share the same GPU). Try the following:

    1. Set your ``num_gpus=1``. This will make a single ``ClientApp`` run on a GPU.
    2. Inspect how much VRAM is being used (use ``nvidia-smi`` for this).
    3. Based on the VRAM you see your single ``ClientApp`` using, calculate how many more would fit within the remaining VRAM. One divided by the total number of ``ClientApps`` is the ``num_gpus`` value you should set.

    Refer to :ref:`clientappresources` for more details.

    If your ``ClientApp`` is using TensorFlow, make sure you are exporting ``TF_FORCE_GPU_ALLOW_GROWTH="1"`` before starting your simulation. For more details, check.

.. dropdown:: How do I know what's the right ``num_cpus`` and ``num_gpus`` for my ``ClientApp``?

    A good practice is to start by running the simulation for a few rounds with higher ``num_cpus`` and ``num_gpus`` than what is really needed (e.g., ``num_cpus=8`` and, if you have a GPU, ``num_gpus=1``). Then monitor your CPU and GPU utilization. For this, you can make use of tools such as ``htop`` and ``nvidia-smi``. If you see overall resource utilization remains low, try lowering ``num_cpus`` and ``num_gpus`` (recall this will make more ``ClientApp`` instances run in parallel) until you see a satisfactory system resource utilization.

    Note that if the workload on your ``ClientApp`` instances is not homogeneous (i.e., some come with a larger compute or memory footprint), you'd probably want to focus on those when coming up with a good value for ``num_gpus`` and ``num_cpus``.

.. dropdown:: Can I assign different resources to each ``ClientApp`` instance?

    No. All ``ClientApp`` objects are assumed to make use of the same ``num_cpus`` and ``num_gpus``. When setting these values (refer to :ref:`clientappresources` for more details), ensure the ``ClientApp`` with the largest memory footprint (either RAM or VRAM) can run in your system with others like it in parallel.

.. dropdown:: Can I run single simulation accross multiple compute nodes (e.g. GPU servers)?

    Yes. If you are using the ``RayBackend`` (the *default* backend) you can first interconnect your nodes through Ray's cli and then launch the simulation. Refer to :ref:`multinodesimulations` for a step-by-step guide.

.. dropdown:: My ``ServerApp`` also needs to make use of the GPU (e.g., to do evaluation of the *global model* after aggregation). Is this GPU usage taken into account by the ``Simulation Engine``?

    No. The ``Simulation Engine`` only manages ``ClientApps`` and therefore is only aware of the system resources they require. If your ``ServerApp`` makes use of substantial compute or memory resources, factor that into account when setting ``num_cpus`` and ``num_gpus``.

.. dropdown:: Can I indicate on what resource a specific instance of a ``ClientApp`` should run? Can I do resource placement?

    Currently, the placement of ``ClientApp`` instances is managed by the ``RayBackend`` (the only backend available as of ``flwr==1.13.0``) and cannot be customized. Implementing a *custom* backend would be a way of achieving resource placement.
