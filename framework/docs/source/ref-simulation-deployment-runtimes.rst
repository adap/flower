:og:description: Flower simulation versus deployment runtime
.. meta::
    :description: Flower simulation versus deployment runtime

Simulation vs. Deployment Runtime
=================================

Flower Apps can be executed with both with `simulation <how-to-run-simulations.html>`_
or `deployment <how-to-run-flower-with-deployment-engine.html>`_ runtimes. Switching
between runtimes simply requires specifying a different type of `federation` when
executing the `flwr run` command via the `Flower CLI <ref-api-cli.html>`_.

While Flower Apps can run in both simulation and deployment without making changes to
their code, there are some differences on how they get executed depending on the
runtime. The following table outlines the key characteristics that differentiate
simulated Flower federations from deployed ones.

.. list-table::
    :widths: 15 25 25
    :header-rows: 1

    - - Dimension
      - Simulation Runtime
      - Deployment Runtime
    - - **Lifecycle Stage**
      - Ideal for rapid prototyping, algorithm validation, research, debugging, and
        experimentation.
      - Deploy validated use cases in production, real-world privacy-preserving
        applications.
    - - **Environment**
      - Local or remote, single-node or multi-node, controlled.
      - Distributed, remote.
    - - **Data**
      - Simulated data partitions, public or private datasets or artificially generated
        - a natural fit for `Flower Datasets <https://flower.ai/docs/datasets/>`_.
      - Real client-side data, residing on local databases or filesystems.
    - - **Backend**
      - Multiple Python processes/workers coordinated using `Ray
        <https://docs.ray.io/>`_.
      - Multiple independent processes or subprocesses running in coordination with the
        SuperLink and SuperNodes.
    - - **Execution Mode**
      - Multiprocessing execution where each process simulates a distinct client.
      - Parallel execution mode across a network of physical machines/devices or
        computing environment.
    - - **Communication**
      - Managed via in-memory communication.
      - Managed via Flower Deployment Runtime through TLS-enabled gRPC.
    - - **Server-side Infrastructure**
      - Simulation runtime coordinates the spawning of workers (python process) which
        play the role of `simulated` SuperNodes. There is no SuperLink.
      - The SuperLink awaits for SuperNodes to connect. User interface with the
        SuperLink using the `Flower CLI <ref-api-cli.html>`_.
    - - **Server-side App execution**
      - A ``ServerApp`` process is initialized inside a controlled environment and
        communicates in-memory with workers.
      - ``ServerApp`` `process or subprocess <ref-flower-network-communication.html>`_
        runs independently from the SuperLink and communicates with it over gRPC via the
        ServerAppIO API.
    - - **Client-side Infrastructure**
      - None. The Simulation runtime is standalone.
      - SuperNodes connect to the SuperLink via TLS-enabled gRPC using the Fleet API.
        Node authentication can be enabled.
    - - **Client-side App execution**
      - Each process executes a ``ClientApp`` on demand. They might execute multiple
        instances of the same ``ClientApp`` to simulate large amounts of clients.
        ``ClientApps`` are stateless.
      - Initialized as a ``ClientApp`` `process or subprocess
        <ref-flower-network-communication.html>`_ runs independently from the SuperNode
        and communicates with it over gRPC via the ClientAppIo API. ``ClientApps`` are
        stateless.
