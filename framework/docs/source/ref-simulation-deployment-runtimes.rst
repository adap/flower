:og:description: Flower simulation versus deployment runtime
.. meta::
    :description: Flower simulation versus deployment runtime

Simulation vs. Deployment Runtime
=================================

From both a developer and user experience perspective, the only change required when
moving from a simulated to a real-world Flower federation is to set the correct
federation address pointing to a SuperLink deployed either in a `simulation
<how-to-run-simulations.html>`_ or `deployment
<how-to-run-flower-with-deployment-engine.html>`_ runtime. In this way, the same
application developed using the Flower simulation runtime can be directly deployed to a
`real-world Flower federation <explanation-flower-architecture.html>`_. Additionally,
the `Flower CLI <ref-api-cli.html>`_ remains the same across both environments, ensuring
a smooth transition without the need for additional configuration or tooling changes.
The following table outlines the key characteristics that differentiate simulated Flower
federations from deployed ones.

.. list-table::
    :widths: 15 25 25
    :header-rows: 1

    - - Dimension
      - Simulation Runtime
      - Deployment Runtime
    - - **Lifecycle Stage**
      - Ideal for rapid prototyping, algorithm validation, research, debugging, and
        experimentation
      - Deploy validated use cases in production, real-world privacy-preserving
        applications
    - - **Environment**
      - Local, standalone, controlled
      - Distributed, remote
    - - **Data**
      - Simulated, e.g., academic sources or artificially generated - a natural fit for
        `Flower Datasets <https://flower.ai/docs/datasets/>`_
      - Real client-side data, residing on local databases or filesystems
    - - **Backend**
      - Multiple Python processes/workers coordinated using `Ray
        <https://docs.ray.io/>`_
      - Multiple independent processes or subprocesses deployed through SuperLink and
        SuperNode components
    - - **Execution Mode**
      - Parallelized or concurrent worker execution mode depending on the available
        resources of the simulated environment
      - Parallel execution mode across a network of physical machines/devices or
        computing environment
    - - **Communication**
      - Managed via in-memory communication
      - Managed via Flower Deployment Runtime through TLS-enabled gRPC
    - - **Server-side**
      - A ``ServerApp`` process initialized inside a controlled environment,
        coordinating multiple simulated clients
      - ``ServerApp`` runs independently on a machine through the SuperLink and the
        communication takes place over gRPC via the ServerAppIO API and users interface
        with the SuperLink via the Exec API
    - - **Client-side**
      - Simulated ``ClientApp`` processes initialized inside a controlled environment
      - Initialized as a ``ClientApp`` `process or subprocess
        <ref-flower-network-communication.html>`_ connected to independent SuperNode
        instances via the ClientAppIo API and SuperNodes communicate with the SuperLink
        over gRPC via the Fleet API
