Simulation vs. Deployment Runtime
=================================

From both a developer and user experience perspective, the only change required when
moving from a simulated to a real-world Flower federation is updating the federation
address. Applications developed using the Flower simulation engine can be seamlessly
deployed in a real-world federation by simply pointing to the appropriate ``SuperLink``
endpoint. Additionally, the Flower CLI remains the same across both environments,
ensuring a smooth transition without the need for additional configuration or tooling
changes. The following table outlines the key characteristics that differentiate
simulated Flower federations from deployed ones.

.. list-table:: Comparing Simulation to Deployment Runtime
    :widths: 25 25 50
    :header-rows: 1

    - - Dimension
      - Simulation Runtime
      - Deployment Runtime
    - - **Lifecycle Stage**
      - Ideal for quick prototype testing, algorithm validation, research, debugging,
        experimentation
      - Deploy validated use cases in production, real-world privacy-preserving
        applications
    - - **Environment**
      - Local, standalone, controlled
      - Distributed, remote
    - - **Data**
      - Simulated, e.g., academic sources or artificially generated
      - Real client-side data, residing on physical devices
    - - **Backed**
      - Multiple ``Python`` processes/workers coordinated using ``Ray``
      - Multiple independent processes or subprocesses deployed through the
        ``SuperLink`` and ``SuperNode`` services
    - - **Execution Mode**
      - Parallelized or concurrent worker execution mode depending on the available
        resources of the simulated environment
      - Parallel execution mode across a network of physical machines/devices
    - - **Communication**
      - Managed via ``Python`` APIs
      - Managed through Flower Deployment Runtime, ``Docker``, ``Kubernetes``, ``Helm``
        charts
    - - **Server**
      - A ``ServerApp`` process initialized inside a controlled environment,
        coordinating multiple simulated clients
      - Runs independently on a machine through the ``SuperLink`` service, which
        coordinates physically distributed ``SuperNode``\s over a network. The
        communication takes place over ``gRPC`` via the ``ServerAppIO``, ``Exec``,
        ``Fleet``, and ``ClientAppIO API``\s.
    - - **Client(s)**
      - Simulated ``ClientApp`` processes initialized inside a controlled environment
      - Initialized as ``ClientApp`` process or subprocess connected to independent
        ``SuperNode`` instances via the ``ClientAppIo API``
