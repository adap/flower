Run Flower with the Deployment Engine
=====================================

This how-to guide demonstrates how to set up and run Flower with the Deployment Engine
using minimal configurations to illustrate the workflow.

In this how-to guide, you will:

- Set up a Flower project from scratch using the PyTorch template.
- Run a Flower using the Deployment Engine.
- Add a new federation configuration in the ``pyproject.toml`` file.
- Start and monitor a run using the flwr CLI.

The how-to guide should take less than 10 minutes to complete.

Prerequisites
-------------

Before you start, make sure that:

- The latest ``flwr`` CLI version is installed on your machine. Follow the installation
  instructions :doc:`here <../how-to-install-flower>`.

Step 1: Set Up
--------------

Create a new Flower project (PyTorch), and follow the instructions show upon executing :code:`flwr new`:

.. code-block:: bash

    $ flwr new my-project --framework PyTorch --username flower

    🔨 Creating Flower project my-project...
    🎊 Project creation successful.

    Use the following command to run your project:

          cd my-project
          pip install -e .
          flwr run

    $ cd my-project
    $ pip install -e .

.. note::

    If you decide to run the project with ``flwr run .``, the Simulation Engine will be
    used. Continue to Step 2 to know how to instead use the Deployment Engine.
Step 2: Start the SuperLink
---------------------------

In a new terminal, start the SuperLink process in insecure mode:

.. code-block:: bash

    $ flower-superlink --insecure

.. dropdown:: Understand the command

    * ``flower-superlink``: Name of the SuperLink binary.
    * | ``--insecure``: This flag tells the SuperLink to operate in an insecure mode, allowing
      | unencrypted communication.

Step 3: Start the SuperNodes
----------------------------

You will need two terminals for this step.

1. **Terminal 1** Start the first SuperNode:

   .. code-block:: bash

       $ flower-supernode \
            --insecure \
            --superlink 127.0.0.1:9092 \
            --node-config "partition-id=0 num-partitions=2" \
            --supernode-address 127.0.0.1:9094 \
            --isolation subprocess

   .. dropdown:: Understand the command

       * ``flower-supernode``: Name of the SuperNode binary.
       * | ``--insecure``: This flag tells the SuperNode to operate in an insecure mode, allowing
         | unencrypted communication.
       * | ``--superlink 127.0.0.1:9092``: Connect to the SuperLink's Fleet API at the address
         | ``127.0.0.1:9092``.
       * | ``--node-config "partition-id=0 num-partitions=2"``: Set the partition ID to ``0`` and the
         | number of partitions to ``2`` for the SuperNode configuration.
       * | ``--supernode-address 127.0.0.1:9094``: Set the address and port number where the
         | SuperNode is listening to communicate with the ClientApp.
       * | ``--isolation subprocess``: Tells the SuperNode to run the ClientApp in a subprocess.

2. **Terminal 2** Start the second SuperNode:

   .. code-block:: shell

       $ flower-supernode \
            --insecure \
            --superlink 127.0.0.1:9092 \
            --node-config "partition-id=1 num-partitions=2" \
            --supernode-address 127.0.0.1:9095 \
            --isolation subprocess

Step 4: Start the SuperExec
---------------------------

In a new terminal, start the SuperExec process with the following command:

.. code-block:: bash

    $ flower-superexec \
        --insecure \
        --executor flwr.superexec.deployment:executor \
        --executor-config 'superlink="127.0.0.1:9091"'

.. dropdown:: Understand the command

    * ``flower-superexec``: Name of the SuperExec binary.
    * | ``--insecure``: This flag tells the SuperExec to operate in an insecure mode, allowing
      | unencrypted communication.
    * | ``--executor flwr.superexec.deployment:executor`` Use the
      | ``flwr.superexec.deployment:executor`` executor to run the ServerApps.
    * | ``--executor-config 'superlink="127.0.0.1:9091"'``: Configure the SuperExec executor to
      | connect to the SuperLink running on port ``9091``.

Step 5: Run the Project
-----------------------

1. Add a new federation configuration in the ``pyproject.toml``:

   .. code-block:: toml
       :caption: pyproject.toml

       [tool.flwr.federations.local-deployment]
       address = "127.0.0.1:9093"
       insecure = true

   .. note::

       You can customize the string that follows ``tool.flwr.federations.`` to fit your
       needs. However, please note that the string cannot contain a dot (``.``).

       In this example, ``local-deployment`` has been used. Just remember to replace
       ``local-deployment`` with your chosen name in both the ``tool.flwr.federations.``
       string and the corresponding ``flwr run .`` command.

2. In another terminal, run the Flower project and follow the ServerApp logs to track
   the execution of the run:

   .. code-block:: bash

       $ flwr run . local-deployment --stream

   If you want to rerun the project or test an updated version by making changes to the
   code, simply re-run the command above.

Step 6: Clean Up
----------------

To stop all Flower service, use the ``Ctrl+C`` command in each terminal to stop the
respective processes.

Where to Go Next
----------------

- :doc:`docker/tutorial-quickstart-docker`
- :doc:`docker/tutorial-quickstart-docker-compose`
