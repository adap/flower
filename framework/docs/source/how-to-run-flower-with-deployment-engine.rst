:og:description: Guide to use Flower's Deployment Engine and run a Flower App trough a federation consisting of a SuperLink and two SuperNodes.
.. meta::
    :description: Guide to use Flower's Deployment Engine and run a Flower App trough a federation consisting of a SuperLink and two SuperNodes.

Run Flower with the Deployment Engine
=====================================

This how-to guide demonstrates how to set up and run Flower with the Deployment Engine
using minimal configurations to illustrate the workflow. This is a complementary guide
to the :doc:`docker/index` guides.

In this how-to guide, you will:

- Create a Flower App using the PyTorch template.
- Start a Flower federation consisting of one ``SuperLink`` ("the server") and two
  ``SuperNodes`` ("the clients").
- Run the Flower App on this federation.

The how-to guide should take less than 10 minutes to complete.

Prerequisites
-------------

Before you start, make sure that:

- The latest ``flwr`` CLI version is installed on your machine. Follow the installation
  instructions :doc:`here <../how-to-install-flower>`.
- This guide assumes all commands to be executed in the same machine in different
  terminals, each making use of the same Python environment.
- This guide also assumes that you are familiar with the basic components in a Flower
  deployment (i.e. ``SuperLink`` and ``SuperNode``), what their roles are and how they
  interact with each other. Please refer to the :doc:`explanation-flower-architecture`
  guide and the :doc:`ref-flower-network-communication` for an overview of what each
  component does and how they interact with each other.

.. note::

    In a real deployment you would typically run the ``SuperLink`` the ``SuperNodes`` in
    different machines/servers from the one you develop your Flower app (i.e. from where
    you do ``flwr new`` and ``flwr run``). The guide presented below is still valid for
    such scenarios but you will need to have setup a Python environment with the right
    set of dependencies for ``SuperLink`` and ``SuperNodes``. An often easier way to
    achieve such deployments is by means of Docker. Check the :doc:`docker/index` to
    gain a better understanding on how to do so.

Step 1: Create a Flower App
---------------------------

Although you could write a Flower app from scratch, it is often easier to start from one
of the templates available via ``flwr new`` and then customize it to your use case.
Create a new Flower app (PyTorch), and follow the instructions show upon executing
``flwr new``:

.. code-block:: bash

    $ flwr new my-project --framework PyTorch --username flower

    🔨 Creating Flower App my-project...
    🎊 Flower App creation successful.

    To run your Flower App, use the following command:

            flwr run my-project

    If you haven not installed all dependencies yet, follow these steps:

            cd my-project
            pip install -e .
            flwr run .

.. note::

    You might want to update the ``torch`` and ``torchvision`` packages that come with
    the proejct to the latets released versions. Do so with: ``pip install -U torch
    torchvision``.

    If you decide to run the project with ``flwr run .``, the Simulation Engine will be
    used. Continue to Step 2 to know how to instead use the Deployment Engine.

.. tip::

    Feel free to inspect the code using your favorite code editor before proceeding.
    Just open the ``my-project`` that was automatically created via ``flwr new``. If you
    would like to get an overview of the code that was generated, take a look at the
    :doc:`tutorial-quickstart-pytorch` tutorial.

Step 2: Launch Flower Federation
--------------------------------

In this section you will learn how to launch a ``SuperLink`` and connect two
``SuperNodes`` to it.

Start a Flower SuperLink
~~~~~~~~~~~~~~~~~~~~~~~~

In a new terminal, activate your environment and start the ``SuperLink`` process in
insecure mode:

.. code-block:: bash

    $ flower-superlink --insecure

.. dropdown:: Understand the command

    * ``flower-superlink``: Name of the ``SuperLink`` binary.
    * | ``--insecure``: This flag tells the SuperLink to operate in an insecure mode, allowing
      | unencrypted communication. Refer to the :doc:`how-to-enable-tls-connections` guide to learn how to run your ``SuperLink`` with TLS.

Start two Flower SuperNodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this step, you will launch two ``SuperNodes`` and connect them to the ``SuperLink``.
You will need two terminals for this step.

.. note::

    Note that the values passed via the ``--node-config`` argument are specific to the
    behaviour of the ``ClientApp``. If you inspect the code generated in the first step
    via ``flwr new``, you'd see that the ``ClientApp`` is expecting a certain set of
    key-value pairs to be present in order to partition and load some data. Typically,
    your ``ClientApp`` wouldn't partition a dataset, instead it would access the data
    directly available. In such cases you would write your ``ClientApp`` and make it
    receive, for example, the path to a directory of images.

1. **Terminal 1** Start the first ``SuperNode`` after activating your environment:

   .. code-block:: bash

       $ flower-supernode \
            --insecure \
            --superlink 127.0.0.1:9092 \
            --clientappio-api-address 127.0.0.1:9094 \
            --node-config "partition-id=0 num-partitions=2"

   .. dropdown:: Understand the command

       * ``flower-supernode``: Name of the ``SuperNode`` binary.
       * | ``--insecure``: This flag tells the ``SuperNode`` to operate in an insecure mode, allowing
         | unencrypted communication. Refer to the :doc:`how-to-enable-tls-connections` guide to learn how to run your ``SuperNode`` with TLS.
       * | ``--superlink 127.0.0.1:9092``: Connect to the SuperLink's Fleet API at the address
         | ``127.0.0.1:9092``. If you had launched the ``SuperLink`` in a different machine, you'd replace ``127.0.0.1`` with the public IP of that machine.
       * | ``--clientappio-api-address 127.0.0.1:9094``: Set the address and port number where the
         | SuperNode is listening to communicate with the ``ClientApp``.
       * | ``--node-config "partition-id=0 num-partitions=2"``: The ``ClientApp`` code generated via the ``flwr new`` template expects those two key-value pairs to be defined at run time. Set the partition ID to ``0`` and the number of partitions to ``2`` for the ``SuperNode`` configuration.

2. **Terminal 2** Start the second ``SuperNode`` after activating your environment:

   .. code-block:: shell

       $ flower-supernode \
            --insecure \
            --superlink 127.0.0.1:9092 \
            --clientappio-api-address 127.0.0.1:9095 \
            --node-config "partition-id=1 num-partitions=2"

   .. dropdown:: Understand the command

       * ``--clientappio-api-address 127.0.0.1:9095``: Note that a different port is being used. This is only needed because you are running two ``SuperNodes`` on the same machine. Typically you would run one node per machine and therefore, the ``--clientappio-api-address`` could be omitted all together and left with its default value.
       * ``--node-config "partition-id=1 num-partitions=2"```: Note here we indicate a different `partition-id`. In this way, a ``ClientApp`` will use a different data partition depending on which ``SuperNode`` runs in.

Step 3: Run a Flower App on the Federation
------------------------------------------

At this point, you have launched two ``SuperNodes`` that are connected to the same
``SuperLink``. The system is idling waiting for a ``Run`` to be submitted. Before you
can run your Flower App through the federation we need a way to tell ``flwr run`` that
the App is to be executed via the SuperLink we just started, instead of using the local
Simulation Engine (the default). Doing this is easy: define a new federation section in
the ``pyproject.toml``, indicate the address of the ``SuperLink`` and pass a certificate
(if any) or set the insecure flag (only when testing locally, real deployments require
TLS).

1. Open the ``pyproject.toml`` file and at the end add a new federation configuration:

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

2. In another terminal and with your Python environment activated, run the Flower App
   and follow the ``ServerApp`` logs to track the execution of the run:

   .. code-block:: bash

       $ flwr run . local-deployment --stream

   If you want to rerun the project or test an updated version by making changes to the
   code, simply re-run the command above.

Step 4: Clean Up
----------------

Use the ``Ctrl+C`` command in each terminal to stop the respective processes.
