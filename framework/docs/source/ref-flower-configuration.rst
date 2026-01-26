:og:description: Learn how to setup the Flower Configuration file and how to use it to run your Flower app and interact with SuperLinks.
.. meta::
    :description: Learn how to setup the Flower Configuration file and how to use it to run your Flower app and interact with SuperLinks.

###########################
 Flower Configuration File
###########################

.. note::

    The Flower Configuration file is a new feature introduced in Flower 1.26.0. While
    its functionality will evolve over time, as a start it serves as a direct
    replacement of the `federations` section in the `pyproject.toml` file of Flower
    Apps.

**What is it?**

The Flower Configuration is a TOML file that lives in the ``FLWR_HOME`` directory (which
defaults to ``$HOME/.flwr``) and is designed to simplify the usage of `Flower CLI
<ref-api-cli.html>`_ commands.

**Why use it?**

This configuration file allows you to define reusable connection configurations (for
example, "local-simulation", "staging-server", "production-server") that you can
reference by name when running ``flwr`` commands. Instead of typing the same connection
details repeatedly, you define them once and reuse them. Additionally, commands like
``flwr log``, ``flwr ls``, and ``flwr stop`` can be run from anywhere on your system
without needing to be inside a Flower app directory. Note: ``flwr run`` still requires
being in an app directory since it needs to access your app code.

.. tip::

    **First-time setup:** If a Flower Configuration file doesn't exist in your system,
    it will be automatically created for you the first time you run a Flower CLI
    command.

    **Upgrading from older versions:** If you are upgrading from a previous version of
    Flower, the ``federations`` section in your ``pyproject.toml`` file will be
    automatically migrated to the Flower Configuration file. The syntax remains the same
    with the exception of the name of the section and the ``superlink`` keyword instead
    of ``tool.flwr.federations``.

*************************************
 Flower Configuration File Structure
*************************************

**Understanding the terminology**

The Flower Configuration file uses the term **"superlink"** to refer to connection
configurations. Each connection configuration describes how to connect to a Flower
**SuperLink** (the central server component that coordinates federated learning). You
can define multiple connection configurations for different scenarios:

- ``superlink.local`` - for running local simulations on your machine
- ``superlink.staging`` - for connecting to a staging server
- ``superlink.production`` - for connecting to a production deployment

The configuration structure is similar to the older ``federations`` section in
``pyproject.toml``, but now lives in a central location and uses clearer naming.

**Basic example**

.. code-block:: toml

    [superlink]
    default = "local"

    [superlink.local]
    options.num-supernodes = 10

    [superlink.local-poc]
    address = "127.0.0.1:9093"
    insecure = true

**Explanation:**

- ``[superlink]`` section defines which connection configuration to use by default
- ``default = "local"`` means the ``superlink.local`` configuration will be used when
  you don't specify a connection explicitly
- ``[superlink.local]`` defines a local simulation configuration with 10 virtual
  SuperNodes
- ``[superlink.local-poc]`` defines a configuration for connecting to a locally running
  SuperLink server at address ``127.0.0.1:9093``

Connection configuration names must be unique and use the ``superlink.`` prefix. The
type of options you specify depends on whether you're configuring a simulation
(``options.num-supernodes``) or a deployment (``address``, ``insecure``).

**************************
 Listing your connections
**************************

You can list all your connection configurations using the ``flwr config ls`` command.
Assuming the default configuration file shown earlier, the expected output will be:

.. code-block:: shell

    $ flwr config ls

    Flower Config file: /path/to/your/.flwr/config.toml
    SuperLink connections:
      local (default)
      local-poc

This shows you have two connection configurations available, with ``local`` set as the
default.

**************************
 Local Simulation Example
**************************

Local simulations allow you to test your federated learning app on your own machine
using virtual SuperNodes instead of real distributed nodes. This is useful for
development and testing before deploying to real distributed environments.

**Basic simulation configuration:**

.. code-block:: toml

    [superlink.local]
    options.num-supernodes = 10

This creates a simulation connection configuration with 10 virtual SuperNodes.

**Simulation with custom resources:**

.. code-block:: toml

    [superlink.local-custom-resources]
    options.num-supernodes = 100
    options.backend.client-resources.num-cpus = 1
    options.backend.client-resources.num-gpus = 0.1

This creates a simulation connection configuration with 100 virtual SuperNodes, where
each is allocated 1 CPU and 10% of a GPU. This is useful when you want to control
resource distribution or simulate resource-constrained environments.

**When to use each:**

- Use the basic configuration for quick testing with default resource allocation
- Use custom resources when you need to simulate specific hardware constraints or want
  to control how many virtual SuperNodes can run in parallel based on your machine's
  resources

Learn more in the `How to Run Simulations
<https://flower.ai/docs/framework/how-to-run-simulations.html>`_ guide about other
optional parameters you can use to configure your local simulation.

***************************
 Remote Deployment Example
***************************

When you're ready to deploy your federated learning app to real distributed nodes, you
configure connections that point to a remote SuperLink.

**Example configuration:**

.. code-block:: toml

    [superlink.remote-deployment]
    address = "<SUPERLINK-ADDRESS>:<PORT>"
    root-certificate = "path/to/root/cert.pem"  # Optional, for TLS
    # insecure = true  # Disable TLS (not recommended for production)

.. dropdown:: Understanding each field

    .. note::

        \* Required fields

    - ``address``\*: The address of the SuperLink Control API to connect to (e.g., ``my-server.example.com:9093``).
    - ``root-certificate``: Path to the root certificate file for TLS encryption. This secures the communication between your CLI and the SuperLink. If omitted, Flower uses the default gRPC root certificate. This field is ignored if ``insecure`` is set to ``true``.
    - ``insecure``: Set to ``true`` to disable TLS encryption (only use this for local testing, never in production). Defaults to ``false`` if omitted, meaning TLS is enabled by default.

**TLS (Transport Layer Security) explained:**

TLS encrypts the communication between your local machine and the remote SuperLink to
prevent eavesdropping and tampering. In production, you should always use TLS by either:

- Providing a ``root-certificate`` file (recommended for custom certificates)
- Omitting both ``root-certificate`` and ``insecure`` to use default certificates

Only set ``insecure = true`` for local testing environments.

Refer to the `deployment documentation <https://flower.ai/docs/framework/deploy.html>`_
for TLS setup and advanced configurations.

********************************************
 Upgrading from previous versions of Flower
********************************************

**If you're new to Flower 1.26.0+, you can skip this section.**

For users upgrading from versions before 1.26.0: The ``federations`` section in your
``pyproject.toml`` file will be automatically migrated to the new Flower Configuration
file the first time you run a ``flwr`` command.

**What happens during migration:**

1. A new config file is created at ``$HOME/.flwr/config.toml``
2. Your federation definitions are copied and renamed (``federations`` → ``superlink``)
3. The old ``[tool.flwr.federations]`` section in ``pyproject.toml`` is commented out
   for your reference

**After migration:**

You can safely delete the commented-out ``federations`` section from your
``pyproject.toml`` file. All connection configurations now live in the central
configuration file and work across all your Flower projects.
