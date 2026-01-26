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

The Flower Configuration is a TOML file that lives in the ``FLWR_HOME`` directory (which
defaults to ``$HOME/.flwr``) and is designed to simplify the usage of `Flower CLI
<ref-api-cli.html>`_ commands.

The Flower Configuration enables the system-wide useage of most Flower CLI commands
without the need of an app (all but ``flwr run``). This is particular useful when
running Flower CLI commands from scripts or when running Flower CLI commands from
different directories.

.. tip::

    If a Flower Configuration files isn't found in your system, it will be automatically
    created for you the first time you run a Flower CLI command. If you are upgrading
    from a previous version of Flower, the ``federations`` section in your
    ``pyproject.toml`` file will be automatically migrated to the Flower Configuration
    file. The syntax remains the same with the exception of the name of the section and
    the ``superlink`` keyword instead of ``tool.flwr.federations``.

*************************************
 Flower Configuration File Structure
*************************************

The Flower Configuration file replaces the ``federations`` section in the
``pyproject.toml`` file of Flower Apps. Crucially, the Flower Configuration drops the
term "federation" and instead uses the term "superlink" to refer to the connection
between the Flower CLI and the ``SuperLink`` that handles the request from the CLI. At a
high level, the content and structure of the configuration file mirrors that of the
``federations`` section in the ``pyproject.toml`` file of Flower Apps. Let's see an
example:

.. code-block:: toml

    [superlink]
    default = "local"

    [superlink.local]
    options.num-supernodes = 10

    [superlink.local-poc]
    address = "127.0.0.1:9093"
    insecure = true

The name of all connections must be unique and use as prefix the term ``superlink.``.
Depending on the type of connection (i.e. simulation or deployment), you may specify
different options. For example, ``superlink.local`` uses ``options.num-supernodes`` to
specify the number of ``SuperNodes`` to use in a Simulation, while
``superlink.local-poc`` uses ``address`` and ``insecure`` to specify the address and
security of the connection to a locally running ``SuperLink``. You can specify as many
connections as you need.

The keyword ``default`` is used to specify the default connection to use when running
``Flower CLI`` commands without explicitly specifying a connection. In the example above
``superlink.local`` is the default connection, so running ``flwr run`` without any
arguments will use the ``superlink.local`` connection.

**************************
 Listing your connections
**************************

You can list all your connections using the ``flwr config ls`` command. Which assuming
the default configuration file will shown earlier, the expected output will be:

.. code-block:: shell

    $ flwr config ls

    Flower Config file: /path/to/your/.flwr/config.toml
    SuperLink connections:
      local (default)
      local-poc

**************************
 Local Simulation Example
**************************

To define a local simulation connection, you can use the following structures for most
common use cases:

.. code-block:: toml

    [superlink.local]
    options.num-supernodes = 10

    [superlink.local-custom-resources]
    options.num-supernodes = 100
    options.backend.client-resources.num-cpus = 1
    options.backend.client-resources.num-gpus = 0.1

The example above define two local simulation connections. ``superlink.local`` is a
local simulation with 10 virtual SuperNodes using ``options.num-supernodes = 10``.
``superlink.local-custom-resources`` is a local simulation with 100 virtual SuperNodes
using ``options.num-supernodes = 100`` and each ``SuperNode`` is allocated 1 CPU and 10%
of a GPU.

Learn more in the `How to Run Simulations
<https://flower.ai/docs/framework/how-to-run-simulations.html>`_ guide about other
optional parameters you can use to configure your local simulation.

***************************
 Remote Deployment Example
***************************

You can also configure connections for remote deployment. For example:

.. code-block:: toml

    [superlink.remote-deployment]
    address = "<SUPERLINK-ADDRESS>:<PORT>"
    root-certificate = "path/to/root/cert.pem"  # Optional, for TLS
    # insecure = true  # Disable TLS (not recommended for production)

.. dropdown:: Understanding each field

    .. note::

        \* Required fields

    - ``address``\*: The address of the SuperLink Control API to connect to.
    - ``root-certificate``: Path to the root certificate file for TLS. Ignored if ``insecure`` is ``true``. If omitted, Flower uses the default gRPC root certificate.
    - ``insecure``: Set to ``true`` to disable TLS (not recommended for production). Defaults to ``false``, if omitted.

Refer to the `deployment documentation <https://flower.ai/docs/framework/deploy.html>`_
for TLS setup and advanced configurations.

********************************************
 Upgrading from previous versions of Flower
********************************************

If you are upgrading from a previous version of Flower, the ``federations`` section in
your ``pyproject.toml`` file will be automatically migrated to the Flower Configuration
file. The syntax remains the same with the exception of the name of the section and the
``superlink`` keyword instead of ``tool.flwr.federations``.

During the migration process, the ``pyproject.toml`` file will be modified in place and
the ``federations`` section will be commented out for your reference. Once migrated, it
is safe to remove the ``federations`` section from the ``pyproject.toml`` file entirely.
