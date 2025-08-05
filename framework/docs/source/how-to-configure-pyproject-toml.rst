:og:description: Learn how to configure your Flower app using the pyproject.toml file, including dependencies, components, runtime settings, and federation setup.
.. meta::
    :description: Learn how to configure your Flower app using the pyproject.toml file, including dependencies, components, runtime settings, and federation setup.

Configure ``pyproject.toml``
============================

All Flower Apps need a ``pyproject.toml``. When you create a new Flower App using ``flwr
new``, a ``pyproject.toml`` file is generated. This file defines your app's
dependencies, configuration, and federation setup.

A complete ``pyproject.toml`` file, for example, looks like this:

.. dropdown:: Example ``pyproject.toml``

    .. code-block:: toml

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [project]
        name = "flower-app"
        version = "1.0.0"
        description = "A Flower app example"
        license = "Apache-2.0"
        dependencies = [
            "flwr[simulation]>=1.20.0",
            "numpy>=2.0.2",
        ]

        [tool.hatch.build.targets.wheel]
        packages = ["."]

        [tool.flwr.app]
        publisher = "your-name-or-organization"

        [tool.flwr.app.components]
        serverapp = "your_module.server_app:app"
        clientapp = "your_module.client_app:app"

        [tool.flwr.app.config]
        num-server-rounds = 3
        any-name-you-like = "any value supported by TOML"

        [tool.flwr.federations]
        default = "local-simulation"

        [tool.flwr.federations.local-simulation]
        options.num-supernodes = 10

Here are a few key sections to look out for:

App Metadata and Dependencies
-----------------------------

.. code-block:: toml

    [project]
    name = "your-flower-app-name"
    version = "1.0.0"
    description = ""
    license = "Apache-2.0"
    dependencies = [
        "flwr[simulation]>=1.20.0",
        "numpy>=2.0.2",
    ]

    [tool.flwr.app]
    publisher = "your-name-or-organization"

.. dropdown:: Understanding each field

    .. note::

        \* Required fields

        These fields follow the standard ``pyproject.toml`` metadata format, commonly used by tools like ``uv``, ``poetry``, and others. Flower reuses these for configuration and packaging.

    - ``name``\*: The name of your Flower app.
    - ``version``\*: The current version of your app, used for packaging and distribution. Must follow Semantic Versioning (e.g., "1.0.0").
    - ``description``: A short summary of what your app does.
    - ``license``: The license your app is distributed under (e.g., Apache-2.0).
    - ``dependencies``\*: A list of Python packages required to run your app.
    - ``publisher``\*: The name of the person or organization publishing the app.

Specify the metadata, including the app name, version, etc., in these sections. Add any
Python packages your app needs under ``dependencies``. These will be installed when you
run:

.. code-block:: shell

    pip install -e .

App Components
--------------

.. code-block:: toml

    [tool.flwr.app.components]
    serverapp = "your_module.server_app:app"
    clientapp = "your_module.client_app:app"

.. dropdown:: Understanding each field

    .. note::

        \* Required fields

    - ``serverapp``\*: The import path to your ``ServerApp`` object.
    - ``clientapp``\*: The import path to your ``ClientApp`` object.

These entries point to your ``ServerApp`` and ``ClientApp`` definitions, using the
format ``<module>:<object>``. Only update these import paths if you rename your modules
or the variables that reference your ``ServerApp`` or ``ClientApp``.

App Configuration
-----------------

.. code-block:: toml

    [tool.flwr.app.config]
    num-server-rounds = 3
    any-name-you-like = "any value supported by TOML"

Define configuration values that should be available to your app at runtime. You can
specify any number of key-value pairs in this section. All the configuration values in
this section are optional.

Access these values in your code using ``context.run_config``. For example:

.. code-block:: python

    server_rounds = context.run_config["num-server-rounds"]

Federation Configuration
------------------------

.. code-block:: toml

    [tool.flwr.federations]
    default = "your-federation-name"

    [tool.flwr.federations.your-federation-name]
    ...  # Federation-specific options

.. dropdown:: Understanding each field

    .. note::

        \* Required fields

    - ``default``\*: The name of the federation to use when running your app with ``flwr run`` without explicitly specifying a federation.

Federations allow you to define how your app will run in different environments. You can
configure multiple federations, such as local simulations or remote deployments, within
the ``[tool.flwr.federations]`` section.

Local Simulation Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

    [tool.flwr.federations.local-simulation]
    options.num-supernodes = 10

When using ``flwr new``, a federation named ``"local-simulation"`` is included and set
as the default. The example above sets up a local simulation federation with 10 virtual
SuperNodes using ``options.num-supernodes = 10``.

Learn more in the `How to Run Simulations
<https://flower.ai/docs/framework/how-to-run-simulations.html>`_ guide.

Remote Deployment Example
~~~~~~~~~~~~~~~~~~~~~~~~~

You can also configure federations for remote deployment. For example:

.. code-block:: toml

    [tool.flwr.federations.remote-deployment]
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

Running a Federation
~~~~~~~~~~~~~~~~~~~~

To run a specific federation, use the following command:

.. code-block:: shell

    flwr run <path-to-your-app> <your-federation-name>

Both positional arguments—the app path and the federation name—are optional. If omitted,
the current directory is used as the app path, and the default federation specified in
the ``pyproject.toml`` file is used.

You can run ``flwr run --help`` for more details.
