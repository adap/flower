:og:description: Learn how to configure your Flower app using the pyproject.toml file, including dependencies, components, runtime settings, and federation setup.
.. meta::
    :description: Learn how to configure your Flower app using the pyproject.toml file, including dependencies, components, runtime settings, and federation setup.

Configure the ``pyproject.toml`` file
=====================================

When you create a new Flower App using ``flwr new``, a ``pyproject.toml`` file is
generated. This file defines your app's dependencies, configuration, and federation
setup.

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
    publisher = "pan"

- ``name``\*: The name of your Flower app.
- ``version``\*: The current version of your app, used for packaging and distribution.
- ``description``: A short summary of what your app does.
- ``license``: The license your app is distributed under (e.g., Apache-2.0).
- ``dependencies``\*: A list of Python packages required to run your app.
- ``publisher``\*: The name of the person or organization publishing the app.

.. note::

    \* Required fields

Add any Python packages your app needs under ``dependencies``. These will be installed
when you run: ``pip install -e .``

App Components
--------------

.. code-block:: toml

    [tool.flwr.app.components]
    serverapp = "your_module.server_app:app"
    clientapp = "your_module.client_app:app"

- ``serverapp``\*: The import path to your ``ServerApp`` object.
- ``clientapp``\*: The import path to your ``ClientApp`` object.

.. note::

    \* Required fields

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
specify any number of key-value pairs in this section.

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

- ``default``\*: The name of the federation to use when running your app with ``flwr
  run`` without explicitly specifying a federation.

.. note::

    \* Required fields

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
    insecure = true
    # root-certificate = "path/to/root/cert.pem"  # Optional, for TLS

- ``address``\*: The address of the SuperLink Exec API to connect to.
- ``insecure``: Set to ``true`` to disable TLS (not recommended for production).
  Defaults to ``false``.
- ``root-certificate``: Path to the root certificate file for TLS. Ignored if
  ``insecure`` is ``true``. If omitted, Flower uses the default gRPC root certificate.

.. note::

    \* Required fields

Refer to the `deployment documentation <https://flower.ai/docs/framework/deploy.html>`_
for TLS setup and advanced configurations.

Running a Federation
~~~~~~~~~~~~~~~~~~~~

To run a specific federation, either:

- Set it as the default in ``pyproject.toml``, or
- Provide the federation name in the command:

.. code-block:: shell

    flwr run <path-to-your-app> <your-federation-name>

You can run ``flwr run --help`` to view all available options.
