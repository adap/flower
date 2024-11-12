Flower CLI reference
====================

.. _flwr-apiref:

flwr CLI
--------

.. click:: flwr.cli.app:typer_click_object
    :prog: flwr
    :nested: full

.. _flower-superlink-apiref:

flower-superlink
----------------

.. argparse::
    :module: flwr.server.app
    :func: _parse_args_run_superlink
    :prog: flower-superlink

.. _flower-supernode-apiref:

flower-supernode
----------------

.. argparse::
    :module: flwr.client.supernode.app
    :func: _parse_args_run_supernode
    :prog: flower-supernode

Advanced Commands
=================

.. _flwr-serverapp-apiref:

``flwr-serverapp``
------------------

.. argparse::
    :module: flwr.server.serverapp.app
    :func: _parse_args_run_flwr_serverapp
    :prog: flwr-serverapp

.. _flwr-clientapp-apiref:

``flwr-clientapp``
------------------

.. argparse::
    :module: flwr.client.clientapp.app
    :func: _parse_args_run_flwr_clientapp
    :prog: flwr-clientapp

Technical Commands
==================

.. _flower-simulation-apiref:

flower-simulation
-----------------

.. argparse::
    :module: flwr.simulation.run_simulation
    :func: _parse_args_run_simulation
    :prog: flower-simulation

Deprecated Commands
===================

.. _flower-server-app-apiref:

flower-server-app
-----------------

.. note::

    Note that since version ``1.11.0``, ``flower-server-app`` no longer supports passing
    a reference to a `ServerApp` attribute. Instead, you need to pass the path to Flower
    app via the argument ``--app``. This is the path to a directory containing a
    `pyproject.toml`. You can create a valid Flower app by executing ``flwr new`` and
    following the prompt.

.. argparse::
    :module: flwr.server.run_serverapp
    :func: _parse_args_run_server_app
    :prog: flower-server-app

.. _flower-superexec-apiref:

flower-superexec
----------------

.. warning::

    Note that from version ``1.13.0``, ``flower-superexec`` is removed. Instead, you
    only need to execute |flower_superlink_link|_.

.. argparse::
    :module: flwr.superexec.app
    :prog: flower-superexec

.. |flower_superlink_link| replace:: ``flower-superlink``

.. _flower_superlink_link: ref-api-cli.html#flower-superlink
