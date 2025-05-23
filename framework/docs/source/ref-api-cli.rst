Flower CLI reference
====================

Basic Commands
--------------

.. _flwr-apiref:

``flwr`` CLI
~~~~~~~~~~~~

.. click:: flwr.cli.app:typer_click_object
    :prog: flwr
    :nested: full

.. _flower-superlink-apiref:

``flower-superlink``
~~~~~~~~~~~~~~~~~~~~

.. argparse::
    :module: flwr.server.app
    :func: _parse_args_run_superlink
    :prog: flower-superlink

.. _flower-supernode-apiref:

``flower-supernode``
~~~~~~~~~~~~~~~~~~~~

.. argparse::
    :module: flwr.supernode.cli.flower_supernode
    :func: _parse_args_run_supernode
    :prog: flower-supernode

Advanced Commands
-----------------

.. _flwr-serverapp-apiref:

``flwr-serverapp``
~~~~~~~~~~~~~~~~~~

.. argparse::
    :module: flwr.server.serverapp.app
    :func: _parse_args_run_flwr_serverapp
    :prog: flwr-serverapp

.. _flwr-clientapp-apiref:

``flwr-clientapp``
~~~~~~~~~~~~~~~~~~

.. argparse::
    :module: flwr.client.clientapp.app
    :func: _parse_args_run_flwr_clientapp
    :prog: flwr-clientapp

Technical Commands
------------------

.. _flower-simulation-apiref:

``flower-simulation``
~~~~~~~~~~~~~~~~~~~~~

.. argparse::
    :module: flwr.simulation.run_simulation
    :func: _parse_args_run_simulation
    :prog: flower-simulation
