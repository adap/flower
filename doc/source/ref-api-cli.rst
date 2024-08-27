Flower CLI reference
====================

.. _flwr-apiref:

flwr CLI
~~~~~~~~

.. click:: flwr.cli.app:typer_click_object
   :prog: flwr
   :nested: full

.. _flower-simulation-apiref:

flower-simulation
~~~~~~~~~~~~~~~~~

.. argparse::
   :module: flwr.simulation.run_simulation
   :func: _parse_args_run_simulation
   :prog: flower-simulation

.. _flower-superlink-apiref:

flower-superlink
~~~~~~~~~~~~~~~~

.. argparse::
   :module: flwr.server.app
   :func:  _parse_args_run_superlink
   :prog: flower-superlink

.. _flower-supernode-apiref:

flower-supernode
~~~~~~~~~~~~~~~~~

.. argparse::
   :module: flwr.client.supernode.app
   :func: _parse_args_run_supernode
   :prog: flower-supernode

.. _flower-server-app-apiref:

flower-server-app
~~~~~~~~~~~~~~~~~

.. note::
   Note that since version :code:`1.11.0`, :code:`flower-server-app` no longer supports passing a reference to a `ServerApp` attribute.
   Instead, you need to pass the path to Flower app via the argument :code:`--app`.
   This is the path to a directory containing a `pyproject.toml`.
   You can create a valid Flower app by executing :code:`flwr new` and following the prompt.

.. argparse::
   :module: flwr.server.run_serverapp
   :func: _parse_args_run_server_app
   :prog: flower-server-app

.. _flower-superexec-apiref:

flower-superexec
~~~~~~~~~~~~~~~~~

.. argparse::
   :module: flwr.superexec.app
   :func: _parse_args_run_superexec
   :prog: flower-superexec