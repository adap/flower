##################
 Configure Flower
##################

*************************
 flower-superlink config
*************************

See `flower-superlink CLI ref
<https://flower.ai/docs/framework/ref-api-cli.html#flower-superlink>`_

Purpose: Pass config values to the long-running SuperLink

*************************
 flower-supernode config
*************************

See ``flower-supernode`` CLI ref (not yet public on `flower.ai
<http://flower.ai>`_)

Part of this is ``--node-config``

Purpose: Pass config values to the long-running SuperNode

Note: What about ``--partition-id``? → Use ``flower-supernode
--node-config partition-id=0,num-partitions=2``

**************************
 flower-simulation config
**************************

See `flower-simulation CLI ref
<https://flower.ai/docs/framework/ref-api-cli.html#flower-simulation>`_

Purpose: Pass config values to the long-running Simulation Engine

**************************
 Run config (Flower 1.10)
**************************

Purpose: Pass config values to the app for one particular run (config
value that take effect on the application level, not on the
infrastructure level)

Examples: dynamically change the number of rounds or the learning rate
without hardcoding them in the FAB

Description
===========

In ``pyproject.toml``, users will be able to specify a set of
configuration values that an app can react to:

.. code:: toml

   # ... [other pyproject.toml configs]

   [tool.flwr.config]
   lr = 0.01
   num_rounds = 100

   [tool.flwr.config.clientapp]
   lr = 0.01  # final key: "clientapp.lr"
   num_rounds = 100

These values are accessible to both the ``ServerApp`` and the
``ClientApp`` via ``Context`` . ``ServerApp`` example:

.. code:: python

   app = ServerApp()

   @app.main()
   def main(driver, context):
       lr = context.run_config["lr"]
       print(lr)  # Output: 0.01

Apart from specifying default values in ``pyproject.toml``, users can
override these values in two ways:

#. Override via CLI overrides: ``flwr run --run-config lr=0.02`` (this
   overrides ``lr`` from ``0.01`` to ``0.02``, but keeps ``num_rounds``
   at the default value of ``100``

#. Override via CLI + additional override config file: ``flwr run
   --config overrides.toml``

*******************************
 Executor config (Flower 1.10)
*******************************

RFC: TODO

Purpose: Pass config values to the SuperExec executor plugin when
starting the SuperExec. Executor config provides a way to set static
configuration values that remain constant for the entire liefetime of a
SuperExec (i.e., changing them would require shutting down the SuperExec
and restarting it with different values).

Example: pass the

Description
===========

When starting the SuperExec, we need a way to pass config values to the
executor plugin:

.. code:: bash

   flower-superexec flwr.superexec.deployment:executor \
     --executor-config superlink="localhost:9093" \
     --executor-config verbose=true

We can use the same approach to configure the simulation engine:

.. code:: bash

   flower-superexec flwr.superexec.simulation:executor \
     --executor-config backend-name="not-ray" \
     --executor-config verbose_logging=true

When using 3rd-party executor plugins, the config values (key names and
value types) that need be be passed on to the executor plugin are not
known in advance:

.. code:: bash

   flower-superexec nvflare.flower:executor \
     --executor-config nvflare-workspace-path="~/flare" \
     --executor-config nvflare-simulation-mode=false

**********************************
 Federations config (Flower 1.10)
**********************************

RFC: TODO

Purpose: Enable the user to specify different federations that an app
can run on. In this case, a federation refers to a SuperExec running on
the same machine or a different machine. Federation config provides a
way to tell ``flwr run`` which SuperExec (address) to connect to, which
options to send along, and whether or not to lazily start a SuperExec on
the same machine (to support ``flwr run`` without the need to start a
SuperExec beforehand).

Description
===========

In ``pyproject.toml``, users will be able to configure different
SuperExecs (federations) that ``flwr run`` is able to connect to:

.. code:: toml

   # ... [other pyproject.toml configs]

   # ... [also pyproject.toml run config defaults]
   # [flower.config]
   # lr = 0.01

   ######################################################################
   # Federations config below 👇
   ######################################################################

   [tool.flwr.federations]
   default = "local-simulation"  # Could also be "bloodcounts" / ...

   [tool.flwr.federations.local-simulation]
   options.num-supernodes = 2

   [tool.flwr.federations.bloodcounts]
   address = "1.2.3.4:5678"  # SuperExec address
   root-certificates = "path/to/certs"
   options.num-supernodes = 2

   [tool.flwr.federations.flwrtune-llm-leaderboard]
   address = "flowertune-llm-leaderboard.federations.flower.ai:9093"
   options = { num-gpus = 8 }

   [tool.flwr.federations.nvidia]
   address = "superexec.nvidia.com:9093"
   options = { email = "info@nvidia.com", password = "flower-rocks", force = true }

A minimal version would look like this:

.. code:: toml

   # ... [other pyproject.toml configs]

   # ... [also pyproject.toml run config defaults]
   # [flower.config]
   # lr = 0.01

   ######################################################################
   # Federations config below 👇
   ######################################################################

   [flower.federations]
   default = "local"

   [flower.federations.local]
   start_lazily = true
   address = "localhost:9093"

When using ``flwr run`` to start a run, users can easily switch between
different federations:

.. code:: bash

   # Connect to the SuperExec running on 1.2.3.4:5678
   flwr run . bloodcounts

   # Connect to the SuperExec (w/ simulation engine) hosted by Flower Labs
   flwr run . flwrtune-llm-leaderboard

   # Connect to the SuperExec hosted by Nvidia (running the NVFLARE executor)
   flwr run . nvidia

Here’s what happens when the user executes one of these commands:

#. ``flwr run`` bundles the FAB (as usual, but everything under the
   ``flower.federations`` key would be excluded)

#. ``flwr run`` loads ``flower.federations``
      #. It looks up the ``nvidia`` key
      #. It finds the ``address`` of the SuperExec to connect to is
         ``"superexec.nvidia.com:9093"``
      #. It finds that a custom ``options`` dict has been specified

#. ``flwr run`` connects to ``"superexec.nvidia.com:9093"`` and sends two things:
      #. The FAB
      #. The custom options dict ``{ email = "info@nvidia.com", password
         = "flower-rocks", force = true }``

#. The SuperExec running on ``"superexec.nvidia.com:9093"`` receives the FAB and the ``options`` dict
      #. It handles the FAB in the usual way

      #. It passes the ``options`` dict to the SuperExec executor plugin
         → this enables both first-party (Flower Simulation Engine
         executor plugin) and third-party (NVFLARE executor plugin)
         executors plugins to receive values

It’s important to mention that the ``[flower.federations]`` config is
independent of, for example, ``[flower.config]``. Researchers can copy
``[flower.federations]`` from one project to another, and we might want
to support something like a “global” ``[flower.federations]`` config
file in the ``~/.flwr`` dir. Copying ``[flower.federations]`` from one
project to another would simply allow the user to run the other project
on the same federations using ``flwr run``.
