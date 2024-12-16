Use JSON outputs for Flower CLIs
================================

The `Flower CLIs <ref-api-cli.html>`_ come with a built-in JSON output mode. This mode
is useful when you want to consume the output of a Flower CLI programmatically. For
example, you might want to use the output of the ``flwr`` CLI in a script or a
continuous integration pipeline.

To enable JSON output, simply pass the ``--format json`` flag to the CLI. In this guide,
we'll show you how to use JSON output with the ``flwr run``, ``ls``, and ``stop``
commands.

.. |flwr_run| replace:: ``flwr run``

.. |flwr_ls| replace:: ``flwr ls``

.. |flwr_stop| replace:: ``flwr stop``

.. _flwr_ls: ref-api-cli.html#flwr-ls

.. _flwr_run: ref-api-cli.html#flwr-run

.. _flwr_stop: ref-api-cli.html#flwr-stop

``flwr run`` JSON output
------------------------

The |flwr_run|_ command runs a Flower app. By default, the command prints the status of
the app build and run process as follows:

.. code-block:: bash

    $ flwr run
    Loading project configuration...
    Success
    🎊 Successfully built flwrlabs.myawesomeapp.1-0-0.014c8eb3.fab
    🎊 Successfully started run 1859953118041441032

To get the output in JSON format, simply pass the ``--format json`` flag:

.. code-block:: bash

    $ flwr run --format json
    {
      "success": true,
      "run-id": 1859953118041441032,
      "fab-id": "flwrlabs/myawesomeapp",
      "fab-name": "myawesomeapp",
      "fab-version": "1.0.0",
      "fab-hash": "014c8eb3",
      "fab-filename": "flwrlabs.myawesomeapp.1-0-0.014c8eb3.fab"
    }

The JSON output contains the following fields:

- ``success``: A boolean indicating whether the command was successful.
- ``run-id``: The ID of the run.
- ``fab-id``: The ID of the Flower app.
- ``fab-name``: The name of the Flower app.
- ``fab-version``: The version of the Flower app.
- ``fab-hash``: The hash of the Flower app.
- ``fab-filename``: The filename of the Flower app.

If the command fails, the JSON output will contain two fields, ``success`` with the
value of ``false`` and ``error-message``. For example, if the command fails to find the
name of the federation on the SuperLink, the output will look like this:

.. _json_error_output:

.. code-block:: bash

    $ flwr run --format json
    {
      "success": false,
      "error-message": "Loading project configuration... \nSuccess\n There is no `[missing]` federation declared in the `pyproject.toml`.\n The following federations were found:\n\nfed-existing-1\nfed-existing-2\n\n"
    }

``flwr ls`` JSON output
-----------------------

The |flwr_ls|_ command lists all the runs in the current project. By default, the
command list the details of one provided run ID or all runs in a Flower federation in a
tabular format:

.. code-block:: bash

    $ flwr ls
    Loading project configuration...
    Success
    📄 Listing all runs...
    ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
    ┃    Run ID    ┃     FAB      ┃    Status    ┃ Elapsed  ┃  Created At  ┃  Running At  ┃ Finished At ┃
    ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
    │ 185995311804 │ flwrlabs/my… │ finished:co… │ 00:00:55 │ 2024-12-16   │ 2024-12-16   │ 2024-12-16  │
    │ 1441032      │ (v1.0.0)     │              │          │ 11:12:33Z    │ 11:12:33Z    │ 11:13:29Z   │
    ├──────────────┼──────────────┼──────────────┼──────────┼──────────────┼──────────────┼─────────────┤
    │ 142007406570 │ flwrlabs/my… │ running      │ 00:00:05 │ 2024-12-16   │ 2024-12-16   │ N/A         │
    │ 11601420     │ (v1.0.0)     │              │          │ 12:18:39Z    │ 12:18:39Z    │             │
    └──────────────┴──────────────┴──────────────┴──────────┴──────────────┴──────────────┴─────────────┘

Similar to the ``flwr run`` command, to get the output in JSON format, simply pass the
``--format json`` flag:

.. code-block:: bash

    $ flwr ls --format json
    {
      "success": true,
      "runs": [
        {
          "run-id": 1859953118041441032,
          "fab-id": "flwrlabs/myawesomeapp1",
          "fab-name": "myawesomeapp1",
          "fab-version": "1.0.0",
          "fab-hash": "014c8eb3",
          "status": "finished:completed",
          "elapsed": "00:00:55",
          "created-at": "2024-12-16 11:12:33Z",
          "running-at": "2024-12-16 11:12:33Z",
          "finished-at": "2024-12-16 11:13:29Z"
        },
        {
          "run-id": 14200740657011601420,
          "fab-id": "flwrlabs/myawesomeapp2",
          "fab-name": "myawesomeapp2",
          "fab-version": "1.0.0",
          "fab-hash": "014c8eb3",
          "status": "running",
          "elapsed": "00:00:09",
          "created-at": "2024-12-16 12:18:39Z",
          "running-at": "2024-12-16 12:18:39Z",
          "finished-at": "N/A"
        },
      ]
    }

If the command fails, the JSON output will return two fields, ``success`` and
``error-message``, as shown in the :ref:`failed command <json_error_output>` example.

``flwr stop`` JSON output
-------------------------

The |flwr_stop|_ command stops a running Flower app for a provided run ID. By default,
the command prints the status of the stop process as follows:

.. code-block:: bash

    $ flwr stop 1859953118041441032
    Loading project configuration...
    Success
    ✋ Stopping run ID 1859953118041441032...
    ✅ Run 1859953118041441032 successfully stopped.

To get the output in JSON format, simply pass the ``--format json`` flag:

.. code-block:: bash

    $ flwr stop 1859953118041441032 --format json
    {
      "success": true,
      "run-id": 1859953118041441032,
    }

If the command fails, the JSON output will contain two fields ``success`` with the value
of ``false`` and ``error-message``, as shown in the :ref:`failed command
<json_error_output>` example.
