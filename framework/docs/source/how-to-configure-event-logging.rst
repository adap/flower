:og:description: Configure SuperLink for event/audit logging to capture or store events such as user interactions and application behavior.
.. meta::
    :description: Configure SuperLink for event/audit logging to capture or store events such as user interactions and application behavior.

Event Logging
=============

.. note::

    This guide describes a Flower Enterprise feature. Subscribe to the `Flower
    Enterprise plan <https://flower.ai/enterprise>`_ to get access to the advanced
    features for your organization.

In this guide, you'll learn how to configure SuperLink with event logging. Event logging
allows you to capture and store events that occur in the SuperLink, such as valuable
insights into the application's behavior and performance, and when a user interfaces
with the system.

The event logging feature brings JSON-formatted log outputs to Flower. It can be
activated for the SuperLink to record events when users or SuperNodes interact with the
SuperLink. System administrators can then configure a logging backend to systematically
capture these events in a database e.g. for audit purposes.

User events occur when a Flower user interacts with SuperLink, such as logging in,
starting a run, or querying the list of runs on the SuperLink. These events capture the
interaction of the user via the ``flwr`` CLI and the SuperLink.

Application events occur when the Flower components interact with one another,
specifically, between the SuperLink and SuperNodes. These events will show, for example,
when messages are pushed and pulled from/to the SuperLink and when a SuperNode
establishes a connection with the SuperLink.

By default, the output schema is as follows:

.. code-block:: json

    {
      "timestamp": "2025-07-31T08:00:00Z",
      "actor": {
        "id": "<Flower Account ID> or <Node ID>",
        "description": "<account_name> or 'SuperNode'",
        "ip_address": "<IP address of user or SuperNode>",
      },
      "event": {
        "action": "<Method name>",
        "run_id": "<Run ID>",
        "fab_hash": "<FAB hash>",
      },
      "status": "<started/completed/failed>"
    }

where,

.. list-table::
    :header-rows: 1

    - - Field
      - Description
    - - ``timestamp``
      - Timestamp of the event in UTC format and RFC-3339 compliant
    - - ``actor.id``
      - Flower account ID (when called by a ``flwr`` CLI user) or SuperNode ID (when
        called by a SuperNode)
    - - ``actor.description``
      - Username registered on the OIDC provider or ``SuperNode``
    - - ``actor.ip_address``
      - IPv4 or IPv6 address of the actor
    - - ``event.action``
      - Name of the servicer method, e.g.
        ``ExecServicer.StartRun``/``FleetServicer.PullMessages``
    - - ``event.run_id``
      - The run ID of the Flower workflow
    - - ``event.fab_hash``
      - The FAB hash of the Flower app
    - - ``status``
      - A string describing whether the action is started, completed or failed

Enable Event Logging
--------------------

To enable event logging, start the SuperLink with the argument ``--enable-event-log`` as
follows:

.. code-block:: shell

    ➜ flower-superlink --enable-event-log <other flags>

Note that the event logging feature can only be activated with the :doc:`user
authentication feature <how-to-login-to-superlink>`.

Here is an example output when a user runs ``flwr run`` (note the ``"action":
"ExecServicer.StartRun"``) :

.. code-block:: shell

    INFO :      [AUDIT] {"timestamp": "2025-07-12T10:24:21Z", "actor": {"actor_id": "...", "description": "...", "ip_address": "..."}, "event": {"action": "ExecServicer.StartRun", "run_id": "...", "fab_hash": "..."}, "status": "started"}
    INFO :      ExecServicer.StartRun
    INFO :      [AUDIT] {"timestamp": "2025-07-12T10:24:21Z", "actor": {"actor_id": "...", "description": "...", "ip_address": "..."}, "event": {"action": "ExecServicer.StartRun", "run_id": "...", "fab_hash": "..."}, "status": "completed"}

Here is another example output when a user runs ``flwr ls``:

.. code-block:: shell

    INFO :      [AUDIT] {"timestamp": "2025-07-12T10:26:35Z", "actor": {"actor_id": "...", "description": "...", "ip_address": "..."}, "event": {"action": "ExecServicer.ListRuns", "run_id": null, "fab_hash": null}, "status": "started"}
    INFO :      ExecServicer.List
    INFO :      [AUDIT] {"timestamp": "2025-07-12T10:26:35Z", "actor": {"actor_id": "...", "description": "...", "ip_address": "..."}, "event": {"action": "ExecServicer.ListRuns", "run_id": null, "fab_hash": null}, "status": "completed"}

And here is an example when a SuperNode pulls a message from the SuperLink:

.. code-block:: shell

    INFO :      [AUDIT] {"timestamp": "2025-07-14T10:27:02Z", "actor": {"actor_id": "...", "description": "SuperNode", "ip_address": "..."}, "event": {"action": "FleetServicer.PullMessages", "run_id": null, "fab_hash": null}, "status": "started"}
    INFO :      [Fleet.PullMessages] node_id=...
    INFO :      [AUDIT] {"timestamp": "2025-07-14T10:27:02Z", "actor": {"actor_id": "...", "description": "SuperNode", "ip_address": "..."}, "event": {"action": "FleetServicer.PullMessages", "run_id": null, "fab_hash": null}, "status": "completed"}
