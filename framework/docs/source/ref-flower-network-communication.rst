:og:description: The Flower Network Communication reference describes all mandatory and optional network connections in Flower federated AI systems.
.. meta::
    :description: The Flower Network Communication reference describes all mandatory and optional network connections in Flower federated AI systems.

Flower Network Communication
============================

This reference complements the `Flower Architecture
<explanation-flower-architecture.html>`_ explanation by detailing the network
connections used in a deployed Flower federated AI system.

.. note::

    Optionally, a third connection to a third-party service can be established to
    provide user-level authentication via `OIDC
    <https://openid.net/developers/how-connect-works/>`_. This means that only users
    authenticated via ``flwr login`` are able to interface with the SuperLink.

.. raw:: html

    <div id="diagram1" style="display:block;">
        <img src="./_static/flower-network-diagram-subprocess.svg" alt="Flower Network Diagram (subprocess)">
    </div>
    <div id="diagram2" style="display:none;">
        <img src="./_static/flower-network-diagram-process.svg" alt="Flower Network Diagram (process)">
    </div>
    <div style="text-align: center; margin-bottom: 1em;">
        <button onclick="document.getElementById('diagram1').style.display='block'; document.getElementById('diagram2').style.display='none';">Subprocess Mode</button>
        <button onclick="document.getElementById('diagram1').style.display='none'; document.getElementById('diagram2').style.display='block';">Process Mode</button>
    </div>

.. tip::

    Click the buttons above to toggle between the network diagrams for isolation modes
    **subprocess** and **process**.

Mandatory Network Connections
-----------------------------

Deployed Flower systems have at least two types of network connections:

- **CLI to SuperLink (Exec API)**: The ``flwr`` `CLI command <ref-api-cli.html>`_,
  typically run on the users workstation, is used to interface with a deployed Flower
  federation consisting of SuperLink and SuperNodes. From a networking perspective, the
  ``flwr`` CLI acts as a gRPC client and the SuperLink acts as a gRPC server. The
  ``flwr`` CLI is the only way for a user (AI researchers, data scientist) to inferface
  with a deployed Flower federation. They cannot, for example, interface directly with
  SuperNodes connected to the SuperLink. The ``flwr`` CLI to SuperLink connection should
  always use TLS, but ``insecure`` mode is supported for local testing.
- **SuperNode to SuperLink (Fleet API)**: In Flower terminology, a Flower federation is
  a set of SuperNodes connected to the same SuperLink. From a networking perspective,
  each SuperNode acts as a gRPC client and the SuperLink acts as a gRPC server. This
  means that, when deploying a SuperNode, only outgoing connections are necessary to
  connect to the SuperLink. Only the SuperNodes can initiate such requests and they do
  not respond to incoming requests. The SuperNode to SuperLink connection should always
  use TLS (see :doc:`how-to-enable-tls-connections` to learn more), but ``insecure``
  mode is supported for local testing.

Optional Network Connections
----------------------------

Depending on the SuperLink and SuperNode configuration, Flower systems can have/use a
number of additional network connections.

Isolation mode
~~~~~~~~~~~~~~

Both Flower SuperLink and Flower SuperNode can use different isolation modes. Isolation
mode ``subprocess`` configures the SuperLink/SuperNode to run ServerApp/ClientApp in a
sub-process. Isolation mode ``process`` expects ServerApp or ClientApp to run in
separate externally-managed processes. This allows, for example, to run SuperNode and
``ClientApp`` in separate Docker containers with different sets of dependencies
installed. Check the :doc:`docker/index` guide to gain a better understanding on how to
use both modes.

In isolation mode ``process``, additional network connections are necessary to allow the
external process running ServerApp or ClientApp to communicate with the long-running
SuperLink or SuperNode:

- **flwr-serverapp to SuperLink (ServerAppIO API)**: The process running the
  ``ServerApp``, ``flwr-serveapp``, acts as a gRPC client and connects to the
  SuperLink's ServerAppIO API. This connection enables the ``flwr-serverapp`` process to
  pull the necessary inputs to execute the ``ServerApp``. It also allows the
  ``ServerApp``, once running, to do typical things like sending/receiving messages
  to/from available SuperNodes (via the SuperLink).
- **flwr-clientapp to SuperNode (ClientAppIO API)**: The process running the
  ``ClientApp``, ``flwr-clientapp``, acts as a gRPC client and connects to the
  SuperNode's ClientAppIO API. This connection enables the ``flwr-clientapp`` process to
  pull the necessary details (e.g., FAB file) to execute the ``ClientApp``, execute the
  ``ClientApp`` (e.g., local model training) and return the execution results (e.g.,
  locally update model parameters) to the SuperNode.

.. note::

    In the current version of Flower, both of these connections are insecure because
    Flower expects SuperLink/SuperNode and ``flwr-serverapp`` / ``flwr-clientapp`` to be
    run in the same network. ``flwr-serverapp`` / ``flwr-clientapp`` and
    SuperLink/SuperNode should never communicate over untrusted networks (e.g., public
    internet).

User authentication
~~~~~~~~~~~~~~~~~~~

When user authentication is enabled, Flower uses an OIDC-compatible server to
authenticate requests:

- **SuperLink to OIDC server**: A SuperLink can optionally be configured to only allow
  authenticated users to interact with it. In this setting, the Flower SuperLink acts as
  a REST client to the OIDC-compatible server.

Application-specific connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users who write Flower Apps (``ServerApp`` and ``ClientApp``) can also make additional
network requests. This is, strictly speaking, not part of Flower as a Federated AI
Platform. It is a decision of (a) the user about what kinds of third-party systems their
Flower App should connect to and (b) the system administrator about what kinds of
connections they want to allow.

Typical examples include:

- **ClientApp to Database**: ``ClientApp`` instances typically need to be able to access
  the data to perform the action they have been designed for (e.g. train locally a
  model, run a DB query). How this connection is established depends on what storage
  technology is used at the client side. Note that in the diagram above, we show two
  representative connections to DBs in Client-A and Client-B. Your DB connection(s) may
  likely be different to the illustration above.
- **ServerApp to Database**: ``ServerApp`` instances might want to access the data to
  perform the action they have been designed for (e.g. evaluate a model on some data
  after aggregation). How this connection is established depends on what storage
  technology used at the client side. Note that in the diagram above we have omitted
  showing a DB connected to the ServerApp components.
- **ServerApp to metric logging service**: Metric logging services like TensorBoard,
  MLFlow and Weights & Biases are often used to track the progress of training runs. In
  this setting, the ``ServerApp`` typically acts as a client to the metric logging
  service.
