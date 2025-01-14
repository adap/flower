Flower Network Diagram
======================

This page complements the content in the `Flower Architecture
<explanation-flower-architecture.html>`_ overview guide by detailing the connections
used in a deployed Flower system.

Fundamentally, there are two types of connections:

1. User ↔ ``SuperLink`` connections initiated via the `Flower CLI <ref-api-cli.html>`_.
2. Connections between Flower infrastructure components (e.g. ``SuperLink`` ↔
   ``SuperNode``).

The first connection enables a user (e.g. a data scientist) to perform actions such as submitting
a ``Run``, query the status of a ongoing ``Run``, and more. The second set of
connections are used by ``SuperLink`` and ``SuperNodes`` to establish and maintain a
secure connections for the ``ServerApp`` and ``ClientApp`` to communicate (e.g. model
parameters, metrics, etc).

.. note::

    Optionally, a third connection to a third-party service can be established to provide
    user-level authentication via `OIDC
    <https://openid.net/developers/how-connect-works/>`_. This means that only users
    authenticated via ``flwr login`` are able to interface with the ``SuperLink``.

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

    Click the buttons above to toggle between **subprocess** and **process** isolation
    modes. As the name suggest, the former implies that ``ServerApp`` and ``ClientApp``
    instances run as sub-processes of the ``SuperLink`` and ``SuperNode`` respectively.
    This mode is ideal when a simplified deployment is a priority. On the other hand,
    `process` isolation mode brings more flexibility to your Flower deployment by
    letting you decide where ``ClientApp``\s and ``ServerApp`` should run (e.g. on
    servers different to those where ``SuperNode``\s or ``SuperLink`` are running). Check the
    :doc:`docker/index` guide to gain a better understanding on how to use both modes.

What follows is a description of what each connection in the diagram represents:

- User ↔ ``SuperLink``: The only mechanism for a user to interface with the Flower
  system (e.g. to submit a ``Run``, retrieve the logs of a ``Run``, etc) is via the
  `Flower CLI <ref-api-cli.html>`_ commands. It is not possible for a user to interface
  with the ``SuperNodes``.
- ``SuperLink`` ↔ ``SuperNode``: A Flower federation is built on a series of
  ``SuperNodes`` connected to the same ``SuperLink``. This connection is used by each
  ``SuperNode`` to periodically check with the ``SuperLink`` if some action involving
  its local data (e.g. locally train a model) is required. Only ``SuperNodes`` can initiate such
  requests. ``SuperNodes`` do not respond to incoming requests. The ``SuperLink`` ↔
  ``SuperNode`` connection is implemented with gRPC. Furthermore, TLS connection can be enabled (see :doc:`how-to-enable-tls-connections` to learn more).
- ``SuperLink`` ↔ ``ServerApp``: This connection allows the ``ServerApp`` to interface
  with the ``LinkState``, a store in the ``SuperLink`` where a register of connected
  ``SuperNodes`` is kept and where messages returned from a ``SuperNode`` (e.g. the
  parameters of locally trained model) are temporally stored. This connection provides
  the ``ServerApp`` with capabilities such as sampling nodes, communicate the state of a
  `global model` (that ``ClientApps`` in the ``SuperNodes`` should train/evaluate
  locally), or perform the aggregation of model updates and metrics. The SuperLink ↔ ``ServerApp`` connection is established via gRPC.
- ``SuperNode`` ↔ ``ClientApp``: This connection allows the ``ClientApp`` to communicate
  with its ``SuperNode`` to fetch the inputs (e.g. model parameters sent from the
  ``ServerApp`` via the ``SuperLink``) needed to execute the task it has been designed
  for. This communication channel is also used for the ``SuperNode`` to receive the
  generated outputs (e.g. the parameters of a trained model) from the process running
  the ``ClientApp`` so that these outputs can be pushed to the ``SuperLink```. The ``SuperNode`` ↔
  ``ClientApp`` connection is established via gRPC.
- ``ClientApp`` ↔ DB: ``ClientApp`` instances need to be able to access the data to
  perform the action they have been designed for (e.g. train locally a model, run a DB
  query). How this connection is established depends on what storage technology used at
  the client side. Note that in the diagram above, we show two representative connections to DBs in Client-A and Client-B. Your DB connection(s) may likely be different to the illustration above.
- ``SuperLink`` ↔ `User Authentication Service`: A ``SuperLink`` may optionally be
  configured to allow authenticated users to interact with it. Flower uses the OpenID Connect authentication protocol to
  implement this feature.
