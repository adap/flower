:og:description: Enable authentication for SuperNodes and SuperLink in Flower with public key authentication, securing federated learning via TLS connections.
.. meta::
    :description: Enable authentication for SuperNodes and SuperLink in Flower with public key authentication, securing federated learning via TLS connections.

.. |flower_cli_supernode_link| replace:: ``Flower CLI``

.. _flower_cli_supernode_link: ref-api-cli.html#flwr-supernode

Authenticate SuperNodes
=======================

When running a Flower Federation (see :doc:`ref-flower-network-communication`) it is
fundamental that an authentication mechanism is available between the SuperLink and the
SuperNodes that connect to it. Flower comes with two different mechanisms to
authenticate SuperNodes that connect to a running SuperLink:

- **Automatic authentication**: In this mode, the SuperLink checks the timestamp-based
  signature in each request from SuperNodes to prevent impersonation and replay attacks.
- **CSV-based authentication**: This mode functions similarly to automatic
  authentication but requires the SuperLink to be provided with a list of authorized
  public keys, allowing only those SuperNodes to connect.

The automatic authentication mode works out of the box and therefore requires no
configuration. On the other hand, CSV-based authentication mode is more sophisticated
and how it works and how it can be used is presented reminder of this guide. Flower's
CSV-based node authentication leverages a signature-based mechanism to verify each
node's identity and is only available when encrypted connections (SSL/TLS) are enabled:

- Each SuperNode must already possess a unique Elliptic Curve (EC) public/private key
  pair.
- The SuperLink (server) maintains a whitelist of EC public keys for all trusted
  SuperNodes (clients).
- A SuperNode signs a timestamp with its private key and sends the signed timestamp to
  the SuperLink.
- The SuperLink verifies the signature and timestamp using the SuperNode's public key.

.. note::

    This guide builds on the Flower App setup presented in the
    :doc:`how-to-enable-tls-connections` guide and extends it to introduce node
    authentication to the SuperLink ↔ SuperNode connection.

.. tip::

    Checkout the `Flower Authentication
    <https://github.com/adap/flower/tree/main/examples/flower-authentication>`_ example
    for a complete self-contained example on how to setup TLS and node authentication.

Generate authentication keys
----------------------------

To establish an authentication mechanism by which only authorized SuperNodes can connect
to a running SuperLink, a set of key pairs for both SuperLink and SuperNodes need to be
created.

We have prepared a script that can be used to generate such set of keys. While using
these are fine for prototyping, we advice you to follow the standards set in your
team/organization and generated the keys and share them with the corresponding parties.
Refer to the **Generate public and private keys for SuperNode authentication** section
in the example linked at the top of this guide.

.. code-block:: bash

    # In the example directory, generate the public/private key pairs
    $ ./generate_auth_keys.sh

This will generate the keys in a new ``keys/`` directory. By default it creates a key
pair for the SuperLink and one for each SuperNode. Copy this directory into the
directory of your app (e.g. a directory generated earlier via ``flwr new``).

Enable node authentication in SuperLink
---------------------------------------

To launch a SuperLink with SuperNode authentication enabled, you need to provide three
aditional files in addition to the certificates needed for the TLS connections. Recall
that the authentication feature can only be enabled in the presence of TLS.

.. code-block:: bash
    :emphasize-lines: 5

    $ flower-superlink \
        --ssl-ca-certfile certificates/ca.crt \
        --ssl-certfile certificates/server.pem \
        --ssl-keyfile certificates/server.key \
        --enable-supernode-auth

.. dropdown:: Understand the command

    * ``--enable-supernode-auth``: Enables SuperNode authentication therefore only Supernodes that are first register on the SuperLink will be able to establish a connection.

Register SuperNodes
-------------------

With a SuperLink already running, we can proceed to register the SuperNodes that will be
allowed to connect to it. This is done via the |flower_cli_supernode_link|_ using the
public keys generated earlier for each SuperNode that we want to eventually connect to
the SuperLink. Let's see how to this looks in code:

.. code-block:: bash

    # flwr supernode register <supernode-pub-key> <app> <federation>
    $ flwr supernode register keys/client_credentials_1.pub . local-deployment

Let's proceed and also register the second SuperNode:

.. code-block:: bash

    $ flwr supernode register keys/client_credentials_2.pub . local-deployment

You can list the registered SuperNodes using the following command:

.. code-block:: bash

    $ flwr supernode list . local-deployment

which should display the IDs of the SuperNodes you just registered as well as their
status. You should see a table similar to the following:

.. code-block:: bash

    ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
    ┃       Node ID        ┃   Owner    ┃   Status   ┃ Elapsed  ┃   Status Changed @   ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
    │ 16019329408659850374 │ sys_noauth │ registered │          │ N/A                  │
    ├──────────────────────┼────────────┼────────────┼──────────┼──────────────────────┤
    │ 8392976743692794070  │ sys_noauth │ registered │          │ N/A                  │
    └──────────────────────┴────────────┴────────────┴──────────┴──────────────────────┘

The status of the SuperNodes will change after they connect to the SuperLink. Let's
proceed and laucnh the SuperNodes.

Enable node authentication in SuperNode
---------------------------------------

Connecting a SuperNode to a SuperLink that has node authentication enabled requires
passing two additional arguments (i.e. the public and private keys of the SuperNode) in
addition to the TLS certificate.

.. code-block:: bash
    :emphasize-lines: 6, 7

    $ flower-supernode \
        --root-certificates certificates/ca.crt \
        --superlink 127.0.0.1:9092 \
        --clientappio-api-address 0.0.0.0:9094 \
        --node-config="partition-id=0 num-partitions=2" \
        --auth-supernode-private-key keys/client_credentials_1 \
        --auth-supernode-public-key keys/client_credentials_1.pub

.. dropdown:: Understand the command

    * ``--auth-supernode-private-key``: the private key of this SuperNode.
    * | ``--auth-supernode-public-key``: the public key of this SuperNode (which should be the same that was added to othe CSV used by the SuperLink).

Follow the same procedure to launch the second SuperNode by passing its corresponding
key pair:

.. code-block:: bash
    :emphasize-lines: 6, 7

    $ flower-supernode \
        --root-certificates certificates/ca.crt \
        --superlink 127.0.0.1:9092 \
        --clientappio-api-address 0.0.0.0:9095 \
        --node-config="partition-id=1 num-partitions=2" \
        --auth-supernode-private-key keys/client_credentials_2 \
        --auth-supernode-public-key keys/client_credentials_2.pub

Security notice
---------------

The system's security relies on the credentials of the SuperLink and each SuperNode.
Therefore, it is imperative to safeguard and safely store the credentials to avoid
security risks such as Public Key Infrastructure (PKI) impersonation attacks. The node
authentication mechanism also involves human interaction, so please ensure that all of
the communication is done in a secure manner, using trusted communication methods.

Conclusion
----------

You should now have learned how to start a long-running Flower SuperLink and SuperNode
with node authentication enabled. You should also know the significance of the private
key and store it securely to minimize risks.

.. note::

    Refer to the :doc:`docker/index` documentation to learn how to setup a federation
    where each component runs in its own Docker container. You can make use of TLS and
    other security features in Flower such as implement a SuperNode authentication
    mechanism.
