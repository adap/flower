:og:description: Enable authentication for SuperNodes and SuperLink in Flower with public key authentication, securing federated learning via TLS connections.
.. meta::
    :description: Enable authentication for SuperNodes and SuperLink in Flower with public key authentication, securing federated learning via TLS connections.

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

.. note::

    This guide covers a preview feature that might change in future versions of Flower.

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
        --auth-list-public-keys keys/client_public_keys.csv

.. dropdown:: Understand the command

    * ``--auth-list-public-keys``: Specify the path to a CSV file storing the public keys of all SuperNodes that should be allowed to connect with the SuperLink. A valid CSV file storing known node public keys should list the keys in OpenSSH format, separated by commas. Refer to the code sample, which contains a CSV file with two known node public keys.

Enable node authentication in SuperNode
---------------------------------------

Connecting a SuperNode to a SuperLink that has node authentication enabled requires
passing one additional argument (i.e. the private key of the SuperNode) in
addition to the TLS certificate.

.. code-block:: bash
    :emphasize-lines: 6

    $ flower-supernode \
        --root-certificates certificates/ca.crt \
        --superlink 127.0.0.1:9092 \
        --clientappio-api-address 0.0.0.0:9094 \
        --node-config="partition-id=0 num-partitions=2" \
        --auth-supernode-private-key keys/client_credentials_1

.. dropdown:: Understand the command

    * ``--auth-supernode-private-key``: the private key of this SuperNode.

Follow the same procedure to launch the second SuperNode by passing its corresponding
key pair:

.. code-block:: bash
    :emphasize-lines: 6

    $ flower-supernode \
        --root-certificates certificates/ca.crt \
        --superlink 127.0.0.1:9092 \
        --clientappio-api-address 0.0.0.0:9095 \
        --node-config="partition-id=1 num-partitions=2" \
        --auth-supernode-private-key keys/client_credentials_2

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
