Authenticate SuperNodes
=======================

Flower has built-in support for authenticated SuperNodes that you can use to verify the
identities of each SuperNode connecting to a SuperLink. For increased security, node
authentication can only be used when encrypted connections (SSL/TLS) are enabled. Flower
node authentication works similar to how GitHub SSH authentication works:

- SuperLink (server) stores a list of public keys of known SuperNodes (clients)
- Using ECDH, both SuperNode and SuperLink independently derive a shared secret
- Shared secret is used to compute the HMAC value of the message sent from SuperNode to
  SuperLink as a token
- SuperLink verifies the token

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
    :emphasize-lines: 5,6,7

    $ flower-superlink \
        --ssl-ca-certfile certificates/ca.crt \
        --ssl-certfile certificates/server.pem \
        --ssl-keyfile certificates/server.key \
        --auth-list-public-keys keys/client_public_keys.csv \
        --auth-superlink-private-key keys/server_credentials \
        --auth-superlink-public-key keys/server_credentials.pub

.. dropdown:: Understand the command

    * ``--auth-list-public-keys``: Specify the path to a CSV file storing the public keys of all SuperNodes that should be allowed to connect with the SuperLink.
      | A valid CSV file storing known node public keys should list the keys in OpenSSH format, separated by commas and without any comments. Refer to the code sample, which contains a CSV file with two known node public keys.
    * | ``--auth-superlink-private-key``: the private key of the SuperLink.
    * | ``--auth-superlink-public-key``: the public key of the SuperLink.

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
