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

.. tip::

    The code snippets shown in this guide are borrowed from the `Flower Authentication
    Example <https://github.com/adap/flower/tree/main/examples/flower-authentication>`_.
    Check it out if you want to learn more on how to setup a federation with TLS and
    authenticated SuperLink ↔ SuperNode connections.

.. note::

    This guide covers a preview feature that might change in future versions of Flower.

Enable node authentication in ``SuperLink``
-------------------------------------------

To enable node authentication, first you need to configure SSL/TLS for **SuperLink ↔
SuperNode** communication. Refer to the :doc:`how-to-enable-ssl-connections` guide for
details. After configuring secure connections, use the following terminal command to
start a Flower SuperLink that has both secure connections and node authentication
enabled:

.. code-block:: bash

    flower-superlink \
        --ssl-ca-certfile certificates/ca.crt \
        --ssl-certfile certificates/server.pem \
        --ssl-keyfile certificates/server.key \
        --auth-list-public-keys keys/client_public_keys.csv \
        --auth-superlink-private-key keys/server_credentials \
        --auth-superlink-public-key keys/server_credentials.pub

.. dropdown:: Understand the command

    * ``--ssl-ca-certfile``: Specify the location of the CA certificate file in your file. This file is a certificate that is used to verify the identity of the SuperLink.
    * | ``--ssl-certfile``: Specify the location of the SuperLink's TLS certificate file. This file is used to identify the SuperLink and to encrypt the packages that are transmitted over the network.
    * | ``--ssl-keyfile``: Specify the location of the SuperLink's TLS private key file. This file is used to decrypt the packages that are transmitted over the network.
    * | ``--auth-list-public-keys``: Specify the path to a CSV file storing the public keys of all SuperNodes that should be allowed to connect with the SuperLink.
      | A valid CSV file storing known node public keys should list the keys in OpenSSH format, separated by commas and without any comments. Refer to the code sample, which contains a CSV file with two known node public keys.
    * | ``--auth-superlink-private-key``: the private key of the SuperLink.
    * | ``--auth-superlink-public-key``: the public key of the SuperLink.

.. note::

    Currently, there is no support for dynamically removing, editing, or adding known
    node public keys to a running SuperLink. To change the set of known nodes, you need
    to shut the SuperLink down, edit the CSV file, and start the SuperLink again.
    Support for dynamically changing the set of known nodes will be available in an
    upcoming Flower release.

Enable node authentication in ``SuperNode``
-------------------------------------------

Similar to the long-running Flower server (``SuperLink``), you can easily enable node
authentication in the long-running Flower client (``SuperNode``). Use the following
terminal command to start an authenticated ``SuperNode``:

.. code-block:: bash

    flower-supernode \
         --root-certificates certificates/ca.crt \
         --superlink 127.0.0.1:9092 \
         --auth-supernode-private-key keys/client_credentials \
         --auth-supernode-public-key keys/client_credentials.pub

.. dropdown:: Understand the command

    * ``--superlink``: The address of the SuperLink this SuperNode is connecting to. By default this connection happens over port 9092. This could be configure by means of the ``--fleet-api-address`` when launching the SuperLink.
    * | ``--auth-supernode-private-key``: the private key of this SuperNode.
    * | ``--auth-supernode-public-key``: the public key of this SuperNode (which should be the same that was added to othe CSV used by the SuperLink).

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
key and store it safely to minimize security risks.
