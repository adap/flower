Enable client authentication
============================

Flower has a built-in client authentication process that you can use to verify the identities of each client that participates in a federated learning routine. 

The client authentication process is similar to how Github SSH authentication works without the encryption:

* Server stores a list of known client public keys
* Using ECDH, both client and server independently derive a shared secret
* Shared secret is used to compute the hmac value of the message sent from client to server as a token
* Server verifies the token sent from the client

You can find the complete code example demonstrating federated learning with Flower in an authenticated setting
`here <https://github.com/adap/flower/tree/main/examples/flower-client-authentication>`_.

.. note::
    This guide covers experimental features that might change in future versions of Flower, and client authentication is only available in the gRPC-rere stack with SSL connection enabled.


Enable client authentication in Flower :code:`Superlink`
--------------------------------------------------------

To enable client authentication, first you need to establish SSL connection so that the server-client communication is secure. You can find the complete guide
`here <https://flower.ai/docs/framework/how-to-enable-ssl-connections.html>`_.
After establishing secure connection, you can now enable client authentication in a long-running Flower server :code:`Superlink` easily by typing the following code in a terminal:

.. code-block:: bash

    flower-superlink
        --certificates certificates/ca.crt certificates/server.pem certificates/server.key
        --require-client-authentication ./keys/client_public_keys.csv ./keys/server_credentials ./keys/server_credentials.pub
    
Let's break down the :code:`--require-client-authentication` flag:

1. The first argument is a path to a CSV file storing all known client public keys. You need to store all known client public keys that are allowed to participate in a federated learning routine in a CSV file.
2. The second and third arguments are paths to the server's private and public keys. You can generate a private and public key pair using :code:`ssh-keygen -t ecdsa -b 384`.

Currently, there is no support to dynamically remove, edit, or add known client public keys to the server, so you need to shutdown the server, manually change the csv file, and restart the server again.


Enable client authentication in Flower :code:`ClientApp`
--------------------------------------------------------

Similar to the long-running Flower server :code:`Superlink`, you can easily enable client authentication in the :code:`ClientApp` by typing the following code in a terminal:

.. code-block:: bash
    
    flower-client-app client:app
        --root-certificates certificates/ca.crt
        --server 127.0.0.1:9092
        --authentication-keys ./keys/client_credentials ./keys/client_credentials.pub

The :code:`--authentication-keys` flag expects two arguments: a path to the client's private and public key file. You can generate a private and public key pair using :code:`ssh-keygen -t ecdsa -b 384`.


Security notice
---------------

The system's security relies on the credentials of the server and each client. Therefore, it is imperative to safeguard and safely store the credentials to avoid security risks such as Public Key Infrastructure (PKI) impersonation attacks.
The client authentication mechanism also involves human interaction, so please ensure that all of the communication is done in a secure manner, using trusted communication methods.


Conclusion
----------

You should now have learned how to start a long-running server :code:`Superlink` and :code:`ClientApp` with client authentication enabled. You should also know the significance of the private key and store it safely to minimize security risks.


Extending client authentication
-------------------------------

Extending client authentication is a bit more involved, but you can implement different kinds of client authentication with Flower quickly. 
All you need to do is create your own server and client interceptor for gRPC. We recommend reading through the Flower source code for client authentication and gRPC examples for authentication before implementing your own authentication process.
