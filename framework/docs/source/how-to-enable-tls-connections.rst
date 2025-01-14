Enable TLS connections
======================

Transport Layer Security (TLS) ensures the communication between endpoints is encrypted.
This guide describes how to establish secure TLS Superlink ↔ SuperNodes as well as User
↔ SuperLink connections.

.. note::

    This guide builds on the Flower App setup presented in
    :doc:`how-to-run-flower-with-deployment-engine` and extends it to replace the use of
    ``--insecure`` in favour of TLS.

.. tip::

    Checkout the `Flower Authentication
    <https://github.com/adap/flower/tree/main/examples/flower-authentication>`_ example
    for a complete self-contained example on how to setup TLS and (optionally) node
    authentication. Check out the :doc:`how-to-authenticate-supernodes` guide to learn
    more about adding an authentication layer to SuperLink ↔ SuperNode connections.

Certificates
------------

Using TLS-enabled connections expects some certificates generated and passed when
launching the SuperLink, the SuperNodes and when a user (e.g. a data scientist that
wants to submit a ``Run``) interacts with the federation via the `flwr CLI
<ref-api-cli>`_.

We have prepared a script that can be used to generate such set of certificates. While
using these are fine for prototyping, we advice you to follow the standards set in your
team/organization and generated the certificates and share them with the corresponding
parties.

As this can become quite complex we are going to ask you to run the script in
``examples/advanced-tensorflow/certificates/generate.sh`` with the following command
sequence:

.. code-block:: bash

    $ cd examples/advanced-tensorflow/certificates && \
        ./generate.sh

This will generate the certificates in
``examples/advanced-tensorflow/.cache/certificates``.

The approach for generating TLS certificates in the context of this example can serve as
an inspiration and starting point, but it should not be used as a reference for
production environments. Please refer to other sources regarding the issue of correctly
generating certificates for production environments. For non-critical prototyping or
research projects, it might be sufficient to use the self-signed certificates generated
using the scripts mentioned in this guide.

Server (SuperLink)
------------------

Navigate to the ``examples/advanced-tensorflow`` folder (`here
<https://github.com/adap/flower/tree/main/examples/advanced-tensorflow>`_) and use the
following terminal command to start a server (SuperLink) that uses the previously
generated certificates:

.. code-block:: bash

    $ flower-superlink \
        --ssl-ca-certfile .cache/certificates/ca.crt \
        --ssl-certfile .cache/certificates/server.pem \
        --ssl-keyfile .cache/certificates/server.key

When providing certificates, the server expects a tuple of three certificates paths: CA
certificate, server certificate and server private key.

Clients (SuperNode)
-------------------

Use the following terminal command to start a client (SuperNode) that uses the
previously generated certificates:

.. code-block:: bash

    $ flower-supernode \
        --root-certificates .cache/certificates/ca.crt \
        --superlink 127.0.0.1:9092 \
        --clientappio-api-address 0.0.0.0:9095 \
        --node-config="partition-id=0 num-partitions=10"

When setting ``root_certificates``, the client expects a file path to PEM-encoded root
certificates.

In another terminal, start a second SuperNode that uses the same certificates:

.. code-block:: bash

    $ flower-supernode \
        --root-certificates .cache/certificates/ca.crt \
        --superlink 127.0.0.1:9092 \
        --clientappio-api-address 0.0.0.0:9096 \
        --node-config="partition-id=1 num-partitions=10"

Note that in the second SuperNode, if you run both on the same machine, you must specify
a different port for the ``ClientAppIO`` API address to avoid clashing with the first
SuperNode.

Executing ``flwr run`` with TLS
-------------------------------

The root certificates used for executing ``flwr run`` is specified in the
``pyproject.toml`` of your app.

.. code-block:: toml

    [tool.flwr.federations.local-deployment]
    address = "127.0.0.1:9093"
    root-certificates = "./.cache/certificates/ca.crt"

Note that the path to the ``root-certificates`` is relative to the root of the project.
Now, you can run the example by executing the following:

.. code-block:: bash

    $ flwr run . local-deployment --stream

Conclusion
----------

You should now have learned how to generate self-signed certificates using the given
script, start an TLS-enabled server and have two clients establish secure connections to
it. You should also have learned how to run your Flower project using ``flwr run`` with
TLS enabled.

.. note::

    For running a Docker setup with TLS enabled, please refer to
    :doc:`docker/enable-tls`.

Additional resources
--------------------

These additional sources might be relevant if you would like to dive deeper into the
topic of certificates:

- `Let's Encrypt <https://letsencrypt.org/docs/>`_
- `certbot <https://certbot.eff.org/>`_
