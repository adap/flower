Enable TLS connections
======================

This guide describes how to a TLS-enabled secure Flower server (``SuperLink``) can be
started and how a Flower client (``SuperNode``) can establish a secure connections to
it.

A complete code example demonstrating a secure connection can be found `here
<https://github.com/adap/flower/tree/main/examples/advanced-tensorflow>`_.

The code example comes with a ``README.md`` file which explains how to start it.
Although it is already TLS-enabled, it might be less descriptive on how it does so.
Stick to this guide for a deeper introduction to the topic.

Certificates
------------

Using TLS-enabled connections requires certificates to be passed to the server and
client. For the purpose of this guide we are going to generate self-signed certificates.
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
