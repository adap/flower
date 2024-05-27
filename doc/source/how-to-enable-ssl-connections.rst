Enable SSL connections
======================

This guide describes how to a SSL-enabled secure Flower server (:code:`SuperLink`) can be started and
how a Flower client (:code:`SuperNode`) can establish a secure connections to it.

A complete code example demonstrating a secure connection can be found 
`here <https://github.com/adap/flower/tree/main/examples/advanced-tensorflow>`_.

The code example comes with a :code:`README.md` file which explains how to start it. Although it is
already SSL-enabled, it might be less descriptive on how it does so. Stick to this guide for a deeper
introduction to the topic.


Certificates
------------

Using SSL-enabled connections requires certificates to be passed to the server and client. For
the purpose of this guide we are going to generate self-signed certificates. As this can become
quite complex we are going to ask you to run the script in
:code:`examples/advanced-tensorflow/certificates/generate.sh`
with the following command sequence:

.. code-block:: bash

  cd examples/advanced-tensorflow/certificates
  ./generate.sh

This will generate the certificates in :code:`examples/advanced-tensorflow/.cache/certificates`.

The approach for generating SSL certificates in the context of this example can serve as an inspiration and
starting point, but it should not be used as a reference for production environments. Please refer to other
sources regarding the issue of correctly generating certificates for production environments.
For non-critical prototyping or research projects, it might be sufficient to use the self-signed certificates generated using
the scripts mentioned in this guide.


Server (SuperLink)
------------------

Use the following terminal command to start a sever (SuperLink) that uses the previously generated certificates:

.. code-block:: bash

    flower-superlink --certificates certificates/ca.crt certificates/server.pem certificates/server.key

When providing certificates, the server expects a tuple of three certificates paths: CA certificate, server certificate and server private key.


Client (SuperNode)
------------------

Use the following terminal command to start a client (SuperNode) that uses the previously generated certificates:

.. code-block:: bash

    flower-client-app client:app
        --root-certificates certificates/ca.crt
        --superlink-fleet-api 127.0.0.1:9092

When setting :code:`root_certificates`, the client expects a file path to PEM-encoded root certificates.


Conclusion
----------

You should now have learned how to generate self-signed certificates using the given script, start an
SSL-enabled server and have a client establish a secure connection to it.


Additional resources
--------------------

These additional sources might be relevant if you would like to dive deeper into the topic of certificates:

* `Let's Encrypt <https://letsencrypt.org/docs/>`_
* `certbot <https://certbot.eff.org/>`_
