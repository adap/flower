Enable SSL connections
======================

This guide describes how to a SSL-enabled secure Flower server can be started and
how a Flower client can establish a secure connections to it.

A complete code example demonstrating a secure connection can be found 
`here <https://github.com/adap/flower/tree/main/examples/advanced-tensorflow>`_.

The code example comes with a README.md file which will explain how to start it. Although it is
already SSL-enabled, it might be less descriptive on how. Stick to this guide for a deeper
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

The approach how the SSL certificates are generated in this example can serve as an inspiration and
starting point but should not be taken as complete for production environments. Please refer to other
sources regarding the issue of correctly generating certificates for production environments.

In case you are a researcher you might be just fine using the self-signed certificates generated using
the scripts which are part of this guide.


Server
------

We are now going to show how to write a sever which uses the previously generated scripts.

.. code-block:: bash

    flower-superlink --certificates certificates/ca.crt certificates/server.pem certificates/server.key

When providing certificates, the server expects a tuple of three certificates paths: CA certificate, server certificate, and server private key.


Client
------

We are now going to show how to write a client which uses the previously generated scripts:

.. code-block:: bash

    flower-client-app client:app
        --root-certificates certificates/ca.crt
        --server 127.0.0.1:9092

When setting :code:`root_certificates`, the client expects a file path to a PEM-encoded root certificates.


Conclusion
----------

You should now have learned how to generate self-signed certificates using the given script, start a
SSL-enabled server, and have a client establish a secure connection to it.


Additional resources
--------------------

These additional sources might be relevant if you would like to dive deeper into the topic of certificates:

* `Let's Encrypt <https://letsencrypt.org/docs/>`_
* `certbot <https://certbot.eff.org/>`_
