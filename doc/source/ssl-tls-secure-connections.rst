Guide: SSL/TLS enabled Server and Client
=================================

In this segment we will learn, how to start a SSL/TLS enabled secure server and
establish a secure connection to it with a Flower client.

A more involved code example in which a connection is used can be found 
`here <https://github.com/adap/flower/tree/main/examples/advanced_tensorflow>`_.

It has it's own README.md file which will 

Certificates
------------

Using SSL/TLS for a secure connection requires certificates to be passed to the server and client.
For the purpose of this guide we are going to generate self-signed certificates. As this can become
quite complex we are going to ask you to run the script in
`examples/advanced_tensorflow/certificates/generate.sh`

with the following command sequence:

.. code-block:: bash

  cd examples/advanced_tensorflow/certificates
  ./generate.sh

This will generate the certificates in `examples/advanced_tensorflow/.cache/certificates`

The approach how the SSL certificates are generated in this example can serve as an inspiration and
starting point but should not be taken as complete for production environments. Please refer to other
sources regarding the issue of correctly generating certificates for production environments.

In case you are a researcher you might be fine using the self signed certificates generated using the
scripts which are part of this guide.

Server Setup
------------

To

.. code-block:: python

  