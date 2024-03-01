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

.. code-block:: python

  from pathlib import Path
  import flwr as fl

  # Start server
  fl.server.start_server(
      server_address="0.0.0.0:8080",
      config=fl.server.ServerConfig(num_rounds=4),
      certificates=(
          Path(".cache/certificates/ca.crt").read_bytes(),
          Path(".cache/certificates/server.pem").read_bytes(),
          Path(".cache/certificates/server.key").read_bytes(),
      )
  )

When providing certificates, the server expects a tuple of three certificates. :code:`Path` can be used to easily read the contents of those files into byte strings, which is the data type :code:`start_server` expects.


Client
------

We are now going to show how to write a client which uses the previously generated scripts:

.. code-block:: python

  from pathlib import Path
  import flwr as fl

  # Define client somewhere
  client = MyFlowerClient()

  # Start client
  fl.client.start_client(
      "localhost:8080",
      client=client.to_client(),
      root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
  )

When setting :code:`root_certificates`, the client expects the PEM-encoded root certificates as a byte string.
We are again using :code:`Path` to simplify reading those as byte strings.


Conclusion
----------

You should now have learned how to generate self-signed certificates using the given script, start a
SSL-enabled server, and have a client establish a secure connection to it.


Additional resources
--------------------

These additional sources might be relevant if you would like to dive deeper into the topic of certificates:

* `Let's Encrypt <https://letsencrypt.org/docs/>`_
* `certbot <https://certbot.eff.org/>`_
