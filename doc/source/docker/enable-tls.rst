Enable TLS for Secure Connections
=================================

When operating in a production environment, it is strongly recommended to enable
Transport Layer Security (TLS) for each Flower Component to ensure secure communication.

To enable TLS, you will need a PEM-encoded root certificate, a PEM-encoded private key
and a PEM-encoded certificate chain.

.. note::

    For testing purposes, you can generate your own self-signed certificates. The
    `Enable SSL connections
    <https://flower.ai/docs/framework/how-to-enable-ssl-connections.html#certificates>`__
    page contains a section that will guide you through the process.

Because Flower containers, by default, run with a non-root user ``app``, the mounted
files and directories must have the proper permissions for the user ID ``49999``.

For example, to change the user ID of all files in the ``certificates/`` directory, you
can run ``sudo chown -R 49999:49999 certificates/*``.

If you later want to delete the directory, you can change the user ID back to the
current user ID by running ``sudo chown -R $USER:$(id -gn) certificates``.

SuperLink
---------

Assuming all files we need are in the local ``certificates`` directory, we can use the
flag ``--volume`` to mount the local directory into the ``/app/certificates/`` directory
of the container:

.. code-block:: bash
    :substitutions:

    $ docker run --rm \
         --volume ./certificates/:/app/certificates/:ro \
         flwr/superlink:|stable_flwr_version| \
         --ssl-ca-certfile certificates/ca.crt \
         --ssl-certfile certificates/server.pem \
         --ssl-keyfile certificates/server.key

.. dropdown:: Understanding the command

    * ``docker run``: This tells Docker to run a container from an image.
    * ``--rm``: Remove the container once it is stopped or the command exits.
    * | ``--volume ./certificates/:/app/certificates/:ro``: Mount the ``certificates`` directory in
      | the current working directory of the host machine as a read-only volume at the
      | ``/app/certificates`` directory inside the container.
      |
      | This allows the container to access the TLS certificates that are stored in the certificates
      | directory.
    * | :substitution-code:`flwr/superlink:|stable_flwr_version|`: The name of the image to be run and the specific
      | tag of the image. The tag :substitution-code:`|stable_flwr_version|` represents a specific version of the image.
    * | ``--ssl-ca-certfile certificates/ca.crt``: Specify the location of the CA certificate file
      | inside the container.
      |
      | The ``certificates/ca.crt`` file is a certificate that is used to verify the identity of the
      | SuperLink.
    * | ``--ssl-certfile certificates/server.pem``: Specify the location of the SuperLink's
      | TLS certificate file inside the container.
      |
      | The ``certificates/server.pem`` file is used to identify the SuperLink and to encrypt the
      | data that is transmitted over the network.
    * | ``--ssl-keyfile certificates/server.key``: Specify the location of the SuperLink's
      | TLS private key file inside the container.
      |
      | The ``certificates/server.key`` file is used to decrypt the data that is transmitted over
      | the network.

SuperNode
---------

Assuming that the ``ca.crt`` certificate already exists locally, we can use the flag
``--volume`` to mount the local certificate into the container's ``/app/`` directory.

.. note::

    If you're generating self-signed certificates and the ``ca.crt`` certificate doesn't
    exist on the SuperNode, you can copy it over after the generation step.

.. code-block:: bash
    :substitutions:

    $ docker run --rm \
         --volume ./ca.crt:/app/ca.crt/:ro \
         flwr/supernode:|stable_flwr_version| \
         --root-certificates ca.crt

.. dropdown:: Understanding the command

    * ``docker run``: This tells Docker to run a container from an image.
    * ``--rm``: Remove the container once it is stopped or the command exits.
    * | ``--volume ./ca.crt:/app/ca.crt/:ro``: Mount the ``ca.crt`` file from the
      | current working directory of the host machine as a read-only volume at the ``/app/ca.crt``
      | directory inside the container.
    * | :substitution-code:`flwr/supernode:|stable_flwr_version|`: The name of the image to be run and the specific
      | tag of the image. The tag :substitution-code:`|stable_flwr_version|` represents a specific version of the image.
    * | ``--root-certificates ca.crt``: This specifies the location of the CA certificate file
      | inside the container.
      |
      | The ``ca.crt`` file is used to verify the identity of the SuperLink.

SuperExec
---------

Assuming all files we need are in the local ``certificates`` directory where the
SuperExec will be executed from, we can use the flag ``--volume`` to mount the local
directory into the ``/app/certificates/`` directory of the container:

.. code-block:: bash
    :substitutions:

    $ docker run --rm \
         --volume ./certificates/:/app/certificates/:ro \
         flwr/superexec:|stable_flwr_version| \
         --ssl-ca-certfile certificates/ca.crt \
         --ssl-certfile certificates/server.pem \
         --ssl-keyfile certificates/server.key \
         --executor-config \
         root-certificates=\"certificates/superlink_ca.crt\"

.. dropdown:: Understanding the command

    * ``docker run``: This tells Docker to run a container from an image.
    * ``--rm``: Remove the container once it is stopped or the command exits.
    * | ``--volume ./certificates/:/app/certificates/:ro``: Mount the ``certificates`` directory in
      | the current working directory of the host machine as a read-only volume at the
      | ``/app/certificates`` directory inside the container.
      |
      | This allows the container to access the TLS certificates that are stored in the certificates
      | directory.
    * | :substitution-code:`flwr/superexec:|stable_flwr_version|`: The name of the image to be run and the specific
      | tag of the image. The tag :substitution-code:`|stable_flwr_version|` represents a specific version of the image.
    * | ``--ssl-ca-certfile certificates/ca.crt``: Specify the location of the CA certificate file
      | inside the container.
      |
      | The ``certificates/ca.crt`` file is a certificate that is used to verify the identity of the
      | SuperExec.
    * | ``--ssl-certfile certificates/server.pem``: Specify the location of the SuperExec's
      | TLS certificate file inside the container.
      |
      | The ``certificates/server.pem`` file is used to identify the SuperExec and to encrypt the
      | data that is transmitted over the network.
    * | ``--ssl-keyfile certificates/server.key``: Specify the location of the SuperExec's
      | TLS private key file inside the container.
      |
      | The ``certificates/server.key`` file is used to decrypt the data that is transmitted over
      | the network.
    * | ``--executor-config root-certificates=\"certificates/superlink_ca.crt\"``: Specify the
      | location of the CA certificate file inside the container that the SuperExec executor
      | should use to verify the SuperLink's identity.
