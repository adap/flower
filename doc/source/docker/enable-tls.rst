Enable TLS for Secure Connections
=================================

When operating in a production environment, it is strongly recommended to enable
Transport Layer Security (TLS) for each Flower component to ensure secure communication.

.. note::

    For testing purposes, you can generate your own self-signed certificates. The
    `Enable SSL connections
    <https://flower.ai/docs/framework/how-to-enable-ssl-connections.html#certificates>`__
    page contains a section that will guide you through the process.

.. note::

    Because Flower containers, by default, run with a non-root user ``app``, the mounted
    files and directories must have the proper permissions for the user ID ``49999``.

    For example, to change the user ID of all files in the ``certificates/`` directory,
    you can run ``sudo chown -R 49999:49999 certificates/*``.

    If you later want to delete the directory, you can change the user ID back to the
    current user ID by running ``sudo chown -R $USER:$(id -gn) certificates``.

.. tab-set::

    .. tab-item:: Isolation Mode ``subprocess``

        By default, the ServerApp is executed as a subprocess within the SuperLink Docker
        container, and the ClientApp is run as a subprocess within the SuperNode Docker
        container. You can learn more about the different process modes here:
        :doc:`run-as-subprocess`.

        To enable TLS between the SuperLink and SuperNode, as well as between the SuperLink and the ``flwr``
        CLI, you will need two sets of PEM-encoded root certificates, private keys, and certificate chains.

        **SuperLink**

        Assuming all files we need are in the local ``superlink-certificates`` and
        ``supernode-certificates`` directories, we can use the flag ``--volume`` to mount the
        local directories into the SuperNode container:

        .. code-block:: bash

            $ docker run --rm \
                --volume ./superlink-certificates/:/app/certificates/:ro \
                <superlink-image> \
                --ssl-ca-certfile certificates/ca.crt \
                --ssl-certfile certificates/server.pem \
                --ssl-keyfile certificates/server.key \
                <additional-args>

        .. dropdown:: Understanding the command

            * ``docker run``: This tells Docker to run a container from an image.
            * ``--rm``: Remove the container once it is stopped or the command exits.
            * | ``--volume ./superlink-certificates/:/app/certificates/:ro``: Mount the ``superlink-certificates``
              | directory in the current working directory of the host machine as a read-only volume
              | at the ``/app/certificates`` directory inside the container.
              |
              | This allows the container to access the TLS certificates that are stored in the certificates
              | directory.
            * ``<superlink-image>``: The name of your SuperLink image to be run.
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

        **SuperNode**

        .. note::

            If you're generating self-signed certificates and the ``ca.crt`` certificate or the
            ``supernode-certificates`` directory doesn't exist on the SuperNode, you can copy it over
            after the generation step.

        .. note::

            Each SuperNode can have its own set of keys and certificates, or they can all share
            the same set.

        .. code-block:: bash

            $ docker run --rm \
                --volume ./supernode-certificates/:/app/certificates/:ro \
                --volume ./superlink-certificates/ca.crt:/app/ca.crt/:ro \
                <supernode-image> \
                --ssl-ca-certfile certificates/ca.crt \
                --ssl-certfile certificates/server.pem \
                --ssl-keyfile certificates/server.key \
                --root-certificates ca.crt \
                <additional-args>

        .. dropdown:: Understanding the command

            * ``docker run``: This tells Docker to run a container from an image.
            * ``--rm``: Remove the container once it is stopped or the command exits.
            * | ``--volume ./supernode-certificates/:/app/certificates/:ro``: Mount the ``supernode-certificates``
              | directory in the current working directory of the host machine as a read-only volume at the
              | ``/app/certificates`` directory inside the container.
              |
              | This allows the container to access the TLS certificates that are stored in the certificates
              | directory.
            * | ``--volume ./superlink-certificates/ca.crt:/app/ca.crt/:ro``: Mount the ``ca.crt``
              | file from the ``superlink-certificates`` directory of the host machine as a read-only
              | volume at the ``/app/ca.crt`` directory inside the container.
            * ``<supernode-image>``: The name of your SuperNode image to be run.
            * | ``--ssl-ca-certfile certificates/ca.crt``: Specify the location of the CA certificate file
              | inside the container.
              |
              | The ``certificates/ca.crt`` file is a certificate that is used to verify the identity of the
              | SuperNode.
            * | ``--ssl-certfile certificates/server.pem``: Specify the location of the SuperNode's
              | TLS certificate file inside the container.
              |
              | The ``certificates/server.pem`` file is used to identify the SuperNode and to encrypt the
              | data that is transmitted over the network.
            * | ``--ssl-keyfile certificates/server.key``: Specify the location of the SuperNode's
              | TLS private key file inside the container.
              |
              | The ``certificates/server.key`` file is used to decrypt the data that is transmitted over
              | the network.
            * | ``--root-certificates ca.crt``: This specifies the location of the CA certificate file
              | inside the container.
              |
              | The ``ca.crt`` file is used to verify the identity of the SuperLink.

    .. tab-item:: Isolation Mode ``process``

        In isolation mode ``process``, the ServerApp and ClientApp run in their own processes.
        Unlike in isolation mode ``subprocess``, the SuperLink or SuperNode does not attempt to
        create the respective processes; instead, they must be created externally.

        It is possible to run only the SuperLink in isolation mode ``subprocess`` and the
        SuperNode in isolation mode ``process``, or vice versa, or even both with isolation mode
        ``process``.

        **SuperLink and ServerApp**

        To enable TLS between all Flower components, you will need two sets of PEM-encoded root
        certificates, private keys, and certificate chains.

        Assuming all files we need are in the local ``superlink-certificates`` and
        ``supernode-certificates`` directories, we can use the flag ``--volume`` to mount the
        local directories into the SuperNode container:

        Start the SuperLink container:

        .. code-block:: bash
            :substitutions:

            $ docker run --rm \
                --volume ./superlink-certificates/:/app/certificates/:ro \
                flwr/superlink:|stable_flwr_version| \
                --ssl-ca-certfile certificates/ca.crt \
                --ssl-certfile certificates/server.pem \
                --ssl-keyfile certificates/server.key \
                --isolation process \
                <additional-args>

        .. dropdown:: Understanding the command

            * ``docker run``: This tells Docker to run a container from an image.
            * ``--rm``: Remove the container once it is stopped or the command exits.
            * | ``--volume ./superlink-certificates/:/app/certificates/:ro``: Mount the
              | ``superlink-certificates`` directory in the current working directory of the host
              | machine as a read-only volume at the ``/app/certificates`` directory inside the container.
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
            * | ``--isolation process``: Tells the SuperLink that the ServerApp is created by separate
              | independent process. The SuperLink does not attempt to create it.

        Start the ServerApp container:

        .. code-block:: bash

            $ docker run --rm \
                --volume ./superlink-certificates/ca.crt:/app/ca.crt:ro \
                <serverapp-image> \
                --root-certificates ca.crt \
                <additional-args>

        .. dropdown:: Understand the command

            * ``docker run``: This tells Docker to run a container from an image.
            * ``--rm``: Remove the container once it is stopped or the command exits.
            * | ``--volume ./superlink-certificates/ca.crt:/app/ca.crt:ro``: Mount the ``ca.crt`` file from
              | the ``superlink-certificates`` directory of the host machine as a read-only volume at the
              | ``/app/ca.crt`` directory inside the container.
            * ``<serverapp-image>``: The name of your ServerApp image to be run.
            * | ``--root-certificates ca.crt``: This specifies the location of the CA
              | certificate file inside the container.
              |
              | The ``ca.crt`` file is used to verify the identity of the SuperLink.

        **SuperNode and ClientApp**

        .. note::

            If you're generating self-signed certificates and the ``ca.crt`` certificate or the
            ``supernode-certificates`` directory doesn't exist on the SuperNode, you can copy it over
            after the generation step.

        .. note::

            Each SuperNode can have its own set of keys and certificates, or they can all share
            the same set.

        Start the SuperNode container:

        .. code-block:: bash
            :substitutions:

            $ docker run --rm \
                --volume ./supernode-certificates/:/app/certificates/:ro \
                --volume ./superlink-certificates/ca.crt:/app/ca.crt/:ro \
                flwr/supernode:|stable_flwr_version| \
                --ssl-ca-certfile=certificates/ca.crt \
                --ssl-certfile=certificates/server.pem \
                --ssl-keyfile=certificates/server.key \
                --root-certificates ca.crt \
                --isolation process \
                <additional-args>

        .. dropdown:: Understanding the command

            * ``docker run``: This tells Docker to run a container from an image.
            * ``--rm``: Remove the container once it is stopped or the command exits.
            * | ``--volume ./supernode-certificates/:/app/certificates/:ro``: Mount the ``supernode-certificates``
              | directory in the current working directory of the host machine as a read-only volume at the
              | ``/app/certificates`` directory inside the container.
              |
              | This allows the container to access the TLS certificates that are stored in the certificates
              | directory.
            * | ``--volume ./superlink-certificates/ca.crt:/app/ca.crt/:ro``: Mount the ``ca.crt`` file from the
              | ``superlink-certificates`` directory of the host machine as a read-only volume at the ``/app/ca.crt``
              | directory inside the container.
            * | :substitution-code:`flwr/supernode:|stable_flwr_version|`: The name of the image to be run and the specific
              | tag of the image. The tag :substitution-code:`|stable_flwr_version|` represents a specific version of the image.
            * | ``--ssl-ca-certfile certificates/ca.crt``: Specify the location of the CA certificate file
              | inside the container.
              |
              | The ``certificates/ca.crt`` file is a certificate that is used to verify the identity of the
              | SuperNode.
            * | ``--ssl-certfile certificates/server.pem``: Specify the location of the SuperNode's
              | TLS certificate file inside the container.
              |
              | The ``certificates/server.pem`` file is used to identify the SuperNode and to encrypt the
              | data that is transmitted over the network.
            * | ``--ssl-keyfile certificates/server.key``: Specify the location of the SuperNode's
              | TLS private key file inside the container.
              |
              | The ``certificates/server.key`` file is used to decrypt the data that is transmitted over
              | the network.
            * | ``--root-certificates ca.crt``: This specifies the location of the CA certificate file
              | inside the container.
              |
              | The ``ca.crt`` file is used to verify the identity of the SuperLink.
            * | ``--isolation process``: Tells the SuperNode that the ClientApp is created by separate
              | independent process. The SuperNode does not attempt to create it.

        Start the ClientApp container:

        .. code-block:: bash

            $ docker run --rm \
                --volume ./supernode-certificates/ca.crt:/app/ca.crt:ro \
                <clientapp-image> \
                --root-certificates ca.crt \
                <additional-args>

        .. dropdown:: Understand the command

            * ``docker run``: This tells Docker to run a container from an image.
            * ``--rm``: Remove the container once it is stopped or the command exits.
            * | ``--volume ./supernode-certificates/ca.crt:/app/ca.crt:ro``: Mount the ``ca.crt`` file from
              | the ``supernode-certificates`` directory of the host machine as a read-only volume at the
              | ``/app/ca.crt`` directory inside the container.
            * ``<clientapp-image>``: The name of your ClientApp image to be run.
            * | ``--root-certificates ca.crt``: This specifies the location of the CA
              | certificate file inside the container.
              |
              | The ``ca.crt`` file is used to verify the identity of the SuperNode.

Append the following lines to the end of the ``pyproject.toml`` file and save it:

.. code-block:: toml
    :caption: pyproject.toml

    [tool.flwr.federations.local-deployment-tls]
    address = "127.0.0.1:9093"
    root-certificates = "../superlink-certificates/ca.crt"

The path of the ``root-certificates`` should be relative to the location of the
``pyproject.toml`` file.

.. note::

    You can customize the string that follows ``tool.flwr.federations.`` to fit your
    needs. However, please note that the string cannot contain a dot (``.``).

    In this example, ``local-deployment-tls`` has been used. Just remember to replace
    ``local-deployment-tls`` with your chosen name in both the
    ``tool.flwr.federations.`` string and the corresponding ``flwr run .`` command.
