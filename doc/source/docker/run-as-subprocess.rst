Run ClientApp or ServerApp as a Subprocess
==========================================

By default, the ServerApp is executed as a subprocess within the SuperLink Docker
container, and the ClientApp is run as a subprocess within the SuperNode Docker
container, rather than using separate containers as shown in the
:doc:`tutorial-quickstart-docker` guide.

This approach reduces the number of running containers, which can be beneficial for
environments with limited resources. However, it also means that the applications are
not isolated from their parent containers, which may introduce additional security
concerns.

.. tab-set::

    .. tab-item:: ClientApp

        **Prerequisites**

        1. Before running the ClientApp as a subprocess, ensure that the FAB dependencies have
        been installed in the SuperNode images. This can be done by extending the SuperNode
        image:

        .. code-block:: dockerfile
            :caption: supernode.Dockerfile
            :linenos:
            :substitutions:

            FROM flwr/supernode:|stable_flwr_version|

            WORKDIR /app
            COPY pyproject.toml .
            RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
                && python -m pip install -U --no-cache-dir .

            ENTRYPOINT ["flower-supernode"]

        2. Next, build the SuperNode Docker image by running the following command in the
        directory where Dockerfile is located:

        .. code-block:: shell

            $ docker build -f supernode.Dockerfile -t flwr_supernode:0.0.1 .

        **Run the ClientApp as a Subprocess**

        Start the SuperNode and run the ClientApp as a subprocess:

        .. code-block:: shell

            $ docker run --rm \
                --detach \
                flwr_supernode:0.0.1 \
                --insecure \
                --superlink <superlink-address>:9092 \
                --node-config <node-config>

    .. tab-item:: ServerApp

        **Prerequisites**

        1. Before running the ServerApp as a subprocess, ensure that the FAB dependencies have
        been installed in the SuperLink images. This can be done by extending the SuperLink
        image:

        .. code-block:: dockerfile
            :caption: superlink.Dockerfile
            :linenos:
            :substitutions:

            FROM flwr/superlink:|stable_flwr_version|

            WORKDIR /app
            COPY pyproject.toml .
            RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
                && python -m pip install -U --no-cache-dir .

            ENTRYPOINT ["flower-superlink"]

        2. Next, build the superlink Docker image by running the following command in the
        directory where Dockerfile is located:

        .. code-block:: shell

            $ docker build -f superlink.Dockerfile -t flwr_superlink:0.0.1 .

        **Run the ServerApp as a Subprocess**

        Start the SuperLink and run the ServerApp as a subprocess:

        .. code-block:: shell

            $ docker run --rm \
                -p 9091:9091 -p 9092:9092 -p 9093:9093 \
                --detach \
                flwr_superlink:0.0.1 \
                --insecure
