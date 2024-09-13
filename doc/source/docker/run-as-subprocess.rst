Run ClientApp as a Subprocess
=============================

In this mode, the ClientApp is executed as a subprocess within the SuperNode Docker container,
rather than running in a separate container. This approach reduces the number of running containers,
which can be beneficial for environments with limited resources. However, it also means that the
ClientApp is no longer isolated from the SuperNode, which may introduce additional security
concerns.

Prerequisites
-------------

#. Before running the ClientApp as a subprocess, ensure that the FAB dependencies have been installed
   in the SuperNode images. This can be done by extending the SuperNode image:

   .. code-block:: dockerfile
      :caption: Dockerfile.supernode
      :linenos:
      :substitutions:

      FROM flwr/supernode:|stable_flwr_version|

      WORKDIR /app
      COPY pyproject.toml .
      RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
          && python -m pip install -U --no-cache-dir .

      ENTRYPOINT ["flower-supernode"]

#. Next, build the SuperNode Docker image by running the following command in the directory where
   Dockerfile is located:

   .. code-block:: shell

      $ docker build -f Dockerfile.supernode -t flwr_supernode:0.0.1 .


Run the ClientApp as a Subprocess
---------------------------------

Start the SuperNode with the flag ``--isolation subprocess``, which tells the SuperNode to execute
the ClientApp as a subprocess:

.. code-block:: shell

   $ docker run --rm \
       --detach \
       flwr_supernode:0.0.1 \
       --insecure \
       --superlink superlink:9092 \
       --node-config "partition-id=1 num-partitions=2" \
       --supernode-address localhost:9094 \
       --isolation subprocess
