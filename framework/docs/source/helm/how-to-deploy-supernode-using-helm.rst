:og:description: Deploy Flower's SuperNode Helm chart to install client federated learning components. Default config mirrors official releases for seamless integration.
.. meta::
    :description: Deploy Flower's SuperNode Helm chart to install client federated learning components. Default config mirrors official releases for seamless integration.

Deploy SuperNode using Helm
===========================

.. note::

    Flower Helm charts are a Flower Enterprise feature. See `Flower Enterprise
    <https://flower.ai/enterprise>`_ for details.

The Flower Framework offers a unified approach to federated learning, analytics, and
evaluation, allowing you to federate any workload, machine learning framework, or
programming language.

This Helm chart installs the client-side components of the Flower Framework,
specifically setting up the SuperNode.

The default installation configuration aims to replicate the functionality and setup of
the provided Flower Framework releases.

Multi Project Setup
-------------------

To install multiple types of SuperNodes, such as a federation for running PyTorch and
another for TensorFlow, you need to install the Helm Chart multiple times with different
names. This allows each deployment to have its own configurations and dependencies.

For instance, you can install the Chart for the PyTorch setup by adjusting the
values.yaml file as shown below:

.. code-block:: yaml

    supernode:
      superlink:
        address: my-superlink.example.com
        port: 9092
      node:
        config:
          partition-id: 0
          num-partitions: 2
      image:
        registry: myregistry.example.com
        repository: flwr/supernode
        tag: 1.19.0-pytorch

Install this configuration using the following command:

.. code-block:: sh

    $ helm install pytorch . --values values.yaml

This will deploy 10 SuperNodes named ``pytorch-flower-client-supernode-<random>``.

For a TensorFlow setup, modify the values.yaml file as follows:

.. code-block:: yaml

    supernode:
      replicas: 3
      superlink:
        address: my-other-superlink.example.com
        port: 9092
      node:
        config:
          partition-id: 1
          num-partitions: 2
      image:
        registry: myregistry.example.com
        repository: flwr/supernode
        tag: 1.19.0-tensorflow

Install this configuration using the following command:

.. code-block:: sh

    $ helm install tensorflow . --values values.yaml

This will deploy 3 SuperNodes named ``tensorflow-flower-client-supernode-<random>``.

Deploy Flower Framework with TLS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure TLS communication within the Flower framework, you need to configure your
deployment with proper TLS certificates.

**Note:** If ``global.insecure`` is set to ``False``, you must pre-provide a secret of
type ``kubernetes.io/tls`` named ``flower-client-tls``.

Example configuration for TLS deployment:

.. code-block:: yaml

    global:
      insecure: false

Deploy Flower Framework without TLS
-----------------------------------

For testing or internal use, you might want to deploy Flower without TLS. Be cautious as
this exposes your deployment to potential security risks.

Example configuration for insecure deployment:

.. code-block:: yaml

    global:
      insecure: true

Node Authentication
-------------------

To enable Node Authentication, you need to specify a private key in either PKCS8 or
OpenSSH (PEM-like) format. This example assumes that the SuperLink is also configured
for Node Authentication and recognizes the ``ecdsa-sha2-nistp384 [...]`` public key of
this SuperNode.

.. code-block:: yaml

    global:
      insecure: false
      [...]
      nodeAuth:
        enabled: true
        authSupernodePrivateKey: |+
          -----BEGIN OPENSSH PRIVATE KEY-----
          [...]
          -----END OPENSSH PRIVATE KEY-----
        authSupernodePublicKey: ecdsa-sha2-nistp384 [...]
    supernode:
      enabled: true
      superlink:
        address: my-superlink.example.com
        port: 9092
    clientapp:
      enabled: true
      supernode:
        address: my-supernode.example.com
        port: 443

Isolated Setup
--------------

Isolation All-in-One
~~~~~~~~~~~~~~~~~~~~

To install SuperNode in isolation mode using the “process” configuration, both the
ClientApp and SuperNode need to be enabled. By default, the ClientApp connects to the
SuperNode internally within the cluster, so there is no need to set
``supernode.address`` and ``supernode.port`` unless the connection is external. This
setup assumes that both components are running within the same cluster.

.. code-block:: yaml

    [...]
    supernode:
      enabled: true
      [...]
      isolationMode: process
    [...]
    clientapp:
      enabled: true
    [...]

Isolation Distributed
~~~~~~~~~~~~~~~~~~~~~

You can also deploy the SuperNode and ClientApp separately. To do this, you need to
deploy the chart twice: once with ``supernode.enabled=true`` and once with
``clientapp.enabled=true``. To allow the ClientApp to connect to the SuperNode in this
configuration, enable the SuperNode ingress by setting
``supernode.ingress.enabled=true``. This setup is intended for scenarios where the
components run on different clusters or a hybrid environment involving Kubernetes and
ClientApp native installations.

.. code-block:: yaml

    [...]
    supernode:
      enabled: true
      ingress:
        enabled: true
        [...]
    [...]
    clientapp:
      enabled: true
      supernode:
        address: my-supernode.example.com
        port: 443
    [...]

Node Configuration
------------------

You can add a node configuration to configure a SuperNode. The YAML datatype is
preserved when passing it in the Python application:

.. code-block:: yaml

    supernode:
      node:
        config:
          bool: false
          int: 1
          negative_int: -1
          float: 21.23
          negative_float: -1.34
          string: value 1
          int-as-string: "1"
