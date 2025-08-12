:og:description: Deploy Flower's SuperLink Helm chart to set up federated learning servers. Default config mirrors official releases, enabling seamless deployment, evaluation.
.. meta::
    :description: Deploy Flower's SuperLink Helm chart to set up federated learning servers. Default config mirrors official releases, enabling seamless deployment, evaluation.

Deploy SuperLink using Helm
===========================

.. note::

    Flower Helm charts are a Flower Enterprise feature. See `Flower Enterprise
    <https://flower.ai/enterprise>`_ for details.

The Flower Framework offers a unified approach to federated learning, analytics, and
evaluation, allowing you to federate any workload, machine learning framework, or
programming language.

This Helm chart installs the server-side components of the Flower Framework,
specifically setting up the SuperLink.

The default installation configuration aims to replicate the functionality and setup of
the provided Flower Framework releases.

Disable the SuperLink component
-------------------------------

.. code-block:: yaml

    superlink:
      name: superlink
      enabled: false

Enable the ServerApp component
------------------------------

.. code-block:: yaml

    serverapp:
      name: serverapp
      enabled: true

Run simulations in Kubernetes using the Simulation Plugin
---------------------------------------------------------

For more details, visit: `Run simulations
<https://flower.ai/docs/framework/how-to-run-simulations.html#run-simulations>`__ guide

.. code-block:: yaml

    superlink:
      enabled: true
      executor:
        plugin: flwr.superexec.simulation:executor
        config:
          num-supernodes: 2
          [...]

Change Log Verbosity Level
--------------------------

The log verbosity level in Flower can be adjusted using the ``FLWR_LOG_LEVEL``
environment variable. This helps control the level of detail included in logs, making
debugging and monitoring easier.

**Setting Global Log Level**

To enable detailed logging (e.g., ``DEBUG`` level) for all services, add
``FLWR_LOG_LEVEL`` to the ``env`` parameter under the ``global`` section in your
``values.yml`` file:

.. code-block:: yaml

    global:
      env:
        - name: FLWR_LOG_LEVEL
          value: DEBUG

**Setting Log Level for a Specific Service**

If you want to enable logging only for a specific service (e.g., ``superlink``), you can
specify it under the respective service section:

.. code-block:: yaml

    superlink:
      env:
        - name: FLWR_LOG_LEVEL
          value: DEBUG

For more details on logging configuration, visit: `Flower Logging Documentation
<https://flower.ai/docs/framework/how-to-configure-logging.html>`__

Enable User Authentication
--------------------------

User authentication can be enabled if you’re using the Flower Enterprise Edition (EE)
Docker images. This is configured in the ``global.userAuth`` section of your
``values.yml`` file.

**Example: Enabling OpenID Connect (OIDC) Authentication**

.. code-block:: yaml

    global:
      userAuth:
        enabled: true
        config:
          authentication:
            auth_type: oidc
            auth_url: https://<domain>/auth/device
            token_url: https://<domain>/token
            validate_url: https://<domain>/userinfo
            oidc_client_id: <client_id>
            oidc_client_secret: <client_secret>

Explanation of Parameters:

- ``auth_type``: The authentication mechanism being used (e.g., OIDC).
- ``auth_url``: The OpenID Connect authentication endpoint where users authenticate.
- ``token_url``: The URL for retrieving access tokens.
- ``validate_url``: The endpoint for validating user authentication.
- ``oidc_client_id``: The client ID issued by the authentication provider.
- ``oidc_client_secret``: The secret key associated with the client ID.

**Use an Existing Secret**

To use an existing secret that contains the user authentication configuration, set
``existingSecret`` to the name of the existing secret:

.. code-block:: yaml

    global:
      userAuth:
        enabled: true
        config: {}
        existingSecret: "existing-user-auth-config"

Note that the existing secret must contain the key ``user-auth-config.yml``:

.. code-block:: yaml

    kind: Secret
    stringData:
      user-auth-config.yml: |
        authentication:
          auth_type: oidc
          auth_url: https://<domain>/auth/device
          token_url: https://<domain>/token
          validate_url: https://<domain>/userinfo
          oidc_client_id: <client_id>
          oidc_client_secret: <client_secret>

**Configuring OpenFGA**

The chart component supports OpenFGA as a fine-grained authorization service, but it is
disabled by default.

To enable OpenFGA change the following value in your ``values.yml`` file:

.. code-block:: yaml

    openfga:
      enabled: true

By default, OpenFGA will run with an in-memory store, which is non-persistent and
suitable only for testing or development.

OpenFGA supports persistent storage using PostgreSQL or MySQL:

- To deploy OpenFGA with a new PostgreSQL/MySQL instance, enable the bundled chart
  configuration.
- To connect to an existing database, provide the appropriate connection details via
  Helm values (e.g., ``openfga.datastore.uri``).

For more information visit the official `OpenFGA Helm Chart Documentation
<https://artifacthub.io/packages/helm/openfga/openfga/0.2.30>`__.

The following commands set up a store, authorization model, and inserts users (using
tuples) into OpenFGA. Run these once the OpenFGA instance is deployed.

Setup the authorization model and tuples:

.. dropdown:: Authorization model file ``model.fga``

    .. code-block:: yaml

        model
          # We are using the 1.1 schema with type restrictions
          schema 1.1

        # Define the 'flwr_aid' type to represent individual users in the system.
        type flwr_aid

        # Define the 'service' type to group users.
        type service
          relations
            # The 'has_access' relation defines users who have access to this service.
            define has_access: [flwr_aid]

.. dropdown:: User permissions file ``tuples.fga``

    .. code-block:: yaml

        - user: flwr_aid:<OIDC_SUB_1>
          relation: has_access
          object: service:<your_grid_name>
        - user: flwr_aid:<OIDC_SUB_2>
          relation: has_access
          object: service:<your_grid_name>

Create store:

.. code-block:: shell

    OPENFGA_URL="<OPENFGA_URL>"
    OPENFGA_STORE_NAME="<OPENFGA_STORE_NAME>"
    docker run --rm -v "$(pwd)":/app -w /app openfga/cli \
      --api-url ${OPENFGA_URL} store create \
      --name ${OPENFGA_STORE_NAME}

The response will include an ``id`` field, which is the OpenFGA store ID associated with
the ``OPENFGA_STORE_NAME`` that was created.

Get store ID (alternative way):

.. code-block:: shell

    docker run --rm -v "$(pwd)":/app -w /app openfga/cli \
      --api-url ${OPENFGA_URL} store list

Set OpenFGA store ID from previous step and write model:

.. code-block:: shell

    OPENFGA_STORE_ID="<STORE_ID_FROM_EARLIER_STEP>"
    docker run --rm -v "$(pwd)":/app -w /app openfga/cli \
      --api-url ${OPENFGA_URL} model write \
      --store-id ${OPENFGA_STORE_ID} \
      --file model.fga

Set OpenFGA model ID from previous step and write tuples:

.. code-block:: shell

    OPENFGA_MODEL_ID="<MODEL_ID_FROM_EARLIER_STEP>"
    docker run --rm -v "$(pwd)":/app -w /app openfga/cli \
      --api-url ${OPENFGA_URL} tuple write \
      --store-id ${OPENFGA_STORE_ID} \
      --model-id ${OPENFGA_MODEL_ID} \
      --file tuples.yaml

Add a new ``authorization`` section under your existing ``global.userAuth``
configuration or directly within your existing secret, depending on your setup. Set the
``OPENFGA_STORE_ID`` and ``OPENFGA_MODEL_ID`` from the previous steps in the file:

.. code-block:: yaml

    authorization:
      authz_type: openfga
      authz_url: <OPENFGA_URL>
      store_id: <OPENFGA_STORE_ID>
      model_id: <OPENFGA_MODEL_ID>
      relation: has_access
      object: service:<your_grid_name>

Change Isolation Mode
---------------------

The isolation mode determines how the SuperLink manages the ServerApp process execution.
This setting can be adjusted using the ``superlink.isolationMode`` parameter:

**Example: Changing Isolation Mode**

.. code-block:: yaml

    superlink:
      isolationMode: process

    # Don’t forget to enable the serverapp if you don’t
    # plan to use an existing one.
    serverapp:
      enabled: true

Deploy Flower Framework with TLS
--------------------------------

To ensure TLS communication within the Flower Framework, you need to configure your
deployment with proper TLS certificates.

.. code-block:: yaml

    global:
      insecure: false
    superlink:
      enabled: true

Override certificate paths
~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the TLS-related flags use the following paths when TLS is enabled:

``--ssl-ca-certfile``: ``/app/cert/ca.crt``, ``--ssl-certfile``: ``/app/cert/tls.crt``,
``--ssl-keyfile``: ``/app/cert/tls.key``.

These paths can be overridden by specifying the flags in the extraArgs, as shown below.

.. code-block:: yaml

    global:
      insecure: false
    superlink:
      enabled: true
      extraArgs:
        - --ssl-ca-certfile
        - /mount/cert/ca.cert
        - --ssl-certfile
        - /mount/cert/tls.cert
        - --ssl-keyfile
        - /mount/cert/tls.key

Deploy Flower Framework without TLS
-----------------------------------

For testing or internal use, you might want to deploy Flower without TLS. Be cautious as
this exposes your deployment to potential security risks.

Example configuration for insecure deployment:

.. code-block:: yaml

    global:
      insecure: true
    superlink:
      enabled: true

Pre-provide TLS Certificate
---------------------------

If certificate creation is disabled, you must provide a pre-existing secret of type
``kubernetes.io/tls`` named ``<flower-server.fullname>-server-tls``.

.. code-block:: yaml

    certificate:
      enabled: false

Ingress Configuration
---------------------

SSL-Passthrough
~~~~~~~~~~~~~~~

When the ``tls`` option is set to ``true``, it expects the existence of the
``<flower-server.fullname>-server-tls`` secret. Flower Framework components will load
TLS certificates on startup.

.. code-block:: yaml

    superlink:
      enabled: true
      ingress:
        annotations:
          nginx.ingress.kubernetes.io/backend-protocol: GRPCS
          nginx.ingress.kubernetes.io/force-ssl-redirect: "false"
          nginx.ingress.kubernetes.io/ssl-passthrough: "false"
          nginx.ingress.kubernetes.io/ssl-redirect: "false"
        ingressClassName: nginx
        tls: true
        api:
          enabled: true
          hostname: control-api.example.com
          path: /
          pathType: ImplementationSpecific
        fleet:
          enabled: true
          hostname: fleet.example.com
          path: /
          pathType: ImplementationSpecific
        driver:
          enabled: true
          hostname: driver.example.com
          annotations:
            nginx.ingress.kubernetes.io/backend-protocol: GRPCS
            nginx.ingress.kubernetes.io/force-ssl-redirect: "false"
            nginx.ingress.kubernetes.io/ssl-passthrough: "false"
            nginx.ingress.kubernetes.io/ssl-redirect: "false"
          path: /
          pathType: ImplementationSpecific

**Pre-Provide TLS Certificate with Additional Hosts**

In this example, we use ``cert-manager`` to create a certificate. By default, the
certificate will only include the DNS name specified in ``common-name``.

In some cases, the server and client charts are deployed in the same cluster, while the
Control API is accessible via the internet.

To allow SuperNodes to connect to the SuperLink via the internal service URL, you need
to add an additional host, as shown below:

.. code-block:: yaml

    certificate:
      enabled: false
    superlink:
     ingress:
        enabled: true
        tls: true
        annotations:
          nginx.ingress.kubernetes.io/backend-protocol: GRPCS
          nginx.ingress.kubernetes.io/force-ssl-redirect: "false"
          nginx.ingress.kubernetes.io/ssl-passthrough: "false"
          nginx.ingress.kubernetes.io/ssl-redirect: "false"
          cert-manager.io/cluster-issuer: cert-manager-selfsigned
          cert-manager.io/common-name: api.example.com
        api:
          enabled: true
          hostname: api.example.com
        extraHosts:
          - name: <superlink_name>.<namespace>.svc.cluster.local
            pathType: ImplementationSpecific
            path: /
            port: 9092

Enable Node Authentication
--------------------------

.. code-block:: yaml

    global:
      insecure: false
      nodeAuth:
        enabled: true
        authListPublicKeys:
          - ecdsa-sha2-nistp384 [...]
          - ecdsa-sha2-nistp384 [...]
    superlink:
      enabled: true
      superlink:
        executor:
          plugin: flwr.superexec.deployment:executor
          config:
            root-certificates: '"/app/cert/ca.crt"'

Public keys can include comments at the end of the key data:

.. code-block:: yaml

    global:
      nodeAuth:
        authListPublicKeys:
          - ecdsa-sha2-nistp384 [...] comment with spaces
