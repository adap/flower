---
myst:
  html_meta:
    description: Deploy Flower's SuperLink Helm chart to set up federated 
      learning servers. Default config mirrors official releases, enabling 
      seamless deployment, evaluation.
    property:og:description: Deploy Flower's SuperLink Helm chart to set up 
      federated learning servers. Default config mirrors official releases, 
      enabling seamless deployment, evaluation.
---

# Deploy SuperLink using Helm

```{note}
Flower Helm charts are a Flower Enterprise feature. See [Flower Enterprise](https://flower.ai/enterprise) for details.
```

The Flower Framework offers a unified approach to federated learning, analytics, and evaluation,
allowing you to federate any workload, machine learning framework, or programming language.

This Helm chart installs the server-side components of the Flower Framework, specifically
setting up the SuperLink.

The default installation configuration aims to replicate the functionality and setup of the
provided Flower Framework releases.

## Disable the SuperLink component

```yaml
superlink:
  name: superlink
  enabled: false
```

## Enable the ServerApp component

```yaml
serverapp:
  name: serverapp
  enabled: true
```

## Run simulations in Kubernetes using the Simulation Plugin

For more details, visit: [Run simulations](../how-to-run-simulations.rst#run-simulations) guide.

```yaml
superlink:
  enabled: true
  executor:
    plugin: flwr.superexec.simulation:executor
    config:
      num-supernodes: 2
      [...]
```

## Change Log Verbosity Level

The log verbosity level in Flower can be adjusted using the `FLWR_LOG_LEVEL` environment variable.
This helps control the level of detail included in logs, making debugging and monitoring easier.

### Setting Global Log Level

To enable detailed logging (e.g., `DEBUG` level) for all services, add `FLWR_LOG_LEVEL` to the `env`
parameter under the `global` section in your `values.yml` file:

```yaml
global:
  env:
    - name: FLWR_LOG_LEVEL
      value: DEBUG
```

### Setting Log Level for a Specific Service

If you want to enable logging only for a specific service (e.g., `superlink`), you can specify it
under the respective service section:

```yaml
superlink:
  env:
    - name: FLWR_LOG_LEVEL
      value: DEBUG
```

For more details on logging configuration, visit:
[Flower Logging Documentation](../how-to-configure-logging.rst)

## License Key

Starting from `1.20.0`, the SuperLink service must be started with
a valid license key.

You can configure the license key in the `global.license` section of your `values.yml` file in one
of two ways:

1. Directly — by setting `global.license.key` to your license key.
2. From an existing Kubernetes Secret — by setting `global.license.existingSecret` to the name of
   a secret that contains your key.

### Example: Setting the License Key Directly

```yaml
global:
  license:
    enabled: true
    key: <YOUR_FLWR_LICENSE_KEY>
    existingSecret: ""
```

In this configuration, the Helm chart will automatically create a Kubernetes Secret and mount it
into the SuperLink container.

### Example: Using an Existing Secret

```yaml
global:
  license:
    enabled: true
    key: ""
    existingSecret: "existing-license-key-secret"
```

If both `key` and `existingSecret` are set, `existingSecret` takes precedence and the `key` value
will be ignored.

Note that the existing secret must contain the key `FLWR_LICENSE_KEY`:

```yaml
kind: Secret
stringData:
  FLWR_LICENSE_KEY: <YOUR_FLWR_LICENSE_KEY>
```

## Enable User Authentication

User authentication can be enabled if you're using the Flower Enterprise Edition (EE) Docker images.
This is configured in the `global.userAuth` section of your `values.yml` file.

### Example: Enabling OpenID Connect (OIDC) Authentication

```yaml
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
```

Explanation of Parameters:

- `auth_type`: The authentication mechanism being used (e.g., oidc).
- `auth_url`: The OpenID Connect authentication endpoint where users authenticate.
- `token_url`: The URL for retrieving access tokens.
- `validate_url`: The endpoint for validating user authentication.
- `oidc_client_id`: The client ID issued by the authentication provider.
- `oidc_client_secret`: The secret key associated with the client ID.

### Use an Existing Secret

To use an existing secret that contains the user authentication configuration, set `existingSecret`
to the name of the existing secret:

```yaml
global:
  userAuth:
    enabled: true
    config: {}
    existingSecret: "existing-user-auth-config"
```

Note that the existing secret must contain the key `user-auth-config.yml`:

```yaml
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
```

### Configuring OpenFGA

The flower-server chat component supports OpenFGA as a fine-grained authorization service,
but it is disabled by default.

To enable OpenFGA change the following value in your `values.yml` file:

```yaml
openfga:
  enabled: true
```

By default, OpenFGA will run with an in-memory store, which is non-persistent and suitable
only for testing or development.

OpenFGA supports persistent storage using PostgreSQL or MySQL:

- To deploy OpenFGA with a new PostgreSQL/MySQL instance, enable the bundled chart configuration.
- To connect to an existing database, provide the appropriate connection details via Helm values
  (e.g., `openfga.datastore.uri`).

For more information visit the official [OpenFGA Helm Chart Documentation](https://artifacthub.io/packages/helm/openfga/openfga/0.2.30).

The following commands set up a store, authorization model, and inserts users (using tuples) into OpenFGA. Run these once the OpenFGA instance is deployed.

Setup the authorization model and tuples:

:::{dropdown} Authorization model file `model.fga`

```text
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
```

:::

:::{dropdown} User permissions file `tuples.fga`

```yaml
- user: flwr_aid:<OIDC_SUB_1>
  relation: has_access
  object: service:<your_grid_name>
- user: flwr_aid:<OIDC_SUB_2>
  relation: has_access
  object: service:<your_grid_name>
```

:::

Create store:

```shell
OPENFGA_URL="<OPENFGA_URL>"
OPENFGA_STORE_NAME="<OPENFGA_STORE_NAME>"
docker run --rm -v "$(pwd)":/app -w /app openfga/cli \
  --api-url ${OPENFGA_URL} store create \
  --name ${OPENFGA_STORE_NAME}
```

The response will include an `id` field, which is the OpenFGA store ID associated with the `OPENFGA_STORE_NAME` that was created.

Get store ID (alternative way):

```shell
docker run --rm -v "$(pwd)":/app -w /app openfga/cli \
  --api-url ${OPENFGA_URL} store list
```

Set OpenFGA store ID from previous step and write model:

```shell
OPENFGA_STORE_ID="<STORE_ID_FROM_EARLIER_STEP>"
docker run --rm -v "$(pwd)":/app -w /app openfga/cli \
  --api-url ${OPENFGA_URL} model write \
  --store-id ${OPENFGA_STORE_ID} \
  --file model.fga
```

Set OpenFGA model ID from previous step and write tuples:

```shell
OPENFGA_MODEL_ID="<MODEL_ID_FROM_EARLIER_STEP>"
docker run --rm -v "$(pwd)":/app -w /app openfga/cli \
  --api-url ${OPENFGA_URL} tuple write \
  --store-id ${OPENFGA_STORE_ID} \
  --model-id ${OPENFGA_MODEL_ID} \
  --file tuples.yaml
```

Add a new `authorization` section under your existing `global.userAuth` configuration or directly within your existing secret, depending on your setup. Set the `OPENFGA_STORE_ID` and `OPENFGA_MODEL_ID` from the previous steps in the file:

```yaml
authorization:
  authz_type: openfga
  authz_url: <OPENFGA_URL>
  store_id: <OPENFGA_STORE_ID>
  model_id: <OPENFGA_MODEL_ID>
  relation: has_access
  object: service:<your_grid_name>
```

## Change Isolation Mode

The isolation mode determines how the SuperLink manages the ServerApp process execution.
This setting can be adjusted using the `superlink.isolationMode` parameter:

**Example: Changing Isolation Mode**

```yaml
superlink:
  isolationMode: process

# Don’t forget to enable the serverapp if you don’t
# plan to use an existing one.
serverapp:
  enabled: true
```

## Deploy Flower Framework with TLS

To ensure TLS communication within the Flower Framework, you need to configure your deployment
with proper TLS certificates.

```yaml
global:
  insecure: false
superlink:
  enabled: true
```

### Override certificate paths

By default, the TLS-related flags use the following paths when TLS is enabled:

`--ssl-ca-certfile`: `/app/cert/ca.crt`,
`--ssl-certfile`: `/app/cert/tls.crt`,
`--ssl-keyfile`: `/app/cert/tls.key`.

These paths can be overridden by specifying the flags in the extraArgs, as shown below.

```yaml
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
```

## Deploy Flower Framework without TLS

To deploy the Flower Framework simply, you need to configure your deployment as insecure.

```yaml
global:
  insecure: true
superlink:
  enabled: true
```

## Pre-provide TLS Certificate

If certificate creation is disabled, you must provide a pre-existing secret of type
`kubernetes.io/tls` named `<flower-server.fullname>-server-tls`.

```yaml
certificate:
  enabled: false
```

## Ingress Configuration

### SSL-Passthrough

When the `tls` option is set to `true`, it expects the existence of the
`<flower-server.fullname>-server-tls` secret. Flower Framework components will load TLS
certificates on startup.

```yaml
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
      hostname: exec-api.example.com
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
```

**Pre-Provide TLS Certificate with Additional Hosts**

In this example, we use `cert-manager` to create a certificate. By default, the certificate will
only include the DNS name specified in `common-name`.

In some cases, the server and client charts are deployed in the same cluster, while the exec
API is accessible via the internet.

To allow SuperNodes to connect to the SuperLink via the internal service URL,
you need to add an additional host, as shown below:

```yaml
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
```

## Enable Node Authentication

```yaml
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
```

Public keys can include comments at the end of the key data:

```yaml
global:
  nodeAuth:
    authListPublicKeys:
      - ecdsa-sha2-nistp384 [...] comment with spaces
```
