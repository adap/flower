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

## Configure Audit Logging

Audit logging provides JSON-formatted log output for Flower, allowing you to capture and store
events that occur in the SuperLink. This feature is useful for monitoring, debugging, or
maintaining a historical record of important actions.

To enable audit logging, add the following flag to the SuperLink configuration:

```yaml
superlink:
  enabled: true
  extraArgs:
    - --enable-event-log
```

For more details on audit logging configuration, visit:
[Flower Audit Logging Documentation](../how-to-configure-audit-logging.rst)

## License Key

Starting from `1.20.0`, the SuperLink service must be started with
a valid license key.

You can configure the license key in the `global.license` section of your `values.yml` file in one
of two ways:

1. Directly — by setting `global.license.key` to your license key.
2. From an existing Kubernetes Secret — by setting `global.license.existingSecret` to the name of
   a secret that contains your key.

```{note}
The SuperLink must be able to connect to `https://api.flower.ai` in order to validate the license.
```

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

```{note}
By default, the `existingSecret` resource must contain a key named `FLWR_LICENSE_KEY`.
```

```yaml
kind: Secret
stringData:
  FLWR_LICENSE_KEY: <YOUR_FLWR_LICENSE_KEY>
```

### Example: Setting a Custom Secret Key

If you prefer to use a different key name instead of the default `FLWR_LICENSE_KEY`,
you can override it by setting `secretKey` to your desired name:

```yaml
global:
  license:
    enabled: true
    key: ""
    secretKey: "license-key"
    existingSecret: "existing-license-key-secret"
```

```yaml
kind: Secret
stringData:
  license-key: <YOUR_FLWR_LICENSE_KEY>
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

## Deploy Flower Framework without TLS

By default, the Flower Framework is deployed without TLS. This means `global.insecure` is
set to `true`.

In this mode, the chart does not create any `cert-manager` `Issuer` or `Certificate` resources,
and any TLS-related configuration on the `Ingress` resource is ignored.

```yaml
global:
  insecure: true
```

## Deploy Flower Framework with TLS

If `cert-manager` is installed in your Kubernetes cluster, this chart can be used to set up TLS in
several ways. By default, it creates both a self-signed `Issuer` and a `Certificate`, but you can
also choose to create only an `Issuer` or create a `Certificate` from an existing
`Issuer`/`ClusterIssuer`.

Regardless of the option you choose, TLS is only enabled when you set `global.insecure: false`.

### Default Behavior (self-signed Issuer + Certificate)

```yaml
global:
  insecure: false
  domain: example.com

tls:
  issuer:
    enabled: true
  certificate:
    enabled: true
```

When deployed with these values, the chart creates a self-signed `Issuer` and then uses it to
issue a TLS certificate. The generated `Certificate` is valid for five years,
is renewed fifteen days before expiry, and is stored in a Kubernetes `Secret`.

If you leave `secretName` empty, the secret name defaults to `<chart-name>-server-tls`.
The certificate includes `global.domain` as its common name, along with service DNS entries and
any `additionalHosts` you provide.

This setup is best suited for testing and validating the Helm charts in a non-production
environment.

You can switch from self-signed to another issuer type by providing a `spec` block
for the `Issuer`.

### Create a Certificate using an Existing Issuer / ClusterIssuer

Use this if you already have an `Issuer` or `ClusterIssuer` (e.g., Let’s Encrypt or a corporate CA)
and you want the chart to create the `Certificate`.

```yaml
global:
  insecure: false
  domain: example.com

tls:
  issuer:
    enabled: false
  certificate:
    enabled: true
    existingIssuer: letsencrypt-clusterissuer
    existingIssuerKind: ClusterIssuer
```

With this configuration, the chart does not create a new `Issuer`. Instead, it references the
specified `ClusterIssuer` named `letsencrypt-clusterissuer` and requests a `Certificate` from it.
It is stored in a Kubernetes `Secret` that, by default, is named `<chart-name>-server-tls`.

To change the secret name, set `tls.certificate.secretName` to the desired name:

```yaml
tls:
  certificate:
    secretName: my-custom-tls-secret-name
```

### Create only an Issuer

In this mode, the chart creates an `Issuer` but does not create a `Certificate`.
This is useful if you want `cert-manager` to automatically generate certificates
based on `Ingress` annotations.

```yaml
global:
  insecure: false

tls:
  issuer:
    enabled: true
  certificate:
    enabled: false
  existingSecret: ""
```

See the [Ingress Configuration](###generate-a-certificate-via-ingress-annotations) section for more information.

## Use an Existing TLS Certificate

If you already have a TLS certificate available as a Kubernetes `Secret`, you can configure the
chart to use it directly instead of creating a new `Certificate` with `cert-manager`. To do this,
disable certificate creation and set `tls.existingSecret` to the name of your `Secret`.

The `Secret` must be of type `kubernetes.io/tls` and contain the standard `tls.crt`, `tls.key`
fields, with `ca.crt` being optional. If `ca.crt` is not provided, `tls.crt` will also be used
as the CA certificate.

If both `tls.existingSecret` and `tls.certificate.enabled` are set, the `tls.existingSecret` value
will be ignored and a new `Certificate` will be issued.

```yaml
global:
  insecure: false

tls:
  certificate:
    enabled: false
  existingSecret: my-custom-tls-secret
```

## Override Certificate Paths

By default, the TLS-related flags use the following paths when TLS is enabled:

`--ssl-ca-certfile`: `/app/cert/ca.crt`,
`--ssl-certfile`: `/app/cert/tls.crt`,
`--ssl-keyfile`: `/app/cert/tls.key`.

These paths can be overridden by specifying the flags in the `extraArgs`, as shown below:

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

## Ingress Configuration

### Generate a Certificate via Ingress Annotations

If you prefer to let `cert-manager` generate a `Certificate` through `Ingress` annotations,
you can disable the chart’s built-in certificate management.
To do this, set `tls.certificate.enabled` to `false` and leave `tls.existingSecret` empty.
Also ensure `global.insecure` is set to `false`.

The annotations you specify under `superlink.ingress.annotations` will be passed directly to the
`Ingress` resource. For supported annotation keys, refer to the `cert-manager`
[documentation](https://cert-manager.io/docs/usage/ingress/).

```yaml
global:
  insecure: false

tls:
  certificate:
    enabled: false
  existingSecret: ""

superlink:
 ingress:
    enabled: true
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
    tls:
      enabled: true
```

The chart creates an `Ingress` for SuperLink with the provided annotations. `cert-manager`
generates a `Certificate`, and stores it in a Kubernetes `Secret`. That `Secret` is then mounted
into the SuperLink container so it can use the TLS keypair for its endpoints. The same `Secret`
is also referenced by the `Ingress` resource, which means the generated certificate is
shared between the `Ingress` and the SuperLink service.

By default, the Ingress-generated certificate is stored in a `Secret` named
`<chart-name>-server-tls`.

You can override this by setting `superlink.ingress.tls.secretName`:

```yaml
superlink:
 ingress:
    enabled: true
    tls:
      enabled: true
      secretName: my-custom-tls-secret
```

**Important:**

- If `tls.certificate.enabled` is set to `true` or if `tls.existingSecret` is specified, the
  `secretName` of the `Ingress` resource is ignored. In that case, the chart will rely on the
  `Certificate` created through its own mechanism.
- Always ensure that when you use `Ingress` annotations, no certificate is generated or referenced
  by this chart. Otherwise, you may end up with conflicting secrets.

### TLS Certificate with Additional Hosts

By default, when `cert-manager` issues a `Certificate` it only includes the DNS name specified in
`commonName`, which is derived from `global.domain`.

In some deployments, the server and client charts may run inside the same Kubernetes cluster,
while the Exec API is exposed publicly over the internet. In this scenario, the SuperNodes
need to connect to the SuperLink using its internal service URL
(e.g., `<superlink>.<namespace>.svc.cluster.local`).

To support this, you can extend the certificate with additional hosts by using
`superlink.ingress.extraHosts`.

The example below shows how to configure the `Ingress` so that the generated `Certificate` covers
both the external hostname and the internal service name:

```yaml
global:
  insecure: false

tls:
  certificate:
    enabled: false
  existingSecret: ""

superlink:
 ingress:
    enabled: true
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
    tls:
      enabled: true
```

### SSL-Passthrough

In some scenarios, you may want to let the SuperLink service terminate TLS directly rather than
having the Ingress controller handle TLS termination. This is useful if you need to use protocols
such as gRPC over TLS end-to-end, where the Ingress should simply forward the encrypted traffic
without decrypting it.

To enable this mode with the NGINX Ingress Controller, you can configure SSL passthrough by setting
the `nginx.ingress.kubernetes.io/ssl-passthrough` annotation to `"true"`. In this configuration,
the Ingress forwards TLS traffic directly to the SuperLink container,
which then handles certificate validation and encryption.

```yaml
global:
  insecure: false

tls:
  certificate:
    enabled: true

superlink:
  enabled: true
  ingress:
    annotations:
      nginx.ingress.kubernetes.io/backend-protocol: GRPCS
      nginx.ingress.kubernetes.io/force-ssl-redirect: "false"
      nginx.ingress.kubernetes.io/ssl-passthrough: "true"
      nginx.ingress.kubernetes.io/ssl-redirect: "false"
    ingressClassName: nginx
    tls: false
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
      path: /
      pathType: ImplementationSpecific
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

## Parameters

### Helm parameters

| Name               | Description                                               | Value           |
| ------------------ | --------------------------------------------------------- | --------------- |
| `nameOverride`     | Override Replaces the name of the chart in the Chart.yaml | `flower-server` |
| `fullnameOverride` | Override Completely replaces the generated name.          | `""`            |

### Global parameters

| Name                                                 | Description                                                       | Value              |
| ---------------------------------------------------- | ----------------------------------------------------------------- | ------------------ |
| `global.annotations`                                 | Default Annotations                                               | `{}`               |
| `global.labels`                                      | Default Labels                                                    | `{}`               |
| `global.podLabels`                                   | Default PodLabels                                                 | `{}`               |
| `global.domain`                                      | Default Domain                                                    | `example.com`      |
| `global.insecure`                                    | Decide if you deploy the Flower Framework with TLS                | `true`             |
| `global.ingressClassName`                            | Default IngressClass                                              | `""`               |
| `global.nodeSelector`                                | Default node selector for all components                          | `{}`               |
| `global.tolerations`                                 | Default tolerations for all components                            | `[]`               |
| `global.affinity.podAntiAffinity`                    | Default affinity preset for all components                        | `soft`             |
| `global.affinity.podAntiAffinity`                    | Default pod anti-affinity rules. Either: `none`, `soft` or `hard` | `soft`             |
| `global.affinity.nodeAffinity.type`                  | Default node affinity rules. Either: `none`, `soft` or `hard`     | `hard`             |
| `global.affinity.nodeAffinity.matchExpressions`      | Default match expressions for node affinity                       | `[]`               |
| `global.nodeAuth.enabled`                            | Enables or Disables Node-Authentication SuperLink \<-> SuperNode  | `false`            |
| `global.nodeAuth.authListPublicKeys`                 | A list of ecdsa-sha2-nistp384 SuperNode keys                      | `[]`               |
| `global.userAuth.enabled`                            | Enables or disables the user authentication plugin.               | `false`            |
| `global.userAuth.config`                             | Set the user authentication configuration.                        | `{}`               |
| `global.userAuth.existingSecret`                     | Existing secret with user authentication configuration.           | `""`               |
| `global.license.enabled`                             | Enables or disables the configuration of the EE license.          | `true`             |
| `global.license.key`                                 | The EE license key.                                               | `""`               |
| `global.license.secretKey`                           | The name of the key inside the Kubernetes Secret                  | `FLWR_LICENSE_KEY` |
| `global.license.existingSecret`                      | Name of an existing Kubernetes Secret                             | `""`               |
| `global.securityContext.runAsUser`                   | Set Security Context runAsUser                                    | `49999`            |
| `global.securityContext.runAsGroup`                  | Set Security Context runAsGroup                                   | `49999`            |
| `global.securityContext.fsGroup`                     | Set Security Context fsGroup                                      | `49999`            |
| `global.podSecurityContext.runAsNonRoot`             | Set Security Context runAsNonRoot                                 | `true`             |
| `global.podSecurityContext.readOnlyRootFilesystem`   | Set Security Context readOnlyRootFilesystem                       | `true`             |
| `global.podSecurityContext.allowPrivilegeEscalation` | Set Security Context allowPrivilegeEscalation                     | `false`            |
| `global.podSecurityContext.seccompProfile.type`      | Set Security Context seccompProfile                               | `RuntimeDefault`   |
| `global.podSecurityContext.capabilities.drop`        | Set Security Context capabilities                                 | `["ALL"]`          |
| `global.env`                                         | Default environment variables                                     | `[]`               |
| `global.image.pullPolicy`                            | Default image pullPolicy                                          | `IfNotPresent`     |

### TLS Configuration

| Name                                 | Description                                                         | Value             |
| ------------------------------------ | ------------------------------------------------------------------- | ----------------- |
| `tls.issuer.enabled`                 | Enable automatic creation of a cert-manager Issuer.                 | `true`            |
| `tls.issuer.name`                    | Name of the Issuer resource to use.                                 | `""`              |
| `tls.issuer.spec`                    | The contents of the `.spec` block for the cert-manager Issuer.      | `{}`              |
| `tls.certificate.enabled`            | Enable automatic creation of a cert-manager Certificate.            | `true`            |
| `tls.certificate.annotations`        | Certificate CRD annotations.                                        | `{}`              |
| `tls.certificate.secretName`         | Name of the Kubernetes Secret to store the TLS key and certificate. | `""`              |
| `tls.certificate.issuerGroup`        | API group for the issuer. Defaults to `cert-manager.io`.            | `cert-manager.io` |
| `tls.certificate.existingIssuer`     | Name of an existing Issuer or ClusterIssuer to use.                 | `""`              |
| `tls.certificate.existingIssuerKind` | Kind of the existing issuer (`Issuer` or `ClusterIssuer`).          | `""`              |
| `tls.certificate.duration`           | The requested ‘duration’ (i.e. lifetime) of the Certificate.        | `43800h`          |
| `tls.certificate.renewBefore`        | How long before the currently issued certificate’s expiry           | `360h`            |
| `tls.certificate.privateKey`         | Private key options. These include the key algorithm and            | `{}`              |
| `tls.certificate.usages`             | Requested key usages and extended key usages.                       | `[]`              |
| `tls.certificate.additionalHosts`    | Additional hosts you want to put into the SAN's                     | `[]`              |
| `tls.existingSecret`                 | Name of an existing Kubernetes Secret                               | `""`              |

### Component SuperLink

| Name                                                           | Description                                                                                                             | Value                                |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `superlink.name`                                               | Name of the SuperLink                                                                                                   | `superlink`                          |
| `superlink.enabled`                                            | Enable or Disable SuperLink                                                                                             | `true`                               |
| `superlink.resources`                                          | Set container requests and limits for different resources like CPU or memory (essential for production workloads)       | `{}`                                 |
| `superlink.volumes`                                            | Specify a list of volumes for the SuperLink pod(s)                                                                      | `[]`                                 |
| `superlink.volumeMounts`                                       | Allows to specify additional VolumeMounts                                                                               | `[]`                                 |
| `superlink.executor.plugin`                                    | The executer plugin of the SuperLink                                                                                    | `flwr.superexec.deployment:executor` |
| `superlink.executor.config`                                    | Set executor config arguments                                                                                           | `{}`                                 |
| `superlink.isolationMode`                                      | The isolation mode of the SuperLink                                                                                     | `subprocess`                         |
| `superlink.automountServiceAccountToken`                       | Automount SA-Token into the pod.                                                                                        | `true`                               |
| `superlink.serviceAccount.enabled`                             | Enabled a service account for the application controller                                                                | `true`                               |
| `superlink.serviceAccount.annotations`                         | Annotations applied to enabled service account                                                                          | `{}`                                 |
| `superlink.serviceAccount.labels`                              | Labels applied to enabled service account                                                                               | `{}`                                 |
| `superlink.serviceAccount.automountServiceAccountToken`        | Automount SA-Token                                                                                                      | `true`                               |
| `superlink.service.servicePortApiName`                         | Prefix of the SuperLink API port                                                                                        | `api`                                |
| `superlink.service.servicePortApi`                             | Port to expose for the SuperLink API                                                                                    | `9093`                               |
| `superlink.service.nodePortApi`                                | Node port for SuperLink API                                                                                             | `""`                                 |
| `superlink.service.type`                                       | Valid are ClusterIP, NodePort or Loadbalancer                                                                           | `ClusterIP`                          |
| `superlink.service.servicePortDriverName`                      | Prefix of the SuperLink Driver API port                                                                                 | `driver`                             |
| `superlink.service.servicePortDriver`                          | Port to expose for the SuperLink Driver API                                                                             | `9091`                               |
| `superlink.service.nodePortDriver`                             | Node port for SuperLink Driver API                                                                                      | `""`                                 |
| `superlink.service.servicePortFleetName`                       | Prefix of the SuperLink Fleet API port                                                                                  | `fleet`                              |
| `superlink.service.servicePortFleet`                           | Port to expose for the SuperLink Fleet API                                                                              | `9092`                               |
| `superlink.service.nodePortFleet`                              | Node port for SuperLink Fleet API                                                                                       | `""`                                 |
| `superlink.service.servicePortSimulationIoName`                | Prefix of the SuperLink SimulationIo API port                                                                           | `simulationio`                       |
| `superlink.service.servicePortSimulationIo`                    | Port to expose for the SuperLink SimulationIo API                                                                       | `9096`                               |
| `superlink.service.nodePortSimulationIo`                       | Node port for SuperLink SimulationIo API                                                                                | `""`                                 |
| `superlink.containerPorts.api`                                 | Container port for SuperLink Exec API                                                                                   | `9093`                               |
| `superlink.containerPorts.driver`                              | Container port for SuperLink Driver API                                                                                 | `9091`                               |
| `superlink.containerPorts.fleet`                               | Container port for SuperLink Fleet API                                                                                  | `9092`                               |
| `superlink.containerPorts.simulationIo`                        | Container port for SuperLink SimulationIo API                                                                           | `9096`                               |
| `superlink.replicaCount`                                       | The number of SuperLink pods to run                                                                                     | `1`                                  |
| `superlink.labels`                                             | Extra labels for SuperLink pods                                                                                         | `{}`                                 |
| `superlink.extraArgs`                                          | Add extra arguments to the default arguments for the SuperLink                                                          | `[]`                                 |
| `superlink.nodeSelector`                                       | Node labels for SuperLink pods which merges with global.nodeSelector                                                    | `{}`                                 |
| `superlink.tolerations`                                        | Node tolerations for SuperLink pods which merges with global.tolerations                                                | `[]`                                 |
| `superlink.updateStrategy.type`                                | SuperLink deployment strategy type                                                                                      | `RollingUpdate`                      |
| `superlink.updateStrategy.rollingUpdate`                       | SuperLink deployment rolling update configuration parameters                                                            | `{}`                                 |
| `superlink.affinity`                                           | Node affinity for SuperLink pods which merges with global.affinity                                                      | `{}`                                 |
| `superlink.env`                                                | Array with extra environment variables to add to SuperLink nodes which merges with global.env                           | `[]`                                 |
| `superlink.podSecurityContext`                                 | Security settings that for the SuperLink Pods                                                                           | `{}`                                 |
| `superlink.securityContext`                                    | Security settings that for the SuperLink                                                                                | `{}`                                 |
| `superlink.ingress.enabled`                                    | Enable the ingress resource                                                                                             | `false`                              |
| `superlink.ingress.annotations`                                | Additional annotations for the ingress                                                                                  | `{}`                                 |
| `superlink.ingress.ingressClassName`                           | Defines which ingress controller which implement the resource                                                           | `""`                                 |
| `superlink.ingress.tls.enabled`                                | Enable TLS termination at the Ingress level.                                                                            | `false`                              |
| `superlink.ingress.tls.secretName`                             | Name of the Kubernetes Secret that will contain the                                                                     | `""`                                 |
| `superlink.ingress.api.enabled`                                | Enable an ingress resource for SuperLink API                                                                            | `false`                              |
| `superlink.ingress.api.hostname`                               | Ingress hostname for the SuperLink API ingress                                                                          | `exec-api.example.com`               |
| `superlink.ingress.api.path`                                   | SuperLink API ingress path                                                                                              | `/`                                  |
| `superlink.ingress.api.pathType`                               | Ingress path type. One of Exact, Prefix or ImplementationSpecific                                                       | `ImplementationSpecific`             |
| `superlink.ingress.fleet.enabled`                              | Enable an ingress resource for SuperLink Fleet API                                                                      | `false`                              |
| `superlink.ingress.fleet.hostname`                             | Ingress hostname for the SuperLink Fleet API ingress                                                                    | `fleet.example.com`                  |
| `superlink.ingress.fleet.path`                                 | SuperLink Fleet API ingress path                                                                                        | `/`                                  |
| `superlink.ingress.fleet.pathType`                             | Ingress path type. One of Exact, Prefix or ImplementationSpecific                                                       | `ImplementationSpecific`             |
| `superlink.ingress.driver.enabled`                             | Enable an ingress resource for SuperLink Driver API                                                                     | `false`                              |
| `superlink.ingress.driver.hostname`                            | Ingress hostname for the SuperLink Driver API ingress                                                                   | `driver.example.com`                 |
| `superlink.ingress.driver.path`                                | SuperLink Driver API ingress path                                                                                       | `/`                                  |
| `superlink.ingress.driver.pathType`                            | Ingress path type. One of Exact, Prefix or ImplementationSpecific                                                       | `ImplementationSpecific`             |
| `superlink.ingress.simulationIo.enabled`                       | Enable an ingress resource for SuperLink SimulationIo API                                                               | `false`                              |
| `superlink.ingress.simulationIo.hostname`                      | Ingress hostname for the SuperLink SimulationIo API ingress                                                             | `simulation.example.com`             |
| `superlink.ingress.simulationIo.path`                          | SuperLink SimulationIo API ingress path                                                                                 | `/`                                  |
| `superlink.ingress.simulationIo.pathType`                      | Ingress path type. One of Exact, Prefix or ImplementationSpecific                                                       | `ImplementationSpecific`             |
| `superlink.ingress.extraHosts`                                 | An array with additional hostname(s) to be covered with the ingress record                                              | `[]`                                 |
| `superlink.ingress.extraTls`                                   | TLS configuration for additional hostname(s) to be covered with this ingress record                                     | `[]`                                 |
| `superlink.ingress.extraRules`                                 | Additional rules to be covered with this ingress record                                                                 | `[]`                                 |
| `superlink.lifecycle`                                          | SuperLink container(s) to automate configuration before or after startup                                                | `{}`                                 |
| `superlink.annotations`                                        | Additional custom annotations for SuperLink                                                                             | `{}`                                 |
| `superlink.selectorLabels`                                     | Extra selectorLabels for SuperLink pods                                                                                 | `{}`                                 |
| `superlink.podAnnotations`                                     | Annotations for SuperLink pods                                                                                          | `{}`                                 |
| `superlink.podLabels`                                          | Extra podLabels for SuperLink pods                                                                                      | `{}`                                 |
| `superlink.imagePullSecrets`                                   | SuperLink image pull secrets which overrides global.imagePullSecrets                                                    | `[]`                                 |
| `superlink.image.registry`                                     | SuperLink image registry                                                                                                | `registry.hub.docker.com`            |
| `superlink.image.repository`                                   | SuperLink image repository                                                                                              | `flwr/superlink-ee`                  |
| `superlink.image.tag`                                          | SuperLink image tag                                                                                                     | `1.20.0-ubuntu`                      |
| `superlink.image.digest`                                       | SuperLink image digest                                                                                                  | `""`                                 |
| `superlink.image.pullPolicy`                                   | SuperLink image pullPolicy which Components image pullPolicy                                                            | `IfNotPresent`                       |
| `superlink.networkPolicy.enabled`                              | Specifies whether a NetworkPolicy should be created                                                                     | `true`                               |
| `superlink.networkPolicy.allowExternal`                        | Allow external ingress traffic                                                                                          | `true`                               |
| `superlink.networkPolicy.allowExternalEgress`                  | Allow unrestricted egress traffic                                                                                       | `true`                               |
| `superlink.networkPolicy.extraIngress`                         | Add extra ingress rules to the NetworkPolicy                                                                            | `[]`                                 |
| `superlink.networkPolicy.extraEgress`                          | Add extra ingress rules to the NetworkPolicy (ignored if allowExternalEgress=true)                                      | `[]`                                 |
| `superlink.networkPolicy.driver.ingressPodMatchLabels`         | Labels to match to allow traffic from other pods. Ignored if `superlink.networkPolicy.allowExternal` is true.           | `{}`                                 |
| `superlink.networkPolicy.driver.ingressNSMatchLabels`          | Labels to match to allow traffic from other namespaces. Ignored if `superlink.networkPolicy.allowExternal` is true.     | `{}`                                 |
| `superlink.networkPolicy.driver.ingressNSPodMatchLabels`       | Pod labels to match to allow traffic from other namespaces. Ignored if `superlink.networkPolicy.allowExternal` is true. | `{}`                                 |
| `superlink.networkPolicy.fleet.ingressPodMatchLabels`          | Labels to match to allow traffic from other pods. Ignored if `superlink.networkPolicy.allowExternal` is true.           | `{}`                                 |
| `superlink.networkPolicy.fleet.ingressNSMatchLabels`           | Labels to match to allow traffic from other namespaces. Ignored if `superlink.networkPolicy.allowExternal` is true.     | `{}`                                 |
| `superlink.networkPolicy.fleet.ingressNSPodMatchLabels`        | Pod labels to match to allow traffic from other namespaces. Ignored if `superlink.networkPolicy.allowExternal` is true. | `{}`                                 |
| `superlink.networkPolicy.api.ingressPodMatchLabels`            | Labels to match to allow traffic from other pods. Ignored if `superlink.networkPolicy.allowExternal` is true.           | `{}`                                 |
| `superlink.networkPolicy.api.ingressNSMatchLabels`             | Labels to match to allow traffic from other namespaces. Ignored if `superlink.networkPolicy.allowExternal` is true.     | `{}`                                 |
| `superlink.networkPolicy.api.ingressNSPodMatchLabels`          | Pod labels to match to allow traffic from other namespaces. Ignored if `superlink.networkPolicy.allowExternal` is true. | `{}`                                 |
| `superlink.networkPolicy.simulationIo.ingressPodMatchLabels`   | Labels to match to allow traffic from other pods. Ignored if `superlink.networkPolicy.allowExternal` is true.           | `{}`                                 |
| `superlink.networkPolicy.simulationIo.ingressNSMatchLabels`    | Labels to match to allow traffic from other namespaces. Ignored if `superlink.networkPolicy.allowExternal` is true.     | `{}`                                 |
| `superlink.networkPolicy.simulationIo.ingressNSPodMatchLabels` | Pod labels to match to allow traffic from other namespaces. Ignored if `superlink.networkPolicy.allowExternal` is true. | `{}`                                 |

### Component ServerApp

| Name                                                    | Description                                                                                                       | Value                       |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `serverapp.name`                                        | Name of the ServerApp                                                                                             | `serverapp`                 |
| `serverapp.enabled`                                     | Enable or disable ServerApp                                                                                       | `false`                     |
| `serverapp.resources`                                   | Set container requests and limits for different resources like CPU or memory (essential for production workloads) | `{}`                        |
| `serverapp.superlink`                                   | Address of the SuperLink the ServerApp should connect to                                                          | `{}`                        |
| `serverapp.volumes`                                     | Optionally specify list of volumes for the ServerApp pod(s)                                                       | `[]`                        |
| `serverapp.volumeMounts`                                | Allows to specify additional VolumeMounts                                                                         | `[]`                        |
| `serverapp.automountServiceAccountToken`                | Automount SA-Token into the pod.                                                                                  | `true`                      |
| `serverapp.serviceAccount.enabled`                      | Enable a service account for this component                                                                       | `true`                      |
| `serverapp.serviceAccount.annotations`                  | Annotations applied to enabled service account                                                                    | `{}`                        |
| `serverapp.serviceAccount.labels`                       | Labels applied to enabled service account                                                                         | `{}`                        |
| `serverapp.serviceAccount.automountServiceAccountToken` | Automount SA-Token                                                                                                | `true`                      |
| `serverapp.podSecurityContext`                          | Security settings that for the SuperLink Pods                                                                     | `{}`                        |
| `serverapp.replicas`                                    | The number of ServerApp pods to run                                                                               | `1`                         |
| `serverapp.labels`                                      | Extra labels for ServerApp pods                                                                                   | `{}`                        |
| `serverapp.extraArgs`                                   | Add extra arguments to the default arguments for the ServerApp                                                    | `[]`                        |
| `serverapp.nodeSelector`                                | Node labels for ServerApp pods which merges with global.nodeSelector                                              | `{}`                        |
| `serverapp.tolerations`                                 | Node tolerations for ServerApp pods which merges with global.tolerations                                          | `[]`                        |
| `serverapp.updateStrategy.type`                         | ServerApp deployment strategy type                                                                                | `RollingUpdate`             |
| `serverapp.updateStrategy.rollingUpdate`                | ServerApp deployment rolling update configuration parameters                                                      | `{}`                        |
| `serverapp.affinity`                                    | Node affinity for ServerApp pods which merges with global.affinity                                                | `{}`                        |
| `serverapp.env`                                         | Array with extra environment variables to add to ServerApp nodes which merges with global.env                     | `[]`                        |
| `serverapp.lifecycle`                                   | ServerApp container(s) to automate configuration before or after startup                                          | `{}`                        |
| `serverapp.annotations`                                 | Additional custom annotations for ServerApp                                                                       | `{}`                        |
| `serverapp.selectorLabels`                              | Extra selectorLabels for ServerApp pods                                                                           | `{}`                        |
| `serverapp.podAnnotations`                              | Annotations for ServerApp pods                                                                                    | `{}`                        |
| `serverapp.podLabels`                                   | Extra podLabels for ServerApp pods                                                                                | `{}`                        |
| `serverapp.imagePullSecrets`                            | ServerApp image pull secrets which overrides global.imagePullSecrets                                              | `[]`                        |
| `serverapp.image.registry`                              | ServerApp image registry                                                                                          | `registry.hub.docker.com`   |
| `serverapp.image.repository`                            | ServerApp image repository                                                                                        | `flwr/serverapp`            |
| `serverapp.image.tag`                                   | Image tag of ServerApp                                                                                            | `1.20.0-py3.11-ubuntu24.04` |
| `serverapp.image.digest`                                | Image digest of ServerApp                                                                                         | `""`                        |
| `serverapp.image.pullPolicy`                            | Components image pullPolicy                                                                                       | `Always`                    |
| `serverapp.networkPolicy.enabled`                       | Specifies whether a NetworkPolicy should be created                                                               | `true`                      |
| `serverapp.networkPolicy.allowExternalEgress`           | Allow unrestricted egress traffic                                                                                 | `true`                      |
| `serverapp.networkPolicy.extraEgress`                   | Add extra ingress rules to the NetworkPolicy (ignored if allowExternalEgress=true)                                | `[]`                        |

### Component OpenFGA

| Name              | Description                                    | Value   |
| ----------------- | ---------------------------------------------- | ------- |
| `openfga.enabled` | Enable the openfga subchart and deploy OpenFGA | `false` |
