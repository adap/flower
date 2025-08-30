---
myst:
  html_meta:
    description: Deploy Flower's SuperNode Helm chart to install client 
      federated learning components. Default config mirrors official releases 
      for seamless integration.
    property:og:description: Deploy Flower's SuperNode Helm chart to install 
      client federated learning components. Default config mirrors official 
      releases for seamless integration.
---

# Deploy SuperNode using Helm

```{note}
Flower Helm charts are a Flower Enterprise feature. See [Flower Enterprise](https://flower.ai/enterprise) for details.
```

The Flower Framework offers a unified approach to federated learning, analytics, and evaluation,
allowing you to federate any workload, machine learning framework, or programming language.

This Helm chart installs the client-side components of the Flower Framework, specifically
setting up the SuperNode.

The default installation configuration aims to replicate the functionality and setup of the
provided Flower Framework releases.

## Multi Project Setup

To install multiple types of SuperNodes, such as a federation for running PyTorch and another for
TensorFlow, you need to install the Helm Chart multiple times with different names. This allows
each deployment to have its own configurations and dependencies.

For instance, you can install the Chart for the PyTorch setup by adjusting the values.yaml file
as shown below:

```yaml
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
    tag: 1.20.0-pytorch
```

Install this configuration using the following command:

```sh
$ helm install pytorch . --values values.yaml
```

This will deploy 10 SuperNodes named `pytorch-flower-client-supernode-<random>`.

For a TensorFlow setup, modify the `values.yaml` file as follows:

```yaml
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
    tag: 1.20.0-tensorflow
```

Install this configuration using the following command:

```sh
$ helm install tensorflow . --values values.yaml
```

This will deploy 3 SuperNodes named `tensorflow-flower-client-supernode-<random>`.

## Deploy Flower Framework without TLS

By default, the Flower Framework is deployed without TLS. This means `global.insecure` is
set to `true`.

You might want to deploy the Flower framework without TLS for testing or internal use. Be
cautious as this exposes your deployment to potential security risks.

```yaml
global:
  insecure: true
```

## Deploy Flower Framework with TLS

When using private CAs, the SuperNode must trust the CA certificate in order to connect
securely to the SuperLink.

To provide the CA certificate, set `global.insecure` to `false` and create a `Secret` of type
`kubernetes.io/tls` named `flower-client-tls`:

```yaml
global:
  insecure: false
```

If you want to use a different `Secret` name, override the default by setting
`supernode.superlink.certificate.existingSecret`:

```yaml
global:
  insecure: false
supernode:
  superlink:
    certificate:
      existingSecret: my-custom-tls-secret-name
```

**Important:**

The recommended practice is to mount different `Secret`s for the SuperLink and the
SuperNodes `existingSecret` parameter. Keeping these `Secrets` separate ensures
that if the `Secret` containing the server’s private key and certificate is ever
tampered with, the client will fail to connect rather than trusting a compromised
server.

For further details, refer to the [`cert-manager` documentation](https://cert-manager.io/docs/trust/).

If the SuperLink certificate (of type `kubernetes.io/tls`) is deployed in the same cluster and
namespace as the SuperNode, you can enable `supernode.superlink.certificate.copyFromExistingSecret`.
This instructs the chart to create a new `Secret` containing the CA certificate.
It copies `ca.crt` from the SuperLink `Secret`, or falls back to `tls.crt` if `ca.crt` is not
present.

By default, the copied `Secret` is named `flower-client-tls`. You can customize this name with
`supernode.superlink.certificate.copyFromExistingSecret.secretName`:

```yaml
global:
  insecure: false
supernode:
  superlink:
    certificate:
      existingSecret: superlink-tls-secret-name
    copyFromExistingSecret:
      enabled: true
      secretName: my-custom-tls-secret-name
```

## Node Authentication

To enable Node Authentication, you need to specify a private key in either PKCS8 or OpenSSH
(PEM-like) format. This example assumes that the SuperLink is also configured for Node
Authentication and recognizes the `ecdsa-sha2-nistp384 [...]` public key of this SuperNode.

```yaml
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
```

## Isolated Setup

### Isolation All-in-One

To install SuperNode in isolation mode using the "process" configuration, both the ClientApp and
SuperNode need to be enabled. By default, the ClientApp connects to the SuperNode internally
within the cluster, so there is no need to set `supernode.address` and `supernode.port` unless the
connection is external. This setup assumes that both components are running within the same cluster.

```yaml
[...]
supernode:
  enabled: true
  [...]
  isolationMode: process
[...]
clientapp:
  enabled: true
[...]
```

### Isolation Distributed

You can also deploy the SuperNode and ClientApp separately. To do this, you need to deploy the
chart twice: once with `supernode.enabled=true` and once with `clientapp.enabled=true`. To allow
the ClientApp to connect to the SuperNode in this configuration, enable the SuperNode ingress by
setting `supernode.ingress.enabled=true`. This setup is intended for scenarios where the components
run on different clusters or a hybrid environment involving Kubernetes and ClientApp native
installations.

```yaml
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
```

## Node Configuration

You can add a node configuration to configure a SuperNode. The YAML datatype is preserved when
passing it in the Python application:

```yaml
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
```

## Parameters

### Helm parameters

| Name               | Description                                      | Value           |
| ------------------ | ------------------------------------------------ | --------------- |
| `nameOverride`     | Replaces the name of the chart in the Chart.yaml | `flower-client` |
| `fullnameOverride` | Completely replaces the generated name.          | `""`            |

### Global parameters

| Name                                                 | Description                                                       | Value            |
| ---------------------------------------------------- | ----------------------------------------------------------------- | ---------------- |
| `global.annotations`                                 | Default Annotations                                               | `{}`             |
| `global.labels`                                      | Default Labels                                                    | `{}`             |
| `global.podLabels`                                   | Default PodLabels                                                 | `{}`             |
| `global.insecure`                                    | Decide if you deploy the Flower Framework with TLS                | `true`           |
| `global.nodeSelector`                                | Default node selector for all components                          | `{}`             |
| `global.tolerations`                                 | Default tolerations for all components                            | `[]`             |
| `global.affinity.podAntiAffinity`                    | Default affinity preset for all components                        | `soft`           |
| `global.affinity.podAntiAffinity`                    | Default pod anti-affinity rules. Either: `none`, `soft` or `hard` | `soft`           |
| `global.affinity.nodeAffinity.type`                  | Default node affinity rules. Either: `none`, `soft` or `hard`     | `hard`           |
| `global.affinity.nodeAffinity.matchExpressions`      | Default match expressions for node affinity                       | `[]`             |
| `global.nodeAuth.enabled`                            | Enables or Disables Node-Authentication SuperLink \<-> SuperNode  | `false`          |
| `global.nodeAuth.authSupernodePrivateKey`            | Specifies the ecdsa-sha2-nistp384 private key                     | `""`             |
| `global.nodeAuth.authSupernodePublicKey`             | Specifies the ecdsa-sha2-nistp384 public key                      | `""`             |
| `global.securityContext.runAsUser`                   | Set Security Context runAsUser                                    | `49999`          |
| `global.securityContext.runAsGroup`                  | Set Security Context runAsGroup                                   | `49999`          |
| `global.securityContext.fsGroup`                     | Set Security Context fsGroup                                      | `49999`          |
| `global.podSecurityContext.runAsNonRoot`             | Set Security Context runAsNonRoot                                 | `true`           |
| `global.podSecurityContext.readOnlyRootFilesystem`   | Set Security Context readOnlyRootFilesystem                       | `true`           |
| `global.podSecurityContext.allowPrivilegeEscalation` | Set Security Context allowPrivilegeEscalation                     | `false`          |
| `global.podSecurityContext.seccompProfile.type`      | Set Security Context seccompProfile                               | `RuntimeDefault` |
| `global.podSecurityContext.capabilities.drop`        | Set Security Context capabilities                                 | `["ALL"]`        |
| `global.env`                                         | Default environment variables                                     | `[]`             |
| `global.image.pullPolicy`                            | Default image pullPolicy                                          | `IfNotPresent`   |

### Component SuperNode

| Name                                                                | Description                                                                                                             | Value                      |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `supernode.name`                                                    | Name of the SuperNode                                                                                                   | `supernode`                |
| `supernode.enabled`                                                 | Enable or disable SuperNode                                                                                             | `true`                     |
| `supernode.resources`                                               | Set container requests and limits for different resources like CPU or memory (essential for production workloads)       | `{}`                       |
| `supernode.node.config`                                             |                                                                                                                         | `{}`                       |
| `supernode.isolationMode`                                           | The isolation mode of the SuperNode                                                                                     | `subprocess`               |
| `supernode.resources`                                               | Set container requests and limits for different resources like CPU or memory (essential for production workloads)       | `{}`                       |
| `supernode.superlink.address`                                       | Address of the SuperLink the SuperNodes should connect to                                                               | `my-superlink.example.com` |
| `supernode.superlink.port`                                          | Port of the SuperLink the SuperNodes should connect to                                                                  | `9092`                     |
| `supernode.superlink.certificate.existingSecret`                    |                                                                                                                         | `""`                       |
| `supernode.superlink.certificate.copyFromExistingSecret.enabled`    |                                                                                                                         | `false`                    |
| `supernode.superlink.certificate.copyFromExistingSecret.secretName` |                                                                                                                         | `""`                       |
| `supernode.volumes`                                                 | Specify a list of volumes for the SuperNode pod(s)                                                                      | `[]`                       |
| `supernode.volumeMounts`                                            | Allows to specify additional VolumeMounts                                                                               | `[]`                       |
| `supernode.automountServiceAccountToken`                            | Automount SA-Token into the pod.                                                                                        | `true`                     |
| `supernode.serviceAccount.enabled`                                  | Enable a service account for this component                                                                             | `true`                     |
| `supernode.serviceAccount.annotations`                              | Annotations applied to enabled service account                                                                          | `{}`                       |
| `supernode.serviceAccount.labels`                                   | Labels applied to enabled service account                                                                               | `{}`                       |
| `supernode.serviceAccount.automountServiceAccountToken`             | Automount SA-Token                                                                                                      | `true`                     |
| `supernode.service.type`                                            | Valid are ClusterIP, NodePort or Loadbalancer                                                                           | `ClusterIP`                |
| `supernode.service.servicePortClientAppIOName`                      | Prefix of the SuperNode ClientAppIO API port                                                                            | `clientappio`              |
| `supernode.service.servicePortClientAppIO`                          | Port to expose for the SuperNode ClientAppIO API                                                                        | `9094`                     |
| `supernode.service.nodePortClientAppIO`                             | Node port for SuperNode ClientAppIO API                                                                                 | `39094`                    |
| `supernode.containerPorts.clientappio`                              | Container port for SuperNode ClientAppIO API                                                                            | `9094`                     |
| `supernode.podSecurityContext`                                      |                                                                                                                         | `{}`                       |
| `supernode.replicas`                                                | The number of SuperNode pods to run                                                                                     | `1`                        |
| `supernode.labels`                                                  | Extra labels for SuperNode pods                                                                                         | `{}`                       |
| `supernode.extraArgs`                                               | Add extra arguments to the default arguments for the SuperNode                                                          | `[]`                       |
| `supernode.nodeSelector`                                            | Node labels for SuperNode pods which merges with global.nodeSelector                                                    | `{}`                       |
| `supernode.tolerations`                                             | Node tolerations for SuperNode pods which merges with global.tolerations                                                | `[]`                       |
| `supernode.updateStrategy.type`                                     | SuperNode deployment strategy type                                                                                      | `RollingUpdate`            |
| `supernode.updateStrategy.rollingUpdate`                            | SuperNode deployment rolling update configuration parameters                                                            | `{}`                       |
| `supernode.affinity`                                                | Node affinity for SuperNode pods which merges with global.affinity                                                      | `{}`                       |
| `supernode.env`                                                     | Array with extra environment variables to add to SuperNode nodes which merges with global.env                           | `[]`                       |
| `supernode.lifecycle`                                               | SuperNode container(s) to automate configuration before or after startup                                                | `{}`                       |
| `supernode.annotations`                                             | Additional custom annotations for SuperNode                                                                             | `{}`                       |
| `supernode.selectorLabels`                                          | Extra selectorLabels for SuperNode pods                                                                                 | `{}`                       |
| `supernode.podAnnotations`                                          | Annotations for SuperNode pods                                                                                          | `{}`                       |
| `supernode.podLabels`                                               | Extra podLabels for SuperNode pods                                                                                      | `{}`                       |
| `supernode.imagePullSecrets`                                        | SuperNode image pull secrets which overrides global.imagePullSecrets                                                    | `[]`                       |
| `supernode.image.registry`                                          | SuperNode image registry                                                                                                | `registry.hub.docker.com`  |
| `supernode.image.repository`                                        | SuperNode image repository                                                                                              | `flwr/supernode-ee`        |
| `supernode.image.tag`                                               | Image tag of SuperNode                                                                                                  | `1.20.0-ubuntu`            |
| `supernode.image.digest`                                            | Image digest of SuperNode                                                                                               | `""`                       |
| `supernode.image.pullPolicy`                                        | Components image pullPolicy                                                                                             | `IfNotPresent`             |
| `supernode.networkPolicy.enabled`                                   | Specifies whether a NetworkPolicy should be created                                                                     | `true`                     |
| `supernode.networkPolicy.allowExternal`                             | Allow external ingress traffic                                                                                          | `true`                     |
| `supernode.networkPolicy.allowExternalEgress`                       | Allow unrestricted egress traffic                                                                                       | `true`                     |
| `supernode.networkPolicy.extraIngress`                              | Add extra ingress rules to the NetworkPolicy                                                                            | `[]`                       |
| `supernode.networkPolicy.extraEgress`                               | Add extra ingress rules to the NetworkPolicy (ignored if allowExternalEgress=true)                                      | `[]`                       |
| `supernode.networkPolicy.ingressPodMatchLabels`                     | Labels to match to allow traffic from other pods. Ignored if `supernode.networkPolicy.allowExternal` is true.           | `{}`                       |
| `supernode.networkPolicy.ingressNSMatchLabels`                      | Labels to match to allow traffic from other namespaces. Ignored if `supernode.networkPolicy.allowExternal` is true.     | `{}`                       |
| `supernode.networkPolicy.ingressNSPodMatchLabels`                   | Pod labels to match to allow traffic from other namespaces. Ignored if `supernode.networkPolicy.allowExternal` is true. | `{}`                       |

### Component ClientApp

| Name                                                    | Description                                                                                                       | Value                       |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `clientapp.name`                                        | Name of the ClientApp                                                                                             | `clientapp`                 |
| `clientapp.enabled`                                     | Enable or disable ClientApp component                                                                             | `false`                     |
| `clientapp.node.config`                                 |                                                                                                                   | `{}`                        |
| `clientapp.resources`                                   | Set container requests and limits for different resources like CPU or memory (essential for production workloads) | `{}`                        |
| `clientapp.supernode`                                   | Address of the supernode the ClientApp should connect to                                                          | `{}`                        |
| `clientapp.volumes`                                     | Specify a list of volumes for the ClientApp pod(s)                                                                | `[]`                        |
| `clientapp.volumeMounts`                                | Allows to specify additional VolumeMounts                                                                         | `[]`                        |
| `clientapp.automountServiceAccountToken`                | Automount SA-Token into the pod.                                                                                  | `true`                      |
| `clientapp.serviceAccount.enabled`                      | Enable a service account for this component                                                                       | `true`                      |
| `clientapp.serviceAccount.annotations`                  | Annotations applied to enabled service account                                                                    | `{}`                        |
| `clientapp.serviceAccount.labels`                       | Labels applied to enabled service account                                                                         | `{}`                        |
| `clientapp.serviceAccount.automountServiceAccountToken` | Automount SA-Token                                                                                                | `true`                      |
| `clientapp.service.type`                                | Valid are ClusterIP, NodePort or Loadbalancer                                                                     | `ClusterIP`                 |
| `clientapp.service.servicePortClientAppIOName`          | Prefix of the ClientApp ClientAppIO API port                                                                      | `clientappio`               |
| `clientapp.service.servicePortClientAppIO`              | Ports to expose for the ClientApp ClientAppIO API                                                                 | `9094`                      |
| `clientapp.service.nodePortClientAppIO`                 | Node port for ClientApp ClientAppIO API                                                                           | `""`                        |
| `clientapp.containerPorts.clientappio`                  | Container port for ClientApp ClientAppIO API                                                                      | `9094`                      |
| `clientapp.podSecurityContext`                          |                                                                                                                   | `{}`                        |
| `clientapp.replicas`                                    | The number of ClientApp pods to run                                                                               | `1`                         |
| `clientapp.labels`                                      | Extra labels for ClientApp pods                                                                                   | `{}`                        |
| `clientapp.extraArgs`                                   | Add extra arguments to the default arguments for the ClientApp                                                    | `[]`                        |
| `clientapp.nodeSelector`                                | Node labels for ClientApp pods which merges with global.nodeSelector                                              | `{}`                        |
| `clientapp.tolerations`                                 | Node tolerations for ClientApp pods which merges with global.tolerations                                          | `[]`                        |
| `clientapp.updateStrategy.type`                         | ClientApp deployment strategy type                                                                                | `RollingUpdate`             |
| `clientapp.updateStrategy.rollingUpdate`                | ClientApp deployment rolling update configuration parameters                                                      | `{}`                        |
| `clientapp.affinity`                                    | Node affinity for ClientApp pods which merges with global.affinity                                                | `{}`                        |
| `clientapp.env`                                         | Array with extra environment variables to add to ClientApp nodes which merges with global.env                     | `[]`                        |
| `clientapp.lifecycle`                                   | ClientApp container(s) to automate configuration before or after startup                                          | `{}`                        |
| `clientapp.annotations`                                 | Additional custom annotations for ClientApp                                                                       | `{}`                        |
| `clientapp.selectorLabels`                              | Extra selectorLabels for ClientApp pods                                                                           | `{}`                        |
| `clientapp.podAnnotations`                              | Annotations for ClientApp pods                                                                                    | `{}`                        |
| `clientapp.podLabels`                                   | Extra podLabels for ClientApp pods                                                                                | `{}`                        |
| `clientapp.imagePullSecrets`                            | ClientApp image pull secrets which overrides global.imagePullSecrets                                              | `[]`                        |
| `clientapp.image.registry`                              | ClientApp image registry                                                                                          | `registry.hub.docker.com`   |
| `clientapp.image.repository`                            | ClientApp image repository                                                                                        | `flwr/clientapp`            |
| `clientapp.image.tag`                                   | Image tag of ClientApp                                                                                            | `1.20.0-py3.11-ubuntu24.04` |
| `clientapp.image.digest`                                | Image digest of ClientApp                                                                                         | `""`                        |
| `clientapp.image.pullPolicy`                            | Components image pullPolicy                                                                                       | `IfNotPresent`              |
| `clientapp.networkPolicy.enabled`                       | Specifies whether a NetworkPolicy should be created                                                               | `true`                      |
| `clientapp.networkPolicy.allowExternalEgress`           | Allow unrestricted egress traffic                                                                                 | `true`                      |
| `clientapp.networkPolicy.extraEgress`                   | Add extra ingress rules to the NetworkPolicy (ignored if allowExternalEgress=true)                                | `[]`                        |
