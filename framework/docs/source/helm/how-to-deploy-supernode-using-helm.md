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

## Deploy Flower Framework with TLS

By default, the Flower Framework is deployed with TLS enabled. This means `tls.enabled` is
set to `true`.

When using private CAs, the SuperNode must trust the CA certificate in order to connect
securely to the SuperLink.

To provide the CA certificate, set `tls.enabled` to `true` and create a `Secret` of type
`kubernetes.io/tls` named `flower-client-tls`:

```yaml
tls:
  enabled: true
```

If you want to use a different `Secret` name, override the default by setting
`supernode.superlink.certificate.existingSecret`:

```yaml
tls:
  enabled: true

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
tls:
  enabled: true

supernode:
  superlink:
    certificate:
      existingSecret: superlink-tls-secret-name
    copyFromExistingSecret:
      enabled: true
      secretName: my-custom-tls-secret-name
```

## Deploy Flower Framework without TLS

You might want to deploy the Flower framework without TLS for testing or internal use. Be
cautious as this exposes your deployment to potential security risks.

```yaml
tls:
  enabled: false
```

## Node Authentication

To enable Node Authentication, you need to specify a private key in either PKCS8 or OpenSSH
(PEM-like) format. This example assumes that the SuperLink is also configured for Node
Authentication and recognizes the `ecdsa-sha2-nistp384 [...]` public key of this SuperNode.

```yaml
global:
  nodeAuth:
    enabled: true
    authSupernodePrivateKey: |+
      -----BEGIN OPENSSH PRIVATE KEY-----
      [...]
      -----END OPENSSH PRIVATE KEY-----
    authSupernodePublicKey: ecdsa-sha2-nistp384 [...]

tls:
  enabled: true

supernode:
  enabled: true
  superlink:
    address: my-superlink.example.com
    port: 9092

clientapp:
  enabled: true
  supernode:
    address: my-supernode.example.com
    port: 9094
```

## Isolated Setup

### Isolation All-in-One

To install SuperNode in isolation mode using the "process" configuration, both the ClientApp and
SuperNode need to be enabled. By default, the ClientApp connects to the SuperNode internally
within the cluster, so there is no need to set `supernode.address` and `supernode.port` unless the
connection is external. This setup assumes that both components are running within the same cluster.

```yaml
supernode:
  enabled: true
  isolationMode: process

clientapp:
  enabled: true
```

### Isolation Distributed

You can also deploy the SuperNode and ClientApp separately. To do this, you need to deploy the
chart twice: once with `supernode.enabled=true` and once with `clientapp.enabled=true`.

```yaml
supernode:
  enabled: true

clientapp:
  enabled: true
  supernode:
    address: my-supernode.example.com
    port: 9094
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
