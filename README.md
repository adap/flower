# Flower (flwr) - A Friendly Federated Learning Research Framework

[![GitHub license](https://img.shields.io/github/license/adap/flower)](https://github.com/adap/flower/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/adap/flower/blob/main/CONTRIBUTING.md)
![Build](https://github.com/adap/flower/workflows/Build/badge.svg)

Flower is a research framework for building federated learning systems. The
design of Flower is based on a few guiding principles:

* **Customizable**: Federated learning systems vary wildly from one use case to
  another. Flower allows for a wide range of different configurations depending
  on the needs of each individual use case.

* **Extendable**: Flower originated from a research project at the Univerity of
  Oxford, so it was build with AI research in mind. Many components can be
  extended and overridden to build new state-of-the-art systems. 

* **Framework-agnostic**: Different machine learning frameworks have different
  strengths. Flower can be used with any machine learning framework, for
  example, [PyTorch](https://pytorch.org),
  [TensorFlow](https://tensorflow.org), or even raw [NumPy](https://numpy.org/)
  for users who enjoy computing gradients by hand.

* **Understandable**: Flower is written with maintainability in mind. The
  community is encouraged to both read and contribute to the codebase.

> Note: Even though Flower is used in production, it is published as
> pre-release software. Incompatible API changes are possible.

## Documentation

* [Documentation](https://flower.dev)

## Installation

Flower can be installed directly from the [PyPI](https://pypi.org/project/flwr/) using `pip`:

```bash
$ pip install flwr
```

## Run Examples

A number of examples show different usage scenarios in `src/flwr_example`. To
run an example, first install the necessary extras. Available extras:

- `examples-tensorflow`
- `examples-pytorch`

```bash
pip install flwr[examples-tensorflow]
```

Once the necessary extras (e.g., TensorFlow) are installed, you might want to
run the Fashion-MNIST example by starting a single server and multiple clients
in two terminals using the following commands.

Start server in the first terminal:

```bash
$ ./src/flwr_example/tf_fashion_mnist/run-server.sh
```

Start the clients in a second terminal:

```bash
$ ./src/flwr_example/tf_fashion_mnist/run-clients.sh
```

### Docker

If you have Docker on your machine you might want to skip most of the setup and
try out the example using the following commands:

```bash
# Create docker network `flower` so that containers can reach each other by name
$ docker network create flower
# Build the Flower docker containers
$ ./dev/docker_build.sh

# Run the docker containers (will tail a logfile created by a central logserver)
$ ./src/flwr_example/tf_fashion_mnist/run-docker.sh
```

This will start a slightly reduced setup with only four clients.

## Contributing to Flower

We welcome contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) to get
started!
