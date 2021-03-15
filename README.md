# Flower - A Friendly Federated Learning Framework

[![GitHub license](https://img.shields.io/github/license/adap/flower)](https://github.com/adap/flower/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/adap/flower/blob/main/CONTRIBUTING.md)
![Build](https://github.com/adap/flower/workflows/Build/badge.svg)
![Downloads](https://pepy.tech/badge/flwr)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://flower.dev/join-slack)

Flower (`flwr`) is a framework for building federated learning systems. The
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
  [TensorFlow](https://tensorflow.org), [MXNet](https://mxnet.apache.org/), or even raw [NumPy](https://numpy.org/)
  for users who enjoy computing gradients by hand.

* **Understandable**: Flower is written with maintainability in mind. The
  community is encouraged to both read and contribute to the codebase.

## Documentation

[Flower Documentation](https://flower.dev):

* [Installation](https://flower.dev/docs/installation.html)
* [Quickstart (TensorFlow)](https://flower.dev/docs/quickstart_tensorflow.html)
* [Quickstart (PyTorch)](https://flower.dev/docs/quickstart_pytorch.html)
* [Quickstart (MXNet)](https://flower.dev/docs/example-mxnet-walk-through.html)

## Flower Usage Examples

A number of examples show different usage scenarios of Flower (in combination
with popular machine learning frameworks such as PyTorch or TensorFlow). To run
an example, first install the necessary extras:

[Usage Examples Documentation](https://flower.dev/docs/examples.html)

Quickstart examples:

* [Quickstart (TensorFlow)](https://github.com/adap/flower/tree/main/examples/quickstart_tensorflow)
* [Quickstart (PyTorch)](https://github.com/adap/flower/tree/main/examples/quickstart_pytorch)
* [Quickstart (MXNet)](https://github.com/adap/flower/tree/main/examples/quickstart_mxnet)

Other [examples](https://github.com/adap/flower/tree/main/examples):

* [Raspberry Pi & Nvidia Jetson Tutorial](https://github.com/adap/flower/tree/main/examples/embedded_devices)
* [PyTorch: From Centralized to Federated](https://github.com/adap/flower/tree/main/examples/pytorch_from_centralized_to_federated)
* [MXNet: From Centralized to Federated](https://github.com/adap/flower/tree/main/examples/mxnet_from_centralized_to_federated)
* [Advanced Flower with TensorFlow/Keras](https://github.com/adap/flower/tree/main/examples/advanced_tensorflow)
* [Single-Machine Simulation of Federated Learning Systems](https://github.com/adap/flower/tree/main/examples/simulation)

## Flower Baselines

*Coming soon* - curious minds can take a peek at [src/py/flwr_experimental/baseline](https://github.com/adap/flower/tree/main/src/py/flwr_experimental/baseline).

## Flower Datasets

*Coming soon* - curious minds can take a peek at [src/py/flwr_experimental/baseline/dataset](https://github.com/adap/flower/tree/main/src/py/flwr_experimental/baseline/dataset).

## Citation

If you publish work that uses Flower, please cite Flower as follows: 

```bibtex
@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Parcollet, Titouan and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
```

Please also consider adding your publication to the list of Flower-based publications in the docs, just open a Pull Request.

## Contributing to Flower

We welcome contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) to get
started!
