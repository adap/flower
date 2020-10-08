# Flower (`flwr`) - A Friendly Federated Learning Research Framework

[![GitHub license](https://img.shields.io/github/license/adap/flower)](https://github.com/adap/flower/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/adap/flower/blob/main/CONTRIBUTING.md)
![Build](https://github.com/adap/flower/workflows/Build/badge.svg)

Flower (`flwr`) is a research framework for building federated learning systems. The
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

## Documentation

* [Documentation](https://flower.dev)
  * [Quickstart](https://flower.dev/quickstart.html)
  * [Installation](https://flower.dev/installation.html)

## Flower Usage Examples

A number of examples show different usage scenarios of Flower (in combination
with popular machine learning frameworks such as PyTorch or TensorFlow). To run
an example, first install the necessary extras:

[Usage Examples Documentation](https://flower.dev/examples.html)

Available [examples](src/py/flwr_example):

* [Quickstart: Keras & MNIST](src/py/flwr_example/quickstart)
* [PyTorch & CIFAR-10](src/py/flwr_example/pytorch)
* [PyTorch & ImageNet](src/py/flwr_example/pytorch_imagenet)
* [TensorFlow & Fashion-MNIST](src/py/flwr_example/tensorflow)

## Flower Baselines

*Coming soon* - curious minds can take a peek at [src/py/flwr_experimental/baseline](src/py/flwr_experimental/baseline).

## Flower Datasets

*Coming soon* - curious minds can take a peek at [src/py/flwr_experimental/baseline/dataset](src/py/flwr_experimental/baseline/dataset).

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
