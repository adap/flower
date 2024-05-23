# Flower: A Friendly Federated Learning Framework

<p align="center">
  <a href="https://flower.ai/">
    <img src="https://flower.ai/_next/image/?url=%2F_next%2Fstatic%2Fmedia%2Fflower_white_border.c2012e70.png&w=640&q=75" width="140px" alt="Flower Website" />
  </a>
</p>
<p align="center">
    <a href="https://flower.ai/">Website</a> |
    <a href="https://flower.ai/blog">Blog</a> |
    <a href="https://flower.ai/docs/">Docs</a> |
    <a href="https://flower.ai/conf/flower-summit-2022">Conference</a> |
    <a href="https://flower.ai/join-slack">Slack</a>
    <br /><br />
</p>

[![GitHub license](https://img.shields.io/github/license/adap/flower)](https://github.com/adap/flower/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/adap/flower/blob/main/CONTRIBUTING.md)
![Build](https://github.com/adap/flower/actions/workflows/framework.yml/badge.svg)
[![Downloads](https://static.pepy.tech/badge/flwr)](https://pepy.tech/project/flwr)
[![Slack](https://img.shields.io/badge/Chat-Slack-red)](https://flower.ai/join-slack)

Flower (`flwr`) is a framework for building federated learning systems. The
design of Flower is based on a few guiding principles:

- **Customizable**: Federated learning systems vary wildly from one use case to
  another. Flower allows for a wide range of different configurations depending
  on the needs of each individual use case.

- **Extendable**: Flower originated from a research project at the University of
  Oxford, so it was built with AI research in mind. Many components can be
  extended and overridden to build new state-of-the-art systems.

- **Framework-agnostic**: Different machine learning frameworks have different
  strengths. Flower can be used with any machine learning framework, for
  example, [PyTorch](https://pytorch.org), [TensorFlow](https://tensorflow.org), [Hugging Face Transformers](https://huggingface.co/), [PyTorch Lightning](https://pytorchlightning.ai/), [scikit-learn](https://scikit-learn.org/), [JAX](https://jax.readthedocs.io/), [TFLite](https://tensorflow.org/lite/), [MONAI](https://docs.monai.io/en/latest/index.html), [fastai](https://www.fast.ai/), [MLX](https://ml-explore.github.io/mlx/build/html/index.html), [XGBoost](https://xgboost.readthedocs.io/en/stable/), [Pandas](https://pandas.pydata.org/) for federated analytics, or even raw [NumPy](https://numpy.org/)
  for users who enjoy computing gradients by hand.

- **Understandable**: Flower is written with maintainability in mind. The
  community is encouraged to both read and contribute to the codebase.

Meet the Flower community on [flower.ai](https://flower.ai)!

## Federated Learning Tutorial

Flower's goal is to make federated learning accessible to everyone. This series of tutorials introduces the fundamentals of federated learning and how to implement them in Flower.

0. **What is Federated Learning?**

   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adap/flower/blob/main/doc/source/tutorial-series-what-is-federated-learning.ipynb) (or open the [Jupyter Notebook](https://github.com/adap/flower/blob/main/doc/source/tutorial-series-what-is-federated-learning.ipynb))

1. **An Introduction to Federated Learning**

   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adap/flower/blob/main/doc/source/tutorial-series-get-started-with-flower-pytorch.ipynb) (or open the [Jupyter Notebook](https://github.com/adap/flower/blob/main/doc/source/tutorial-series-get-started-with-flower-pytorch.ipynb))

2. **Using Strategies in Federated Learning**

   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adap/flower/blob/main/doc/source/tutorial-series-use-a-federated-learning-strategy-pytorch.ipynb) (or open the [Jupyter Notebook](https://github.com/adap/flower/blob/main/doc/source/tutorial-series-use-a-federated-learning-strategy-pytorch.ipynb))

3. **Building Strategies for Federated Learning**

   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adap/flower/blob/main/doc/source/tutorial-series-build-a-strategy-from-scratch-pytorch.ipynb) (or open the [Jupyter Notebook](https://github.com/adap/flower/blob/main/doc/source/tutorial-series-build-a-strategy-from-scratch-pytorch.ipynb))

4. **Custom Clients for Federated Learning**

   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adap/flower/blob/main/doc/source/tutorial-series-customize-the-client-pytorch.ipynb) (or open the [Jupyter Notebook](https://github.com/adap/flower/blob/main/doc/source/tutorial-series-customize-the-client-pytorch.ipynb))

Stay tuned, more tutorials are coming soon. Topics include **Privacy and Security in Federated Learning**, and **Scaling Federated Learning**.

## 30-Minute Federated Learning Tutorial

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adap/flower/blob/main/examples/flower-in-30-minutes/tutorial.ipynb) (or open the [Jupyter Notebook](https://github.com/adap/flower/blob/main/examples/flower-in-30-minutes/tutorial.ipynb))

## Documentation

[Flower Docs](https://flower.ai/docs):

- [Installation](https://flower.ai/docs/framework/how-to-install-flower.html)
- [Quickstart (TensorFlow)](https://flower.ai/docs/framework/tutorial-quickstart-tensorflow.html)
- [Quickstart (PyTorch)](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)
- [Quickstart (Hugging Face)](https://flower.ai/docs/framework/tutorial-quickstart-huggingface.html)
- [Quickstart (PyTorch Lightning)](https://flower.ai/docs/framework/tutorial-quickstart-pytorch-lightning.html)
- [Quickstart (Pandas)](https://flower.ai/docs/framework/tutorial-quickstart-pandas.html)
- [Quickstart (fastai)](https://flower.ai/docs/framework/tutorial-quickstart-fastai.html)
- [Quickstart (JAX)](https://flower.ai/docs/framework/tutorial-quickstart-jax.html)
- [Quickstart (scikit-learn)](https://flower.ai/docs/framework/tutorial-quickstart-scikitlearn.html)
- [Quickstart (Android [TFLite])](https://flower.ai/docs/framework/tutorial-quickstart-android.html)
- [Quickstart (iOS [CoreML])](https://flower.ai/docs/framework/tutorial-quickstart-ios.html)

## Flower Baselines

Flower Baselines is a collection of community-contributed projects that reproduce the experiments performed in popular federated learning publications. Researchers can build on Flower Baselines to quickly evaluate new ideas. The Flower community loves contributions! Make your work more visible and enable others to build on it by contributing it as a baseline!

- [DASHA](https://github.com/adap/flower/tree/main/baselines/dasha)
- [DepthFL](https://github.com/adap/flower/tree/main/baselines/depthfl)
- [FedBN](https://github.com/adap/flower/tree/main/baselines/fedbn)
- [FedMeta](https://github.com/adap/flower/tree/main/baselines/fedmeta)
- [FedMLB](https://github.com/adap/flower/tree/main/baselines/fedmlb)
- [FedPer](https://github.com/adap/flower/tree/main/baselines/fedper)
- [FedProx](https://github.com/adap/flower/tree/main/baselines/fedprox)
- [FedNova](https://github.com/adap/flower/tree/main/baselines/fednova)
- [HeteroFL](https://github.com/adap/flower/tree/main/baselines/heterofl)
- [FedAvgM](https://github.com/adap/flower/tree/main/baselines/fedavgm)
- [FedStar](https://github.com/adap/flower/tree/main/baselines/fedstar)
- [FedWav2vec2](https://github.com/adap/flower/tree/main/baselines/fedwav2vec2)
- [FjORD](https://github.com/adap/flower/tree/main/baselines/fjord)
- [MOON](https://github.com/adap/flower/tree/main/baselines/moon)
- [niid-Bench](https://github.com/adap/flower/tree/main/baselines/niid_bench)
- [TAMUNA](https://github.com/adap/flower/tree/main/baselines/tamuna)
- [FedVSSL](https://github.com/adap/flower/tree/main/baselines/fedvssl)
- [FedXGBoost](https://github.com/adap/flower/tree/main/baselines/hfedxgboost)
- [FedPara](https://github.com/adap/flower/tree/main/baselines/fedpara)
- [FedAvg](https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/fedavg_mnist)
- [FedOpt](https://github.com/adap/flower/tree/main/baselines/flwr_baselines/flwr_baselines/publications/adaptive_federated_optimization)

Please refer to the [Flower Baselines Documentation](https://flower.ai/docs/baselines/) for a detailed categorization of baselines and for additional info including:
* [How to use Flower Baselines](https://flower.ai/docs/baselines/how-to-use-baselines.html)
* [How to contribute a new Flower Baseline](https://flower.ai/docs/baselines/how-to-contribute-baselines.html)

## Flower Usage Examples

Several code examples show different usage scenarios of Flower (in combination with popular machine learning frameworks such as PyTorch or TensorFlow).

Quickstart examples:

- [Quickstart (TensorFlow)](https://github.com/adap/flower/tree/main/examples/quickstart-tensorflow)
- [Quickstart (PyTorch)](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch)
- [Quickstart (Hugging Face)](https://github.com/adap/flower/tree/main/examples/quickstart-huggingface)
- [Quickstart (PyTorch Lightning)](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch-lightning)
- [Quickstart (fastai)](https://github.com/adap/flower/tree/main/examples/quickstart-fastai)
- [Quickstart (Pandas)](https://github.com/adap/flower/tree/main/examples/quickstart-pandas)
- [Quickstart (JAX)](https://github.com/adap/flower/tree/main/examples/quickstart-jax)
- [Quickstart (MONAI)](https://github.com/adap/flower/tree/main/examples/quickstart-monai)
- [Quickstart (scikit-learn)](https://github.com/adap/flower/tree/main/examples/sklearn-logreg-mnist)
- [Quickstart (Android [TFLite])](https://github.com/adap/flower/tree/main/examples/android)
- [Quickstart (iOS [CoreML])](https://github.com/adap/flower/tree/main/examples/ios)
- [Quickstart (MLX)](https://github.com/adap/flower/tree/main/examples/quickstart-mlx)
- [Quickstart (XGBoost)](https://github.com/adap/flower/tree/main/examples/xgboost-quickstart)

Other [examples](https://github.com/adap/flower/tree/main/examples):

- [Raspberry Pi & Nvidia Jetson Tutorial](https://github.com/adap/flower/tree/main/examples/embedded-devices)
- [PyTorch: From Centralized to Federated](https://github.com/adap/flower/tree/main/examples/pytorch-from-centralized-to-federated)
- [Vertical FL](https://github.com/adap/flower/tree/main/examples/vertical-fl)
- [Federated Finetuning of OpenAI's Whisper](https://github.com/adap/flower/tree/main/examples/whisper-federated-finetuning)
- [Federated Finetuning of Large Language Model](https://github.com/adap/flower/tree/main/examples/llm-flowertune)
- [Federated Finetuning of a Vision Transformer](https://github.com/adap/flower/tree/main/examples/vit-finetune)
- [Advanced Flower with TensorFlow/Keras](https://github.com/adap/flower/tree/main/examples/advanced-tensorflow)
- [Advanced Flower with PyTorch](https://github.com/adap/flower/tree/main/examples/advanced-pytorch)
- Single-Machine Simulation of Federated Learning Systems ([PyTorch](https://github.com/adap/flower/tree/main/examples/simulation-pytorch)) ([Tensorflow](https://github.com/adap/flower/tree/main/examples/simulation-tensorflow))
- [Comprehensive Flower+XGBoost](https://github.com/adap/flower/tree/main/examples/xgboost-comprehensive)
- [Flower through Docker Compose and with Grafana dashboard](https://github.com/adap/flower/tree/main/examples/flower-via-docker-compose)
- [Flower with KaplanMeierFitter from the lifelines library](https://github.com/adap/flower/tree/main/examples/federated-kaplna-meier-fitter)
- [Sample Level Privacy with Opacus](https://github.com/adap/flower/tree/main/examples/opacus)
- [Sample Level Privacy with TensorFlow-Privacy](https://github.com/adap/flower/tree/main/examples/tensorflow-privacy)

## Community

Flower is built by a wonderful community of researchers and engineers. [Join Slack](https://flower.ai/join-slack) to meet them, [contributions](#contributing-to-flower) are welcome.

<a href="https://github.com/adap/flower/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=adap/flower" />
</a>

## Citation

If you publish work that uses Flower, please cite Flower as follows:

```bibtex
@article{beutel2020flower,
  title={Flower: A Friendly Federated Learning Research Framework},
  author={Beutel, Daniel J and Topal, Taner and Mathur, Akhil and Qiu, Xinchi and Fernandez-Marques, Javier and Gao, Yan and Sani, Lorenzo and Kwing, Hei Li and Parcollet, Titouan and Gusmão, Pedro PB de and Lane, Nicholas D},
  journal={arXiv preprint arXiv:2007.14390},
  year={2020}
}
```

Please also consider adding your publication to the list of Flower-based publications in the docs, just open a Pull Request.

## Contributing to Flower

We welcome contributions. Please see [CONTRIBUTING.md](CONTRIBUTING.md) to get started!
