---
title: Federated Learning with Label Distribution Skew via Logits Calibration 
url: https://arxiv.org/abs/2209.00189
labels: [data heterogeneity, image classification] 
dataset: [cifar10] 
---

> [!IMPORTANT]
> This is the template for your `README.md`. Please fill-in the information in all areas with a :warning: symbol.
> Please refer to the [Flower Baselines contribution](https://flower.ai/docs/baselines/how-to-contribute-baselines.html) and [Flower Baselines usage](https://flower.ai/docs/baselines/how-to-use-baselines.html) guides for more details.
> Please complete the metadata section at the very top of this README. This generates a table at the top of the file that will facilitate indexing baselines.
> Please remove this [!IMPORTANT] block once you are done with your `README.md` as well as all the `:warning:` symbols and the comments next to them.

> [!IMPORTANT]
> To help having all baselines similarly formatted and structured, we have included two scripts in `baselines/dev` that when run will format your code and run some tests checking if it's formatted.
> These checks use standard packages such as `isort`, `black`, `pylint` and others. You as a baseline creator will need to install additional pacakges. These are already specified in the `pyproject.toml` of
> your baseline. Follow these steps:

```bash
# Create a python env
pyenv virtualenv 3.10.14 fedlc

# Activate it
pyenv activate fedlc

# Install project including developer packages
# Note the `-e` this means you install it in editable mode
# so even if you change the code you don't need to do `pip install`
# again. However, if you add a new dependency to `pyproject.toml` you
# will need to re-run the command below
pip install -e ".[dev]"

# Even without modifying or adding new code, you can run your baseline
# with the placeholder code generated when you did `flwr new`. If you
# want to test this to familiarise yourself with how flower apps are
# executed, execute this from the directory where you `pyproject.toml` is:
flwr run .

# At anypoint during the process of creating your baseline you can
# run the formatting script. For this do:
cd .. # so you are in the `flower/baselines` directory

# Run the formatting script (it will auto-correct issues if possible)
./dev/format-baseline.sh fedlc

# Then, if the above is all good, run the tests.
./dev/test-baseline.sh fedlc
```

# Federated Learning with Label Distribution Skew via Logits Calibration

> [!NOTE]
> If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** https://arxiv.org/abs/2209.00189

**Authors:** Jie Zhang, Zhiqi Li, Bo Li, Jianghe Xu, Shuang Wu, Shouhong Ding, Chao Wu

**Abstract:** Traditional federated optimization methods perform poorly with heterogeneous data (ie, accuracy reduction), especially for highly skewed data. In this paper, we investigate the label distribution skew in FL, where the distribution of labels varies across clients. First, we investigate the label distribution skew from a statistical view. We demonstrate both theoretically and empirically that previous methods based on softmax cross-entropy are not suitable, which can result in local models heavily overfitting to minority classes and missing classes. Additionally, we theoretically introduce a deviation bound to measure the deviation of the gradient after local update. At last, we propose FedLC (\textbf {Fed} erated learning via\textbf {L} ogits\textbf {C} alibration), which calibrates the logits before softmax cross-entropy according to the probability of occurrence of each class. FedLC applies a fine-grained calibrated cross-entropy loss to local update by adding a pairwise label margin. Extensive experiments on federated datasets and real-world datasets demonstrate that FedLC leads to a more accurate global model and much improved performance. Furthermore, integrating other FL methods into our approach can further enhance the performance of the global model.

## About this baseline

**What’s implemented:** This repo contains an implementation for FedLC (Federated Learning with Logits Correction) introduced by the paper. It also contains code to replicate Table 1 for CIFAR10 and CIFAR100 datasets, Table 5 for CIFAR10 showing performance of FedLC for different local epochs and Table 6 showing test accuracy with different number of clients.

**Datasets:** CIFAR10

**Hardware Setup:** The paper uses 8x V100 GPU set up. However, due to resource constraints, the experiments were run with 1xRTX3090 GPU. FedLC on CIFAR10 with 20 clients and 400 rounds took __ seconds. 

**Contributors:** [@flydump](https://github.com/flydump)

## Experimental Setup

**Task:** Image classification

**Model:** ResNet-18

**Dataset:** :warning: _*Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table.*_

**Training Hyperparameters:** :warning: _*Include a table with all the main hyperparameters in your baseline. Please show them with their default value.*_

## Environment Setup

:warning: _Specify the steps to create and activate your environment and install the baseline project. Most baselines are expected to require minimal steps as shown below. These instructions should be comprehensive enough so anyone can run them (if non standard, describe them step-by-step)._


```bash
# Create the virtual environment
pyenv virtualenv 3.10.14 <name-of-your-baseline-env>

# Activate it
pyenv activate <name-of-your-baseline-env>

# Install the baseline
pip install -e .
```

## Running the Experiments

:warning: _Make sure you have adjusted the `client-resources` in the federation in `pyproject.toml` so your simulation makes the best use of the system resources available._

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._

:warning: _You might want to add more hyperparameters and settings for your baseline. You can do so by extending `[tool.flwr.app.config]` in `pyproject.toml`. In addition, you can create a new `.toml` file that can be passed with the `--run-config` command (see below an example) to override several config values **already present** in `pyproject.toml`._

```bash
# it is likely that for one experiment you need to override some arguments.
flwr run . --run-config learning-rate=0.1,coefficient=0.123

# or you might want to load different `.toml` configs all together:
flwr run . --run-config <my-big-experiment-config>.toml
```

:warning: _It is preferable to show a single commmand (or multilple commands if they belong to the same experiment) and then a table/plot with the expected results, instead of showing all the commands first and then all the results/plots._
:warning: _If you present plots or other figures, please include either a Jupyter notebook showing how to create them or include a utility function that can be called after the experiments finish running._
:warning: If you include plots or figures, save them in `.png` format and place them in a new directory named `_static` at the same level as your `README.md`.
